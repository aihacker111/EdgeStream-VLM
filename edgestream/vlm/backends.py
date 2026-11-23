"""VLM backends.

* ``openai``       : any OpenAI-compatible server: vLLM, SGLang, llama.cpp server, LM Studio,
                     Ollama (/v1). Sends a JSON schema for constrained decoding and asks for
                     token logprobs to compute the verdict confidence.
* ``transformers`` : local Hugging Face model (Qwen3-VL family by default).
* ``simulated``    : NO model. A deterministic stand-in that answers from the candidate's own
                     features, so the pipeline, budget accounting and logs can be tested end to
                     end without a GPU. Its verdicts are NOT VLM judgements.
"""
from __future__ import annotations

import base64
import json
import math
import re
import time
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from .cards import output_schema
from .parse import extract_json, normalize
from .prompt import VLMRequest
from ..types import Verdict


@dataclass
class ChatResult:
    """Generic answer of :meth:`VLMBackend.chat`: text plus, per generated token, its top
    alternatives as {token: logprob}. Used by the RMOT verifiers to read P(yes) per field."""
    text: str
    tokens: list                   # [(token_str, {alt_token_str: logprob})]
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


def first_token_yes_probability(tokens: list, text: str = "") -> float | None:
    """P(yes) / (P(yes) + P(no)) from the alternatives of the first generated token (one-token
    yes/no answers). Falls back to the decoded text (1.0 / 0.0); None if neither is readable."""
    if tokens:
        tok, alts = tokens[0]
        py = pn = 0.0
        for a, lp in (alts or {tok: 0.0}).items():
            w = a.strip().strip('"').strip(".").lower()
            if w in ("yes", "y", "ye"):
                py += math.exp(lp)
            elif w in ("no", "n"):
                pn += math.exp(lp)
        if py + pn > 0:
            return py / (py + pn)
    w = (text or "".join(t for t, _ in tokens or [])).strip().strip('"').strip(".").lower()
    if w.startswith("yes"):
        return 1.0
    if w.startswith("no"):
        return 0.0
    return None


def field_yes_probability(tokens: list, keys: list, yes: str = "yes", no: str = "no") -> dict:
    """P(yes) / (P(yes) + P(no)) at the value of each JSON field ``"<key>": "yes|no"``.

    Works whether the tokenizer emits the opening quote of the value separately or glued to the
    value (``"yes``). Fields without readable alternatives are filled from the decoded text
    (1.0 / 0.0) so the result always has every key the model answered.
    """
    want = {str(k) for k in keys}
    out: dict = {}
    acc = ""
    pat_q = re.compile(r'"([^"]+)"\s*:\s*"$')
    pat_n = re.compile(r'"([^"]+)"\s*:\s*$')
    for tok, alts in tokens:
        m = pat_q.search(acc) or pat_n.search(acc)
        if m and m.group(1) in want and m.group(1) not in out:
            py = pn = 0.0
            for a, lp in (alts or {tok: 0.0}).items():
                w = a.strip().strip('"').strip().lower()
                if not w:
                    continue
                if yes.startswith(w) or w.startswith(yes):
                    py += math.exp(lp)
                elif no.startswith(w) or w.startswith(no):
                    pn += math.exp(lp)
            if py + pn > 0:
                out[m.group(1)] = py / (py + pn)
        acc += tok
    obj = extract_json(acc) or {}
    for k in want:
        if k not in out and isinstance(obj.get(k), str):
            v = obj[k].strip().lower()
            if v in (yes, no):
                out[k] = 1.0 if v == yes else 0.0
    return out


@dataclass
class RawResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    confidence: float | None = None


class VLMBackend:
    name = "base"

    def generate(self, req: VLMRequest) -> RawResult:  # pragma: no cover - interface
        raise NotImplementedError

    def chat(self, system: str, parts: list, max_new_tokens: int = 128, schema: dict | None = None,
             top_logprobs: int = 5) -> ChatResult:  # pragma: no cover - interface
        """Free-form request: ``parts`` = [("text", str) | ("image", BGR ndarray)]."""
        raise NotImplementedError(f"{self.name} backend has no generic chat()")

    def __call__(self, req: VLMRequest) -> Verdict:
        raw = self.generate(req)
        obj, ok = normalize(extract_json(raw.text), req.card.id, req.card.severity)
        return Verdict(cand_id=req.cand.cand_id, verdict=obj["verdict"], event_type=obj["event_type"],
                       severity=obj["severity"], description=obj["description"],
                       confidence=raw.confidence, parse_ok=ok, raw_text=raw.text, backend=self.name,
                       latency_s=raw.latency_s, prompt_tokens=raw.prompt_tokens,
                       completion_tokens=raw.completion_tokens, visual_tokens=req.visual_tokens)


def _approx_text_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3) + 1


# ---------------------------------------------------------------------- simulated
class SimulatedBackend(VLMBackend):
    """Stand-in for plumbing tests. Answers 'yes' when the cheap-tier evidence is strong."""
    name = "simulated"

    def __init__(self, yes_threshold: float = 0.75, min_duration_s: float = 2.0):
        self.yes_threshold = yes_threshold
        self.min_duration_s = min_duration_s

    def generate(self, req: VLMRequest) -> RawResult:
        c = req.cand
        score = 0.5 * c.s_rule + 0.3 * c.s_anom + 0.2 * c.quality - 0.5 * c.recovered_frac
        if c.duration < self.min_duration_s:
            verdict = "uncertain"
        else:
            verdict = "yes" if score >= self.yes_threshold else "no"
        conf = 1.0 / (1.0 + math.exp(-8 * (score - self.yes_threshold)))
        out = {"verdict": verdict,
               "event_type": req.meta.get("default_event", c.type_hint) if verdict != "no" else "none",
               "severity": req.card.severity,
               "description": f"[simulated] {c.type_hint} score={score:.2f} dur={c.duration:.1f}s"}
        text = json.dumps(out)
        return RawResult(text, _approx_text_tokens(req.system + req.text) + req.visual_tokens,
                         _approx_text_tokens(text), 0.0, conf)   # confidence = P(yes), like the logprob backends


# ---------------------------------------------------------------------- OpenAI-compatible
def _jpeg_data_url(img: np.ndarray, quality: int = 85) -> str:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()


def verdict_confidence_from_logprobs(content: list) -> float | None:
    """P(yes) / (P(yes)+P(no)+P(uncertain)) at the token where the verdict value starts."""
    acc = ""
    for item in content or []:
        if re.search(r'"verdict"\s*:\s*"$', acc):
            probs = {"yes": 0.0, "no": 0.0, "uncertain": 0.0}
            for alt in item.get("top_logprobs", []) or [item]:
                tok = alt.get("token", "").strip().lower()
                for k in probs:
                    if tok and k.startswith(tok):
                        probs[k] += math.exp(alt.get("logprob", -1e9))
                        break
            z = sum(probs.values())
            return probs["yes"] / z if z > 0 else None
        acc += item.get("token", "")
    return None


class OpenAICompatBackend(VLMBackend):
    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout_s: float = 120.0,
                 use_schema: bool = True, logprobs: bool = True, temperature: float = 0.0,
                 chat_template_kwargs: dict | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.timeout = model, api_key, timeout_s
        self.use_schema, self.logprobs, self.temperature = use_schema, logprobs, temperature
        # Qwen3.5 / hybrid-thinking models: no reasoning tokens (templates without the flag ignore it)
        self.chat_template_kwargs = {"enable_thinking": False} if chat_template_kwargs is None else chat_template_kwargs

    def _payload(self, req: VLMRequest) -> dict:
        content = []
        for kind, val in req.parts:
            if kind == "text":
                content.append({"type": "text", "text": val})
            else:
                content.append({"type": "image_url", "image_url": {"url": _jpeg_data_url(val)}})
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": req.max_new_tokens,
                "messages": [{"role": "system", "content": req.system},
                             {"role": "user", "content": content}]}
        if self.use_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "verdict", "schema": output_schema(list(req.meta.get("card_ids", [req.card.id]))),
                "strict": True}}
        if self.logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 5
        return body

    def _post(self, payload: dict) -> dict:
        r = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(r, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def chat(self, system: str, parts: list, max_new_tokens: int = 128, schema: dict | None = None,
             top_logprobs: int = 5) -> ChatResult:
        import urllib.error
        content = [{"type": "text", "text": v} if k == "text" else
                   {"type": "image_url", "image_url": {"url": _jpeg_data_url(v)}} for k, v in parts]
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": max_new_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        if self.chat_template_kwargs:
            body["chat_template_kwargs"] = self.chat_template_kwargs
        t0 = time.perf_counter()
        while True:
            b = dict(body)
            if schema is not None and self.use_schema:
                b["response_format"] = {"type": "json_schema",
                                        "json_schema": {"name": "answer", "schema": schema, "strict": True}}
            if self.logprobs:
                b["logprobs"], b["top_logprobs"] = True, top_logprobs
            try:
                data = self._post(b)
                break
            except urllib.error.HTTPError as e:
                if e.code in (400, 422) and self.logprobs:
                    self.logprobs = False
                elif e.code in (400, 422) and self.use_schema and schema is not None:
                    self.use_schema = False
                else:
                    raise
        dt = time.perf_counter() - t0
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        toks = []
        for it in (choice.get("logprobs") or {}).get("content") or []:
            alts = {a.get("token", ""): a.get("logprob", -1e9) for a in it.get("top_logprobs") or []}
            alts.setdefault(it.get("token", ""), it.get("logprob", 0.0))
            toks.append((it.get("token", ""), alts))
        if not toks:
            toks = [(text, {})]
        usage = data.get("usage", {})
        return ChatResult(text, toks, int(usage.get("prompt_tokens", 0)),
                          int(usage.get("completion_tokens", 0)), dt)

    def generate(self, req: VLMRequest) -> RawResult:
        import urllib.error
        t0 = time.perf_counter()
        while True:
            try:
                data = self._post(self._payload(req))
                break
            except urllib.error.HTTPError as e:
                # some servers reject logprobs or json_schema: degrade once per feature, then give up
                if e.code in (400, 422) and self.logprobs:
                    self.logprobs = False
                elif e.code in (400, 422) and self.use_schema:
                    self.use_schema = False
                else:
                    raise
        dt = time.perf_counter() - t0
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        usage = data.get("usage", {})
        conf = verdict_confidence_from_logprobs((choice.get("logprobs") or {}).get("content"))
        return RawResult(text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)),
                         dt, conf)


# ---------------------------------------------------------------------- transformers
def template_kwargs(processor) -> dict:
    """``enable_thinking=False`` for chat templates that read it (Qwen3 / Qwen3.5 hybrids), nothing
    otherwise: an unknown template kwarg makes transformers log a warning on every call."""
    t = getattr(processor, "chat_template", None)
    if isinstance(t, dict):
        t = " ".join(str(v) for v in t.values())
    if not t:
        t = getattr(getattr(processor, "tokenizer", None), "chat_template", "") or ""
    return {"enable_thinking": False} if "enable_thinking" in str(t) else {}


class TransformersBackend(VLMBackend):
    """Local Hugging Face model, no server, e.g. ``Qwen/Qwen3-VL-2B-Instruct`` (+ a LoRA adapter).

    Runs on CUDA, Apple-silicon MPS or CPU (``device: auto`` picks in that order). Needs
    transformers >= 4.57 (Qwen3-VL).

    One-token yes/no requests (prompt v2: ``max_new_tokens=1``, no schema) never call generate():
    they are ONE forward pass reading the logits of the first answer token, and requests made at
    the same time from several threads (the verifier pool) are grouped into one padded batch by a
    single model thread, so the GPU sees batches like a vLLM server would. Every request of a batch
    reports the batch's wall time as its latency (they ran concurrently). Longer generations
    (JSON prompts) take the generate() path, one at a time.

    ``adapter``: a PEFT/LoRA folder from scripts/rmot_train_lora.py; merged into the weights at load.
    """
    name = "transformers"

    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto",
                 attn_implementation: str | None = None, adapter: str | None = None,
                 max_batch: int = 8, batch_wait_ms: float = 4.0):
        import threading
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        if dtype == "auto":
            dtype = {"cuda": torch.bfloat16, "mps": torch.bfloat16}.get(device.split(":")[0], torch.float32)
        elif isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.device = torch.device(device)
        kw = {"dtype": dtype}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        self.processor = AutoProcessor.from_pretrained(model_id)
        self._tkw = template_kwargs(self.processor)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
        except TypeError:                                   # transformers < 4.56 calls it torch_dtype
            kw["torch_dtype"] = kw.pop("dtype")
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
        if adapter:
            from peft import PeftModel                      # merged: no LoRA overhead per forward
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model.to(self.device).eval()
        self.model_id = model_id
        self.adapter = adapter
        self.max_batch = max(1, int(max_batch))
        self.batch_wait_s = max(0.0, float(batch_wait_ms)) / 1000.0
        tok = self.processor.tokenizer
        self._answer_ids = {}                               # always reported, even outside the top-k
        for w in ("yes", "Yes", " yes", " Yes", "YES", "no", "No", " no", " No", "NO"):
            ids = tok.encode(w, add_special_tokens=False)
            if ids:
                self._answer_ids.setdefault(ids[0], tok.decode([ids[0]]))
        self._lock = threading.Lock()                       # generate() path and the model thread
        self._cv = threading.Condition()
        self._queue: list = []
        self._thread = None

    # ------------------------------------------------------------ one-token scoring (batched)
    def _messages(self, system: str, parts: list):
        from PIL import Image
        content = [{"type": "text", "text": v} if k == "text" else
                   {"type": "image", "image": Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))}
                   for k, v in parts]
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
        imgs = [c["image"] for c in content if c["type"] == "image"]
        return msgs, imgs

    def _render(self, msgs) -> str:
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                  **self._tkw)

    def score_batch(self, reqs: list, top_logprobs: int = 10) -> list:
        """[(system, parts)] -> [ChatResult] with ONE generated token each, in one forward pass."""
        torch = self.torch
        texts, images = [], []
        for system, parts in reqs:
            msgs, imgs = self._messages(system, parts)
            texts.append(self._render(msgs))
            images.extend(imgs)
        tok = self.processor.tokenizer
        with self._lock:
            old = tok.padding_side
            tok.padding_side = "left"                      # last position = first answer token, every row
            try:
                enc = self.processor(text=texts, images=images or None, return_tensors="pt", padding=True)
            finally:
                tok.padding_side = old
            enc = enc.to(self.device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                try:
                    logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
                except TypeError:
                    logits = self.model(**enc).logits[:, -1, :]
                lp = torch.log_softmax(logits.float(), dim=-1)
                k = max(1, min(int(top_logprobs), lp.shape[-1]))
                v, ix = lp.topk(k, dim=-1)
                ans_ids = list(self._answer_ids)
                ans_lp = lp[:, ans_ids].tolist() if ans_ids else [[] for _ in reqs]
                v, ix = v.tolist(), ix.tolist()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps":
                torch.mps.synchronize()
            dt = time.perf_counter() - t0
        n_in = enc["attention_mask"].sum(-1).tolist()
        out = []
        for r in range(len(reqs)):
            alts = {tok.decode([int(j)]): float(x) for x, j in zip(v[r], ix[r])}
            for j, x in zip(ans_ids, ans_lp[r]):
                alts.setdefault(self._answer_ids[j], float(x))
            first = tok.decode([int(ix[r][0])])
            out.append(ChatResult(first, [(first, alts)], int(n_in[r]), 1, dt))
        return out

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
                if self.batch_wait_s and len(self._queue) < self.max_batch:
                    self._cv.wait(timeout=self.batch_wait_s)   # let concurrent requests join the batch
                batch, self._queue = self._queue[: self.max_batch], self._queue[self.max_batch:]
            try:
                res = self.score_batch([(sys_, parts) for sys_, parts, _, _ in batch], max(t for _, _, t, _ in batch))
                for (_, _, _, slot), r in zip(batch, res):
                    slot["result"] = r
            except Exception as e:                         # every waiter gets the error
                for *_, slot in batch:
                    slot["error"] = e
            for *_, slot in batch:
                slot["done"].set()

    def _score(self, system: str, parts: list, top_logprobs: int) -> ChatResult:
        import threading
        slot = {"done": threading.Event()}
        with self._cv:
            if self._thread is None:
                self._thread = threading.Thread(target=self._worker, name="vlm-batcher", daemon=True)
                self._thread.start()
            self._queue.append((system, parts, top_logprobs, slot))
            self._cv.notify_all()
        slot["done"].wait()
        if "error" in slot:
            raise slot["error"]
        return slot["result"]

    def generate(self, req: VLMRequest) -> RawResult:
        from PIL import Image
        content = []
        for kind, val in req.parts:
            if kind == "text":
                content.append({"type": "text", "text": val})
            else:
                content.append({"type": "image", "image": Image.fromarray(cv2.cvtColor(val, cv2.COLOR_BGR2RGB))})
        messages = [{"role": "system", "content": [{"type": "text", "text": req.system}]},
                    {"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=req.max_new_tokens, do_sample=False,
                                      output_scores=True, return_dict_in_generate=True)
        if self.device.type == "mps":
            self.torch.mps.synchronize()
        dt = time.perf_counter() - t0
        n_in = inputs["input_ids"].shape[1]
        gen = out.sequences[0, n_in:]
        text = self.processor.batch_decode([gen], skip_special_tokens=True)[0]
        conf = self._confidence(gen, out.scores)
        return RawResult(text, int(n_in), int(gen.shape[0]), dt, conf)

    def chat(self, system: str, parts: list, max_new_tokens: int = 128, schema: dict | None = None,
             top_logprobs: int = 5) -> ChatResult:
        if max_new_tokens == 1 and schema is None:          # prompt v2: batched forward pass, no generate
            return self._score(system, parts, top_logprobs)
        messages, _ = self._messages(system, parts)
        with self._lock:
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                        return_dict=True, return_tensors="pt",
                                                        **self._tkw).to(self.device)
            t0 = time.perf_counter()
            with self.torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                          output_scores=True, return_dict_in_generate=True)
            if self.device.type == "mps":
                self.torch.mps.synchronize()
            dt = time.perf_counter() - t0
        n_in = inputs["input_ids"].shape[1]
        gen = out.sequences[0, n_in:].tolist()
        tok = self.processor.tokenizer
        toks = []
        for i, tid in enumerate(gen):
            lp = self.torch.log_softmax(out.scores[i][0].float(), dim=-1)
            v, ix = lp.topk(top_logprobs)
            alts = {tok.decode([int(j)]): float(x) for x, j in zip(v.tolist(), ix.tolist())}
            toks.append((tok.decode([tid]), alts))
        text = self.processor.batch_decode([out.sequences[0, n_in:]], skip_special_tokens=True)[0]
        return ChatResult(text, toks, int(n_in), len(gen), dt)

    def _confidence(self, gen, scores) -> float | None:
        tok = self.processor.tokenizer
        acc = ""
        for i, tid in enumerate(gen.tolist()):
            if re.search(r'"verdict"\s*:\s*"$', acc):
                probs = self.torch.softmax(scores[i][0].float(), dim=-1)
                p = {}
                for k in ("yes", "no", "uncertain"):
                    ids = tok.encode(k, add_special_tokens=False)
                    p[k] = float(probs[ids[0]]) if ids else 0.0
                z = sum(p.values())
                return p["yes"] / z if z > 0 else None
            acc += tok.decode([tid])
        return None


def build_backend(cfg: dict) -> VLMBackend:
    kind = cfg.get("backend", "simulated")
    if kind == "simulated":
        return SimulatedBackend(**cfg.get("simulated", {}))
    if kind == "openai":
        return OpenAICompatBackend(**cfg["openai"])
    if kind == "transformers":
        return TransformersBackend(**cfg["transformers"])
    raise ValueError(f"unknown VLM backend {kind!r}")
