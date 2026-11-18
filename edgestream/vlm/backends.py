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
                 use_schema: bool = True, logprobs: bool = True, temperature: float = 0.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.timeout = model, api_key, timeout_s
        self.use_schema, self.logprobs, self.temperature = use_schema, logprobs, temperature

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
class TransformersBackend(VLMBackend):
    """Local HF model, e.g. ``Qwen/Qwen3-VL-2B-Instruct``. JSON is enforced by parsing only."""
    name = "transformers"

    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device).eval()

    def generate(self, req: VLMRequest) -> RawResult:
        from PIL import Image
        content, images = [], []
        for kind, val in req.parts:
            if kind == "text":
                content.append({"type": "text", "text": val})
            else:
                images.append(Image.fromarray(cv2.cvtColor(val, cv2.COLOR_BGR2RGB)))
                content.append({"type": "image"})
        messages = [{"role": "system", "content": [{"type": "text", "text": req.system}]},
                    {"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to(self.model.device)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=req.max_new_tokens, do_sample=False,
                                      output_scores=True, return_dict_in_generate=True)
        dt = time.perf_counter() - t0
        n_in = inputs["input_ids"].shape[1]
        gen = out.sequences[0, n_in:]
        text = self.processor.batch_decode([gen], skip_special_tokens=True)[0]
        conf = self._confidence(gen, out.scores)
        return RawResult(text, int(n_in), int(gen.shape[0]), dt, conf)

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
