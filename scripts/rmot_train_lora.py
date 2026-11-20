#!/usr/bin/env python
"""LoRA fine-tuning of a Qwen3-VL verifier on data from scripts/rmot_make_sft.py.

The loss is the cross-entropy of the ANSWER tokens only ("yes"/"no" + end of turn); everything
else (system, frame, description, close-ups) is context. The language model gets the LoRA; the
vision tower stays frozen (enough for domain adaptation, cheap on one 24 GB GPU).

    python scripts/rmot_train_lora.py --data data/rmot_sft --model Qwen/Qwen3-VL-2B-Instruct \
        --out runs/lora_vl2b --epochs 1 --batch 4 --grad-accum 4 --lr 1e-4

During training the validation AUC is measured with the same rule as inference: P(yes) from the
first answer token (yes vs no), on held-out VIDEOS. The best adapter is kept in <out>/best.

Serve it with vLLM (the base must be the BF16 model the adapter was trained on):
    vllm serve Qwen/Qwen3-VL-2B-Instruct --enable-lora --lora-modules rmot=<out>/best \\
        --max-lora-rank 16 --enable-prefix-caching ...
and call it with --vlm-model rmot in rmot_benchmark.py / rmot_diagnose.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402


def load_jsonl(path: str, limit: int | None = None, seed: int = 0) -> list:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if limit and len(rows) > limit:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


def auc(y, s) -> float:
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        ranks[m] = ranks[m].mean()
    npos = y.sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


class Batcher:
    """Turns samples into model inputs; labels cover only the assistant answer."""

    def __init__(self, processor, data_dir: str):
        from PIL import Image
        self.proc, self.dir, self.Image = processor, data_dir, Image
        tok = processor.tokenizer
        self.yes_ids = sorted({i for w in ("yes", "Yes", " yes", " Yes") for i in tok.encode(w, add_special_tokens=False)[:1]})
        self.no_ids = sorted({i for w in ("no", "No", " no", " No") for i in tok.encode(w, add_special_tokens=False)[:1]})

    def _msgs(self, rec, with_answer: bool):
        msgs = json.loads(json.dumps(rec["messages"]))
        if not with_answer and msgs[-1]["role"] == "assistant":
            msgs = msgs[:-1]
        imgs = []
        for m in msgs:
            for c in m["content"]:
                if c["type"] == "image":
                    im = self.Image.open(os.path.join(self.dir, c["image"])).convert("RGB")
                    c["image"] = im
                    imgs.append(im)
        return msgs, imgs

    def _render(self, msgs, gen_prompt: bool) -> str:
        kw = dict(tokenize=False, add_generation_prompt=gen_prompt)
        try:
            return self.proc.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            return self.proc.apply_chat_template(msgs, **kw)

    def train_batch(self, recs: list):
        import torch
        texts, images, answers = [], [], []
        for r in recs:
            full_msgs, imgs = self._msgs(r, True)
            prompt = self._render(full_msgs[:-1], True)
            full = self._render(full_msgs, False)
            if not full.startswith(prompt):              # template adds something before the answer
                prompt = full[: full.rfind(r["messages"][-1]["content"][0]["text"])]
            texts.append(full)
            answers.append(full[len(prompt):])
            images.extend(imgs)
        enc = self.proc(text=texts, images=images or None, return_tensors="pt", padding=True)
        labels = torch.full_like(enc["input_ids"], -100)
        tok = self.proc.tokenizer
        for i, ans in enumerate(answers):
            ids = tok.encode(ans, add_special_tokens=False)
            row = enc["input_ids"][i]
            mask = enc["attention_mask"][i].bool()
            pos = torch.nonzero(mask).flatten()
            end = int(pos[-1]) + 1 if tok.padding_side != "left" else len(row)
            n = len(ids)
            if n and row[end - n:end].tolist() == ids:
                labels[i, end - n:end] = row[end - n:end]
            else:                                        # tokenisation at the boundary differs: label the
                want = set(self.yes_ids + self.no_ids)   # last yes/no token and what follows it
                cand = [j for j in range(end) if int(row[j]) in want]
                if cand:
                    labels[i, cand[-1]:end] = row[cand[-1]:end]
        enc["labels"] = labels
        return enc

    def eval_batch(self, recs: list):
        texts, images = [], []
        for r in recs:
            msgs, imgs = self._msgs(r, False)
            texts.append(self._render(msgs, True))
            images.extend(imgs)
        old = self.proc.tokenizer.padding_side
        self.proc.tokenizer.padding_side = "left"        # last position = first answer token for every row
        try:
            enc = self.proc(text=texts, images=images or None, return_tensors="pt", padding=True)
        finally:
            self.proc.tokenizer.padding_side = old
        return enc


def evaluate(model, batcher: Batcher, val: list, device, bs: int = 4) -> dict:
    import torch
    model.eval()
    ps, ys = [], []
    with torch.inference_mode():
        for i in range(0, len(val), bs):
            recs = val[i:i + bs]
            enc = batcher.eval_batch(recs).to(device)
            try:                                         # only the last position: avoids a [B, L, 150k] tensor
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            except TypeError:
                logits = model(**enc).logits[:, -1, :].float()
            ly = torch.logsumexp(logits[:, batcher.yes_ids], -1)
            ln = torch.logsumexp(logits[:, batcher.no_ids], -1)
            ps += torch.sigmoid(ly - ln).tolist()
            ys += [int(r["label"]) for r in recs]
    model.train()
    p, y = np.array(ps), np.array(ys)
    yh = p > 0.5
    tp, fp, fn = (yh & (y == 1)).sum(), (yh & (y == 0)).sum(), (~yh & (y == 1)).sum()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"auc": auc(y, p), "acc": float((yh == y).mean()), "prec": float(prec), "rec": float(rec),
            "f1": float(2 * prec * rec / max(prec + rec, 1e-9)), "yes_rate": float(yh.mean()), "n": len(y)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="output folder of rmot_make_sft.py")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--qlora", action="store_true", help="4-bit base (bitsandbytes), for 4B+ on 24 GB")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=200, help="optimizer steps between validations")
    ap.add_argument("--eval-only", action="store_true", help="only measure the base model (zero-shot AUC)")
    ap.add_argument("--adapter", default=None, help="evaluate / continue from this adapter")
    ap.add_argument("--merge-to", default=None,
                    help="with --adapter: write base+LoRA merged into a plain model folder (vLLM serves it like "
                         "any checkpoint, no --enable-lora; can be quantised to FP8 afterwards)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    transformers.logging.set_verbosity_error()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import rmot_benchmark as RB
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                       ("cpu" if a.device == "auto" else a.device))
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    proc = AutoProcessor.from_pretrained(a.model)
    kw = {}
    if a.qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=dtype, **kw)
    if not a.qlora:
        model.to(dev)
    if a.merge_to:
        from peft import PeftModel
        if not a.adapter or a.qlora:
            sys.exit("--merge-to needs --adapter and a BF16 base (no --qlora)")
        merged = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
        merged.save_pretrained(a.merge_to)
        proc.save_pretrained(a.merge_to)
        print(f"merged model written to {a.merge_to}\n  vllm serve {os.path.abspath(a.merge_to)} --served-model-name rmot ...")
        return
    train = load_jsonl(os.path.join(a.data, "train.jsonl"), a.max_train, a.seed)
    val = load_jsonl(os.path.join(a.data, "val.jsonl"), a.max_val, a.seed)
    batcher = Batcher(proc, a.data)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")

    from peft import LoraConfig, PeftModel, get_peft_model
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter, is_trainable=not a.eval_only)
    if a.eval_only:
        r = evaluate(model, batcher, val, dev, bs=a.batch)
        print(json.dumps({"eval_only": True, "model": a.model, "adapter": a.adapter, **r}, indent=1))
        return
    if not a.adapter:
        targets = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                                 target_modules=targets, task_type="CAUSAL_LM"))
    if dev.type == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"train {len(train)} samples ({sum(r['label'] for r in train)} yes), val {len(val)} "
          f"({sum(r['label'] for r in val)} yes); trainable params {n_tr / 1e6:.1f}M on {dev}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / (a.batch * a.grad_accum))
    total = max(1, int(steps_per_epoch * a.epochs))
    warm = max(1, int(total * a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    base = evaluate(model, batcher, val, dev, bs=a.batch)
    print(f"before training: val {json.dumps(base)}", flush=True)
    log.write(json.dumps({"step": 0, **base}) + "\n")
    best = base["auc"] if base["auc"] == base["auc"] else -1
    model.train()
    step, micro, t0 = 0, 0, time.time()
    order = []
    pb = RB.bar(total, "train", "step")
    run_loss = 0.0
    while step < total:
        if not order:
            order = list(range(len(train)))
            random.shuffle(order)
        idx = [order.pop() for _ in range(min(a.batch, len(order)))]
        enc = batcher.train_batch([train[i] for i in idx]).to(dev)
        out = model(**enc)
        (out.loss / a.grad_accum).backward()
        run_loss += float(out.loss) / a.grad_accum
        micro += 1
        if micro % a.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        pb.set_postfix_str(f"loss {run_loss:.4f}, lr {sched.get_last_lr()[0]:.1e}, best AUC {best:.3f}")
        pb.update(1)
        log.write(json.dumps({"step": step, "loss": run_loss, "lr": sched.get_last_lr()[0]}) + "\n")
        run_loss = 0.0
        if step % a.eval_every == 0 or step == total:
            r = evaluate(model, batcher, val, dev, bs=a.batch)
            log.write(json.dumps({"step": step, **r}) + "\n")
            log.flush()
            pb.write(f"step {step}: val AUC {r['auc']:.3f} acc {r['acc']:.3f} F1 {r['f1']:.3f} yes-rate {r['yes_rate']:.2f}")
            if r["auc"] > best:
                best = r["auc"]
                model.save_pretrained(os.path.join(a.out, "best"))
            model.save_pretrained(os.path.join(a.out, "last"))
    pb.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min; best val AUC {best:.3f} (zero-shot {base['auc']:.3f}); "
          f"adapter: {os.path.join(a.out, 'best')}")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"model": a.model, "zero_shot": base, "best_auc": best, "args": vars(a)}, f, indent=1)
    print("\nmerge (simplest to serve, works with any vLLM version):\n  python scripts/rmot_train_lora.py --data "
          f"{a.data} --model {a.model} --out {a.out} --adapter {os.path.join(a.out, 'best')} --merge-to {a.out}/merged")
    print("\nor serve the adapter directly:\n  vllm serve " + a.model + " --enable-lora --lora-modules rmot=" +
          os.path.abspath(os.path.join(a.out, "best")) + f" --max-lora-rank {a.rank} --enable-prefix-caching "
          "--max-model-len 4096 --max-num-seqs 16 --limit-mm-per-prompt '{\"image\": 2, \"video\": 0}' "
          "--gpu-memory-utilization 0.60 --port 8000\n  then: --vlm-model rmot")


if __name__ == "__main__":
    main()
