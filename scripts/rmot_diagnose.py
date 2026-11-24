#!/usr/bin/env python
"""How good are the VLM's answers? Every refer answer is compared with the ground truth.

    python scripts/rmot_diagnose.py --dataset aerialmind --root data/AerialMind --split in_domain \
        --seqmap data/AerialMind/seqmap.txt --vlm openai --vlm-url http://localhost:8000/v1 \
        --vlm-model Qwen/Qwen3-VL-2B-Instruct --max-videos 2 --max-expr 5 --out runs/am_diag

For each prompt type (set = several marked candidates in one call, single = one candidate, as
in the keyframe filter) it reports, over all (candidate, answer) pairs:
    n, missing    answers and candidates whose P(yes) could not be read from the output
    P(yes)>0.5    how often the VLM says yes; truth+ = how often the truth is yes
    acc, prec, rec, F1 at the 0.5 threshold
    AUC           ranking quality independent of the threshold (0.5 = chance)
    best thr      the threshold with the best F1 and its F1 (a bias the threshold could fix)
Truth of a candidate = the referred fraction of its frames >= 0.5 (box IoU >= 0.5 with a GT box
of a referred object). Answers are written to <out>/answers.jsonl for further analysis.
Uses the same perception cache and answer cache as rmot_benchmark.py (no extra VLM calls for
prompts already asked by a run with the same --out parent... only identical prompts are reused).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import rmot_benchmark as RB  # noqa: E402
from edgestream.rmot.datasets import load_split  # noqa: E402
from edgestream.rmot.perception import has_cache, load_mot_tracks, load_perception, run_perception  # noqa: E402
from edgestream.rmot.pipeline import INF, GTOracle, OnlineRMOT, RMOTConfig  # noqa: E402
from edgestream.rmot.verifier import CachedVerifier, FrameSource  # noqa: E402


class Recorder:
    """Wraps a verifier; logs every refer answer with the truth of each candidate."""

    def __init__(self, inner, oracle, video, expr, method, sink):
        self.inner, self.oracle, self.video, self.expr, self.method, self.sink = inner, oracle, video, expr, method, sink

    @property
    def parallel(self):
        return getattr(self.inner, "parallel", False)

    @property
    def facts(self):
        return getattr(self.inner, "facts", None)

    @facts.setter
    def facts(self, v):
        self.inner.facts = v

    def refer(self, g, group, anchor_rank, expr, frames, single=False):
        ans = self.inner.refer(g, group, anchor_rank, expr, frames, single=single)
        n = len(group)
        size = "1" if n == 1 else ("2-4" if n <= 4 else "5-8")
        pv = getattr(getattr(self.inner, "inner", self.inner), "prompt", "v1")
        one = single or n == 1
        kind = ((f"one candidate ({pv})" if one else f"set prompt g={size}")
                + (" +facts" if self.facts else " -facts"))
        for s in group:
            p = ans.probs.get(s.sid)
            self.sink.append({"video": self.video, "expr": self.expr, "method": self.method, "kind": kind,
                              "group": len(group), "cls": s.cls_name, "p": p, "truth": self.oracle.refer(s),
                              "n_frames": s.n, "raw": (ans.text or "")[:200]})
        return ans

    def link(self, g, a, b, frames):
        return self.inner.link(g, a, b, frames)


def auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                       # average ranks of ties
        m = s == v
        ranks[m] = ranks[m].mean()
    npos = y.sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def platt(z, y, l2: float = 1.0, iters: int = 100):
    """Fit P(truth) = sigmoid(a * z + b) (z = logit of the VLM's P(yes)) by damped Newton steps with
    a backtracking line search on the L2-penalised log-likelihood: stable even when the VLM's
    probabilities are saturated (many identical z), where plain Newton diverges."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    X = np.stack([z, np.ones_like(z)], 1)

    def loss(w):
        t = X @ w
        return float(np.sum(np.logaddexp(0, t) - y * t) + 0.5 * l2 * w @ w)
    w = np.array([1.0, 0.0])
    f = loss(w)
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (pr - y) + l2 * w
        Hm = (X * (pr * (1 - pr))[:, None]).T @ X + l2 * np.eye(2)
        step = np.linalg.solve(Hm, g)
        t = 1.0
        while t > 1e-6:
            w2 = w - t * step
            f2 = loss(w2)
            if f2 <= f:
                break
            t /= 2
        if f - f2 < 1e-9:
            w = w2 if f2 <= f else w
            break
        w, f = w2, f2
    a, b = float(np.clip(w[0], -10, 10)), float(np.clip(w[1], -20, 20))
    return a, b


def report(rows, title):
    print(f"\n## {title}")
    print("| prompt | n | missing | P(yes)>0.5 | truth+ | acc | prec | rec | F1 | AUC | best thr (F1) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        ok = [r for r in rs if r["p"] is not None]
        miss = 1 - len(ok) / max(len(rs), 1)
        if not ok:
            print(f"| {kind} | {len(rs)} | {miss:.0%} | - | - | - | - | - | - | - | - |")
            continue
        p = np.array([r["p"] for r in ok])
        y = np.array([r["truth"] >= 0.5 for r in ok])

        def prf(th):
            yh = p > th
            tp, fp, fn = (yh & y).sum(), (yh & ~y).sum(), (~yh & y).sum()
            pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            return (yh == y).mean(), pr, rc, 2 * pr * rc / max(pr + rc, 1e-9)
        acc, pr, rc, f1 = prf(0.5)
        ths = np.unique(np.concatenate([[0.5], np.quantile(p, np.linspace(0.02, 0.98, 49))]))
        bt = max(ths, key=lambda t: prf(t)[3])
        print(f"| {kind} | {len(rs)} | {miss:.0%} | {(p > 0.5).mean():.0%} | {y.mean():.0%} | {acc:.2f} | {pr:.2f} | "
              f"{rc:.2f} | {f1:.2f} | {auc(y, p):.3f} | {bt:.2f} ({prf(bt)[3]:.2f}) |")


def probe(a, split, vids, cache_root, fps, cached, ver):
    """Ask every prompt variant on the same fixed candidates (no scheduler, no solver)."""
    import random
    import time
    from concurrent.futures import ThreadPoolExecutor

    from edgestream.rmot.expression import parse
    from edgestream.rmot.tracklets import GraphConfig, TrackletGraph
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    rng = random.Random(0)
    jobs = []                                    # (video, expr, graph, segment, anchor, truth, frames)
    for vid in vids:
        video = split.videos[vid]
        W, Hh = video.image_wh
        if a.tracks_dir:
            perc, aff = load_mot_tracks(os.path.join(a.tracks_dir, f"{vid}.txt"), len(video), (Hh, W), fps)
        else:
            perc, aff = load_perception(cache_root, vid)
        g = TrackletGraph(GraphConfig(fps=fps))
        for r, fo in enumerate(perc):
            if fo is not None:
                g.update(r, r / fps, fo.tracks, (W, Hh), aff[r] if aff else None)
        frames = FrameSource(video.frames, cache=256)
        for expr in split.expressions[vid][: a.max_expr]:
            q = parse(expr.text)
            oracle = GTOracle(video, expr)
            segs = [s for s in g.segs.values() if q.class_ok(s.cls_name) and s.n >= 3]
            lab = [(s, oracle.refer(s)) for s in segs]
            pos = [x for x in lab if x[1] >= 0.5]
            neg = [x for x in lab if x[1] < 0.5]
            rng.shuffle(pos)
            rng.shuffle(neg)
            k = min(len(pos), a.samples // 2)
            pick = pos[:k] + neg[: a.samples - k]
            jobs += [(vid, expr.text, g, s, s.mid().rank, t, frames) for s, t in pick]
    print(f"probe: {len(jobs)} candidates ({sum(j[5] >= 0.5 for j in jobs)} positive) x {len(variants)} variants",
          flush=True)
    rows = []
    pb = RB.bar(len(jobs) * len(variants), "probe", "q")
    for var in variants:
        pv = "v1" if var.startswith("v1") else ("v2m" if var.startswith("v2m") else "v2")
        facts = "+facts" in var
        ver.prompt = pv
        cached.facts = facts

        def ask(j):
            vid, expr, g, s, anchor, truth, frames = j
            ans = cached.refer(g, [s], anchor, expr, frames, single=(pv == "v1"))
            return j, ans
        t0 = time.time()
        by_vid: dict = {}
        for j in jobs:
            by_vid.setdefault(j[0], []).append(j)
        for vid, js in by_vid.items():
            cached.scope = vid
            with ThreadPoolExecutor(max(1, a.threads)) as ex:
                for j, ans in ex.map(ask, js):
                    rows.append({"variant": var, "video": vid, "expr": j[1], "kind": var, "group": 1,
                                 "cls": j[3].cls_name, "p": ans.probs.get(j[3].sid), "truth": j[5],
                                 "latency": ans.latency_s, "prompt_tokens": ans.prompt_tokens,
                                 "raw": (ans.text or "")[:80]})
                    pb.update(1)
        pb.set_postfix_str(f"{var} done in {time.time() - t0:.0f}s")
    pb.close()
    cached.close()
    with open(os.path.join(a.out, "probe_answers.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    report(rows, f"PROBE: same {len(jobs)} candidates for every variant ({len(vids)} videos, "
                 f"{a.max_expr} expressions each)")
    print("\n| variant | mean latency (s) | mean prompt tokens | saturated p (<=.02 or >=.98) |")
    print("|---|---|---|---|")
    for var in variants:
        rs = [r for r in rows if r["variant"] == var and r["p"] is not None]
        if rs:
            sat = np.mean([(r["p"] <= 0.02 or r["p"] >= 0.98) for r in rs])
            print(f"| {var} | {np.mean([r['latency'] for r in rs]):.3f} | {np.mean([r['prompt_tokens'] for r in rs]):.0f} | {sat:.0%} |")
    print("\nPlatt per variant (fit on --split train only):")
    for var in variants:
        rs = [r for r in rows if r["variant"] == var and r["p"] is not None]
        if len(rs) >= 50 and len({r["truth"] >= 0.5 for r in rs}) == 2:
            pz = np.clip(np.array([r["p"] for r in rs]), 0.02, 0.98)
            ab = platt(np.log(pz / (1 - pz)), np.array([float(r["truth"] >= 0.5) for r in rs]))
            print(f"  {var}: [{ab[0]:.3f}, {ab[1]:.3f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aerialmind", choices=["aerialmind", "refer_kitti_v2", "refer_kitti"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="in_domain")
    ap.add_argument("--seqmap", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--methods", default="search,filter_kf")
    ap.add_argument("--vlm", default="openai", choices=["openai", "transformers"])
    ap.add_argument("--vlm-url", default=None)
    ap.add_argument("--vlm-model", default=None)
    ap.add_argument("--vlm-adapter", default=None, help="transformers: LoRA folder (runs/lora_vl2b/best)")
    ap.add_argument("--vlm-batch", type=int, default=None, help="transformers: max requests per forward pass")
    ap.add_argument("--belief-params", default=None, help="YAML from rmot_fit_belief.py (questions as the method asks them)")
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--max-videos", type=int, default=2)
    ap.add_argument("--max-expr", type=int, default=5)
    ap.add_argument("--perception-cache", default=None)
    ap.add_argument("--tracks-dir", default=None, help="external MOT tracks <dir>/<video_id>.txt")
    ap.add_argument("--mode", default="replay", choices=["replay", "probe"],
                    help="replay: run the methods and grade the answers they asked for; probe: ask every "
                         "prompt variant on ONE fixed, class-balanced sample of candidates (apples to apples)")
    ap.add_argument("--variants", default="v2+facts,v2-facts,v2m+facts,v1-facts,v1+facts",
                    help="probe: prompt variants. v2 = one-token shared-prefix prompt, v2m = v2 with the "
                         "candidate boxed on the frame, v1 = the earlier single-candidate prompt (marked "
                         "frame + gallery + JSON); +/-facts = motion text")
    ap.add_argument("--samples", type=int, default=40, help="probe: candidates per (video, expression)")
    ap.add_argument("--threads", type=int, default=8, help="probe: concurrent requests")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg_path = a.config or os.path.join(ROOT, "configs", "rmot_kitti.yaml" if "kitti" in a.dataset else "rmot_aerialmind.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    fps = float(cfg.get("fps", 30))
    base = dict(cfg.get("rmot") or {})
    if a.belief_params:                          # the deployed model: questions chosen as in the benchmark
        with open(a.belief_params) as f:
            base = RB.deep_update(base, (yaml.safe_load(f) or {}).get("rmot") or {})
        base.setdefault("belief", {})["enabled"] = True
    base.setdefault("sched", {})["n_workers"] = a.n_workers
    base["zoom"] = False
    split = load_split(a.dataset, a.root, a.split, a.seqmap, fps=fps)
    vids = list(split.videos)[: a.max_videos]
    if not vids:
        sys.exit(f"no video of {a.dataset}/{a.split} found under {a.root}. If --split train is not "
                 f"recognised, edgestream/rmot/datasets.py on this machine is outdated (sync it).")
    cache_root = a.perception_cache or os.path.join(os.path.dirname(os.path.abspath(a.out)), "perception_cache", a.dataset)
    os.makedirs(a.out, exist_ok=True)
    if not a.tracks_dir:                         # run AERO-MOT once for videos that have no cache yet
        need = [v for v in vids if not has_cache(cache_root, v, len(split.videos[v]))]
        if need:
            from perception import Perceptor
            perceptor = Perceptor.from_yaml(os.path.join(ROOT, cfg["perception_config"]))
            pp = RB.bar(sum(len(split.videos[v]) for v in need), "perception", "frame")
            for i, v in enumerate(need):
                pp.set_postfix_str(f"video {i + 1}/{len(need)} {v}")
                run_perception(split.videos[v].frames, perceptor, cache_root, v, fps, pbar=pp)
            pp.close()
            del perceptor
    args = argparse.Namespace(vlm=a.vlm, vlm_url=a.vlm_url, vlm_model=a.vlm_model,
                              vlm_adapter=a.vlm_adapter, vlm_batch=a.vlm_batch)
    kind, ver = RB.build_vlm(cfg, args)
    cached = CachedVerifier(ver, os.path.join(a.out, f"vlm_cache_{RB.vlm_tag(cfg, args)}.jsonl"))
    if a.mode == "probe":
        return probe(a, split, vids, cache_root, fps, cached, ver)
    rows = []
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    total = sum(min(len(split.expressions[v]), a.max_expr) for v in vids) * len(methods)
    pb = RB.bar(total, "diagnose", "run")
    for vid in vids:
        video = split.videos[vid]
        W, Hh = video.image_wh
        if a.tracks_dir:
            perc, aff = load_mot_tracks(os.path.join(a.tracks_dir, f"{vid}.txt"), len(video), (Hh, W), fps)
        else:
            perc, aff = load_perception(cache_root, vid)
        frames = FrameSource(video.frames, cache=96)
        cached.scope = vid
        for expr in split.expressions[vid][: a.max_expr]:
            oracle = GTOracle(video, expr)
            for m in methods:
                spec = copy.deepcopy(RB.METHODS[m])
                spec.pop("_oracle", None)
                rc = RMOTConfig.from_dict(RB.deep_update(base, {**spec, "lag_s": INF}))
                if m.startswith("filter"):
                    rc.filter_propagate = "nearest"
                rec = Recorder(cached, oracle, vid, expr.text, m, rows)
                pb.set_postfix_str(f"{vid} | {m} | {expr.text[:40]} | {len(rows)} answers, cache hits {cached.hits}")
                OnlineRMOT(rc, rec).run(perc, expr.text, fps, (W, Hh), frames, aff)
                pb.update(1)
    pb.close()
    cached.close()
    with open(os.path.join(a.out, "answers.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    report(rows, f"VLM answers vs ground truth ({len(vids)} videos, {a.max_expr} expressions each)")
    for m in methods:
        report([r for r in rows if r["method"] == m], f"method {m}")
    print("\n## Platt calibration per method. Paste under `rmot:` in configs/rmot_aerialmind.yaml")
    print("## (ONLY values fitted on --split train may be used for reported results)")
    print("  calib_by_method:")
    for m in methods:
        rs = [r for r in rows if r["method"] == m and r["p"] is not None]
        if len(rs) < 50:
            continue
        cal = {}
        for k in ("single", "set"):
            sub = [r for r in rs if (r["group"] == 1) == (k == "single")]
            if len(sub) < 50 or len({r["truth"] >= 0.5 for r in sub}) < 2:
                continue
            pz = np.clip(np.array([r["p"] for r in sub]), 0.02, 0.98)
            z = np.log(pz / (1 - pz))
            y = np.array([float(r["truth"] >= 0.5) for r in sub])
            cal[k] = [round(v, 3) for v in platt(z, y)]
        print(f"    {m}: {json.dumps(cal)}")
    if a.split in ("in_domain", "cross_domain", "test"):
        print("  (fitted on a TEST split: use only to understand the model, not as reported settings)")
    bad = [r for r in rows if r["p"] is None][:3]
    if bad:
        print("\nexamples of unreadable outputs:")
        for r in bad:
            print("  ", repr(r["raw"]))


if __name__ == "__main__":
    main()
