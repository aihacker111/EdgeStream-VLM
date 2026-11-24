#!/usr/bin/env python
"""Fit the parameters of the belief model (edgestream/rmot/belief.py) on a TRAIN split.

Everything the method uses at test time is estimated here from ground truth, so nothing is tuned
by hand or on test videos:

    pi_reason[r]   P(same object) across a cut made for reason r          GT identity of the two pieces
    stitch_ab      P(same object | stitch hypothesis) = sigmoid(a * prior_logit + b)   logistic fit
    rho_prior[t]   Beta prior of the base rate of an expression of type t  (static / motion) moments
    hazard[t]      rate (1/s) at which the referred state of one object changes          switch counts
    kin_ab, kin_base  P(referred | measured-motion score), logistic fit, and its base rate
    verifier       Platt calibration (calib.single), its base rate, TPR / FPR           --verifier-scores
                   (val_scores_*.jsonl written by rmot_train_lora.py --eval-only)

The tracker pieces come from the same perception cache (or --tracks-dir) as the benchmark, so the
statistics describe the tracker the method will run on.

    python scripts/rmot_fit_belief.py --root data/AerialMind --split train \\
        --verifier-scores runs/lora_vl2b/val_scores_best.jsonl --out configs/belief_aerialmind.yaml
    python scripts/rmot_benchmark.py ... --belief-params configs/belief_aerialmind.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import rmot_benchmark as RB  # noqa: E402
from edgestream.rmot.belief import query_type  # noqa: E402
from edgestream.rmot.datasets import load_split  # noqa: E402
from edgestream.rmot.expression import parse  # noqa: E402
from edgestream.rmot.hota import box_iou_xywh  # noqa: E402
from edgestream.rmot.perception import has_cache, load_mot_tracks, load_perception, run_perception  # noqa: E402
from edgestream.rmot.tracklets import GraphConfig, TrackletGraph  # noqa: E402


def logistic(x, y, w=None, l2=1e-2, iters=50):
    """1-D logistic regression y ~ sigmoid(a x + b) by damped Newton. Returns (a, b)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    a = b = 0.0
    for _ in range(iters):
        z = np.clip(a * x + b, -30, 30)
        p = 1 / (1 + np.exp(-z))
        r = w * (p - y)
        ga, gb = (r * x).sum() + l2 * a, r.sum()
        h = w * p * (1 - p)
        H = np.array([[(h * x * x).sum() + l2, (h * x).sum()], [(h * x).sum(), h.sum() + 1e-9]])
        try:
            d = np.linalg.solve(H, [ga, gb])
        except np.linalg.LinAlgError:
            break
        step = 1.0
        while step > 1e-3 and np.max(np.abs(step * d)) > 5:
            step *= 0.5
        a, b = a - step * d[0], b - step * d[1]
        if np.max(np.abs(step * d)) < 1e-6:
            break
    return float(a), float(b)


def beta_moments(rates, fallback=(1.0, 3.0), max_strength=10.0):
    """Beta(a, b) with the mean and variance of the per-expression base rates; a + b is capped at
    ``max_strength`` pseudo-pieces so a stream's own answers soon outweigh the prior."""
    r = np.asarray(rates, float)
    if len(r) < 3:
        return list(fallback)
    m, v = float(r.mean()), float(r.var())
    m = min(max(m, 1e-3), 1 - 1e-3)
    v = max(v, 1e-4)
    k = min(m * (1 - m) / v - 1, max_strength)
    if k <= 0:
        return list(fallback)
    return [round(max(m * k, 0.2), 4), round(max((1 - m) * k, 0.2), 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aerialmind", choices=["aerialmind", "refer_kitti_v2", "refer_kitti"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train", help="train (all non-test videos) or a comma list")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tracks-dir", default=None)
    ap.add_argument("--perception-cache", default=None)
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--max-expr", type=int, default=30, help="expressions per video (random subset)")
    ap.add_argument("--verifier-scores", default=None, help="val_scores_*.jsonl from rmot_train_lora.py --eval-only")
    ap.add_argument("--verifier-answers", default=None,
                    help="answers.jsonl from rmot_diagnose.py --mode replay on TRAIN videos: the verifier on the "
                         "questions the method really asks (preferred over --verifier-scores)")
    ap.add_argument("--out", required=True, help="YAML with rmot.belief (+ rmot.calib) to pass as --belief-params")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cfg_path = a.config or os.path.join(ROOT, "configs", "rmot_kitti.yaml" if "kitti" in a.dataset else "rmot_aerialmind.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    fps = float(cfg.get("fps", 30))
    gkw = {k: v for k, v in ((cfg.get("rmot") or {}).get("graph") or {}).items() if k in GraphConfig.__dataclass_fields__}
    split = load_split(a.dataset, a.root, a.split)
    vids = sorted(split.videos)
    rng.shuffle(vids)
    vids = sorted(vids[: a.max_videos]) if a.max_videos else sorted(vids)
    if not vids:
        sys.exit(f"no video in {a.root} for split {a.split}")
    cache_root = a.perception_cache or os.path.join(ROOT, "runs", "perception_cache", a.dataset)
    if not a.tracks_dir:
        need = [v for v in vids if not has_cache(cache_root, v, len(split.videos[v]))]
        if need:
            from perception import Perceptor
            perceptor = Perceptor.from_yaml(os.path.join(ROOT, cfg["perception_config"]))
            pp = RB.bar(sum(len(split.videos[v]) for v in need), "perception", "frame")
            for v in need:
                run_perception(split.videos[v].frames, perceptor, cache_root, v, fps, pbar=pp)
            pp.close()
            del perceptor

    same_by_reason = defaultdict(lambda: [0, 0])          # reason -> [same, known]
    stitch_x, stitch_y = [], []
    rates = defaultdict(list)                              # qtype -> per-expression base rates
    sw = defaultdict(lambda: [0, 0, 0.0])                  # qtype -> [switches, pairs, sum dt]
    kin_x, kin_y, kin_w8 = [], [], []
    pb = RB.bar(len(vids), "videos", "video")
    for vid in vids:
        video = split.videos[vid]
        W, H = video.image_wh
        if a.tracks_dir:
            perc, aff = load_mot_tracks(os.path.join(a.tracks_dir, f"{vid}.txt"), len(video), (H, W), fps)
        else:
            perc, aff = load_perception(cache_root, vid)
        g = TrackletGraph(GraphConfig(**{**gkw, "fps": fps}))
        for r, fo in enumerate(perc):
            if fo is not None:
                g.update(r, r / fps, fo.tracks, (W, H), aff[r] if aff else None)
        # GT identity of every observation (independent of the expression)
        gt_cache: dict = {}

        def gid_of(rank, box):
            if rank not in gt_cache:
                gt_cache[rank] = video.gt_boxes(rank)
            gt = gt_cache[rank]
            if not len(gt):
                return None
            iou = box_iou_xywh(np.asarray(box)[None], gt[:, 1:5])[0]
            k = int(np.argmax(iou))
            return int(gt[k, 0]) if iou[k] >= 0.5 else None
        obs_gid = {sid: [gid_of(o.rank, o.box) for o in s.obs] for sid, s in g.segs.items()}
        same_e = {}
        for key, e in g.edges.items():
            ga, gb = obs_gid[e.i][-1], obs_gid[e.j][0]
            if ga is None or gb is None:
                continue
            same_e[key] = ga == gb
            if e.kind == "cont":
                c = same_by_reason[g.segs[e.j].cut_reason]
                c[0] += int(ga == gb)
                c[1] += 1
            else:
                stitch_x.append(e.prior_logit)
                stitch_y.append(int(ga == gb))
        exprs = list(split.expressions[vid])
        rng.shuffle(exprs)
        for ex in exprs[: a.max_expr]:
            q = parse(ex.text)
            qt = query_type(q)
            truth, num, den = {}, 0.0, 0.0
            for sid, s in g.segs.items():
                if s.n < 3 or not q.class_ok(s.cls_name):
                    continue
                hit = sum(1 for o, gi in zip(s.obs, obs_gid[sid])
                          if gi is not None and gi in video.referred_ids(ex, o.rank))
                r = int(hit / s.n >= 0.5)
                truth[sid] = r
                num += r * s.n
                den += s.n
                if qt == "motion":
                    kin_x.append(g.kinematic_prior(s, q))
                    kin_y.append(r)
                    kin_w8.append(1.0)
            if den > 0:
                rates[qt].append(num / den)
            for key, same in same_e.items():
                e = g.edges[key]
                if not same or e.i not in truth or e.j not in truth:
                    continue
                si, sj = g.segs[e.i], g.segs[e.j]
                c = sw[qt]
                c[0] += int(truth[e.i] != truth[e.j])
                c[1] += 1
                c[2] += max(0.0, 0.5 * (sj.t0 + sj.t1) - 0.5 * (si.t0 + si.t1))
        pb.update(1)
    pb.close()

    belief = {"enabled": True}
    belief["pi_reason"] = {r: round((c[0] + 1) / (c[1] + 2), 4) for r, c in sorted(same_by_reason.items())}
    if len(stitch_x) >= 20 and 0 < sum(stitch_y) < len(stitch_y):
        belief["stitch_ab"] = [round(v, 4) for v in logistic(stitch_x, stitch_y)]
    belief["rho_prior"] = {t: beta_moments(rates.get(t, [])) for t in ("static", "motion")}
    hz = {}
    for t in ("static", "motion"):
        s, n, dt = sw.get(t, [0, 0, 0.0])
        if n >= 20 and dt > 0:
            p = min((s + 0.5) / (n + 1), 0.95)
            hz[t] = round(-math.log(1 - p) / (dt / n), 4)
    if hz:
        belief["hazard"] = hz
    if len(kin_x) >= 50 and 0 < sum(kin_y) < len(kin_y) and np.std(kin_x) > 0:
        ka, kb = logistic(kin_x, kin_y, kin_w8)
        belief["kin_ab"] = [round(max(ka, 0.0), 4), round(kb, 4)]
        belief["kin_base"] = round(float(np.mean(kin_y)), 4)
    out = {"rmot": {"belief": belief}}
    if a.verifier_scores or a.verifier_answers:
        ps, ys = [], []
        for path in (a.verifier_answers or a.verifier_scores).split(","):
            for line in open(path):
                r = json.loads(line)
                if r.get("p") is None:
                    continue
                if a.verifier_answers:                  # replay log: one candidate per question only
                    if r.get("group", 1) != 1 or r.get("truth") is None:
                        continue
                    lab = int(float(r["truth"]) >= 0.5)
                else:
                    lab = int(r["label"])
                ps.append(min(max(float(r["p"]), 1e-4), 1 - 1e-4))
                ys.append(lab)
        if not ys:
            sys.exit("no usable verifier answers")
        z = np.log(np.array(ps) / (1 - np.array(ps)))
        A, B = logistic(z, ys)
        base = float(np.mean(ys))
        cz = A * z + B                                     # calibrated logit (posterior on the val base rate)
        yes = cz > math.log(base / (1 - base))              # LLR > 0
        y = np.array(ys, bool)
        tpr = float((yes & y).sum() / max(y.sum(), 1))
        fpr = float((yes & ~y).sum() / max((~y).sum(), 1))
        out["rmot"]["calib"] = {"single": [round(A, 4), round(B, 4)]}
        belief.update({"calib_base_rate": round(base, 4), "tpr": round(min(max(tpr, 0.51), 0.999), 4),
                       "fpr": round(min(max(fpr, 0.001), 0.49), 4)})
    stats = {"videos": len(vids), "cuts": {r: c[1] for r, c in same_by_reason.items()}, "stitch_edges": len(stitch_x),
             "expressions": {t: len(v) for t, v in rates.items()},
             "state_pairs": {t: v[1] for t, v in sw.items()}, "kin_samples": len(kin_x)}
    with open(a.out, "w") as f:
        f.write("# fitted by scripts/rmot_fit_belief.py on " + f"{a.dataset}/{a.split}: " + json.dumps(stats) + "\n")
        yaml.safe_dump(out, f, sort_keys=False)
    print(json.dumps(stats, indent=1))
    print(open(a.out).read())


if __name__ == "__main__":
    main()
