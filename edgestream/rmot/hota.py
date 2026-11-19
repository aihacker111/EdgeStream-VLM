"""HOTA / DetA / AssA / LocA for referring MOT, numerically identical to TrackEval.

This is a port of ``trackeval.metrics.HOTA`` (Luiten et al., IJCV 2021; MIT licence) together
with the parts of the RMOT TrackEval fork that matter for Refer-KITTI(-V2) and AerialMind:

* one "sequence" = one ``video_id+expression_id`` pair (a row of the seqmap);
* GT and predictions live side by side: ``<root>/<video_id>/<expression_id>/{gt.txt,predict.txt}``;
* line format ``frame,id,x,y,w,h,1,1,1`` with frames 1..N, N = number of images of the video;
* GT classes are all 1 (pedestrian) and zero_marked = 1, so the MOT17 preprocessing removes
  nothing and the similarity is plain box IoU;
* sequences are combined the TrackEval way (sum of TP/FN/FP, AssA and LocA weighted by TP).

``tests/test_rmot.py`` checks this module against the TrackEval sources when they are available.
Reported numbers are the mean over the 19 alpha thresholds, times 100, like the papers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

ALPHAS = np.arange(0.05, 0.99, 0.05)
_INT_FIELDS = ("HOTA_TP", "HOTA_FN", "HOTA_FP")
_ARR_FIELDS = ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA")
SUMMARY = ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA")
_EPS = np.finfo("float").eps


def box_iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between (N,4) and (M,4) boxes in x,y,w,h (TrackEval ``_calculate_box_ious``)."""
    a = np.asarray(a, np.float64).reshape(-1, 4).copy()
    b = np.asarray(b, np.float64).reshape(-1, 4).copy()
    a[:, 2:] += a[:, :2]
    b[:, 2:] += b[:, :2]
    mn = np.minimum(a[:, None, :], b[None, :, :])
    mx = np.maximum(a[:, None, :], b[None, :, :])
    inter = np.maximum(mn[..., 2] - mx[..., 0], 0) * np.maximum(mn[..., 3] - mx[..., 1], 0)
    area1 = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area2 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    inter[area1 <= _EPS, :] = 0
    inter[:, area2 <= _EPS] = 0
    inter[union <= _EPS] = 0
    union[union <= _EPS] = 1
    return inter / union


@dataclass
class SeqData:
    """Per-timestep ids (relabelled 0..n-1) and boxes for one sequence."""
    gt_ids: list
    gt_dets: list
    tr_ids: list
    tr_dets: list

    @classmethod
    def from_rows(cls, gt_rows: np.ndarray, tr_rows: np.ndarray, num_timesteps: int) -> "SeqData":
        """Rows are ``frame,id,x,y,w,h,...`` with frame in 1..num_timesteps."""
        def split(rows, what):
            rows = np.asarray(rows, np.float64)
            if rows.ndim != 2 or rows.size == 0:
                rows = np.zeros((0, 6))
            ids, dets = [], []
            if len(rows):
                fr = rows[:, 0].astype(int)
                bad = (fr < 1) | (fr > num_timesteps)
                if bad.any():
                    raise ValueError(f"{what} has frames outside 1..{num_timesteps}: {sorted(set(fr[bad]))[:5]}")
            for t in range(num_timesteps):
                r = rows[rows[:, 0].astype(int) == t + 1] if len(rows) else np.zeros((0, 6))
                i = r[:, 1].astype(int)
                if len(np.unique(i)) != len(i):
                    raise ValueError(f"{what} has the same id more than once in frame {t + 1}")
                ids.append(i)
                dets.append(r[:, 2:6])
            return ids, dets
        gi, gd = split(gt_rows, "GT")
        ti, td = split(tr_rows, "prediction")
        return cls(_relabel(gi), gd, _relabel(ti), td)


def _relabel(ids: list) -> list:
    allv = np.unique(np.concatenate(ids)) if ids and sum(len(x) for x in ids) else np.zeros(0, int)
    if len(allv) == 0:
        return [x.astype(int) for x in ids]
    m = {int(v): k for k, v in enumerate(allv)}
    return [np.array([m[int(v)] for v in x], int) for x in ids]


def _final(res: dict) -> dict:
    res["DetRe"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"])
    res["DetPr"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FP"])
    res["DetA"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"] + res["HOTA_FP"])
    res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
    res["OWTA"] = np.sqrt(res["DetRe"] * res["AssA"])
    return res


def hota_sequence(d: SeqData) -> dict:
    """TrackEval ``HOTA.eval_sequence``: per-alpha arrays."""
    nA = len(ALPHAS)
    res = {f: np.zeros(nA) for f in _INT_FIELDS + _ARR_FIELDS}
    n_gt = sum(len(x) for x in d.gt_ids)
    n_tr = sum(len(x) for x in d.tr_ids)
    if n_tr == 0:
        res["HOTA_FN"] = n_gt * np.ones(nA)
        res["LocA"] = np.ones(nA)
        return _final(res)
    if n_gt == 0:
        res["HOTA_FP"] = n_tr * np.ones(nA)
        res["LocA"] = np.ones(nA)
        return _final(res)
    G = int(max(x.max() for x in d.gt_ids if len(x))) + 1
    T = int(max(x.max() for x in d.tr_ids if len(x))) + 1
    sims = [box_iou_xywh(g, t) for g, t in zip(d.gt_dets, d.tr_dets)]
    pot = np.zeros((G, T))
    gt_cnt = np.zeros((G, 1))
    tr_cnt = np.zeros((1, T))
    for gi, ti, s in zip(d.gt_ids, d.tr_ids, sims):
        denom = s.sum(0)[None, :] + s.sum(1)[:, None] - s
        si = np.zeros_like(s)
        m = denom > 0 + _EPS
        si[m] = s[m] / denom[m]
        pot[gi[:, None], ti[None, :]] += si
        gt_cnt[gi] += 1
        tr_cnt[0, ti] += 1
    gas = pot / (gt_cnt + tr_cnt - pot)
    matches = [np.zeros_like(pot) for _ in ALPHAS]
    for gi, ti, s in zip(d.gt_ids, d.tr_ids, sims):
        if len(gi) == 0:
            res["HOTA_FP"] += len(ti)
            continue
        if len(ti) == 0:
            res["HOTA_FN"] += len(gi)
            continue
        score = gas[gi[:, None], ti[None, :]] * s
        r, c = linear_sum_assignment(-score)
        for a, alpha in enumerate(ALPHAS):
            ok = s[r, c] >= alpha - _EPS
            ar, ac = r[ok], c[ok]
            k = len(ar)
            res["HOTA_TP"][a] += k
            res["HOTA_FN"][a] += len(gi) - k
            res["HOTA_FP"][a] += len(ti) - k
            if k:
                res["LocA"][a] += s[ar, ac].sum()
                matches[a][gi[ar], ti[ac]] += 1
    for a in range(nA):
        mc = matches[a]
        tp = np.maximum(1, res["HOTA_TP"][a])
        res["AssA"][a] = np.sum(mc * (mc / np.maximum(1, gt_cnt + tr_cnt - mc))) / tp
        res["AssRe"][a] = np.sum(mc * (mc / np.maximum(1, gt_cnt))) / tp
        res["AssPr"][a] = np.sum(mc * (mc / np.maximum(1, tr_cnt))) / tp
    res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])
    return _final(res)


def combine(all_res: list[dict]) -> dict:
    """TrackEval ``HOTA.combine_sequences``."""
    nA = len(ALPHAS)
    if not all_res:
        return _final({f: np.zeros(nA) for f in _INT_FIELDS + _ARR_FIELDS})
    res = {f: sum(r[f] for r in all_res) for f in _INT_FIELDS}
    for f in ("AssRe", "AssPr", "AssA"):
        res[f] = sum(r[f] * r["HOTA_TP"] for r in all_res) / np.maximum(1.0, res["HOTA_TP"])
    loc = sum(r["LocA"] * r["HOTA_TP"] for r in all_res)
    res["LocA"] = np.maximum(1e-10, loc) / np.maximum(1e-10, res["HOTA_TP"])
    return _final(res)


def summarize(res: dict) -> dict:
    """Mean over alphas, x100 (the numbers papers report)."""
    out = {k: float(np.mean(res[k]) * 100) for k in SUMMARY}
    out["HOTA(0)"] = float(res["HOTA"][0] * 100)
    return out


# ---------------------------------------------------------------------- folder layout
def read_mot(path: str) -> np.ndarray:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 9))
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.replace(" ", "").split(",")])
    if not rows:
        return np.zeros((0, 9))
    n = max(len(r) for r in rows)
    return np.array([r + [1.0] * (n - len(r)) for r in rows])


def write_mot(path: str, rows) -> None:
    """``rows`` = iterable of (frame, id, x, y, w, h)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for fr, i, x, y, w, h in rows:
            f.write(f"{int(fr)},{int(i)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")


def read_seqmap(path: str) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip().split(",")[0].strip()
            if s and "+" in s and s.lower() not in ("name", "seq"):
                out.append(s)
    return out


def evaluate_folder(root: str, seqs: list[str], seq_lengths: dict, missing_pred: str = "empty") -> dict:
    """Evaluate ``root/<video>/<expr>/{gt,predict}.txt`` for ``video+expr`` names in ``seqs``.

    ``seq_lengths`` maps video_id -> number of images. ``missing_pred``: "empty" counts a missing
    predict.txt as no predictions; "error" raises (what TrackEval does).
    Returns {"combined": summary, "per_seq": {name: summary}, "raw": combined arrays}.
    """
    per, raws = {}, []
    for s in seqs:
        vid, expr = s.split("+", 1)
        d = os.path.join(root, vid, expr)
        gt_p, pr_p = os.path.join(d, "gt.txt"), os.path.join(d, "predict.txt")
        if not os.path.isfile(gt_p):
            raise FileNotFoundError(gt_p)
        if not os.path.isfile(pr_p) and missing_pred == "error":
            raise FileNotFoundError(pr_p)
        data = SeqData.from_rows(read_mot(gt_p), read_mot(pr_p), int(seq_lengths[vid]))
        r = hota_sequence(data)
        raws.append(r)
        per[s] = summarize(r)
    comb = combine(raws)
    return {"combined": summarize(comb), "per_seq": per, "raw": comb}
