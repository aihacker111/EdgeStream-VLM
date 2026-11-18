"""Video anomaly detection metrics (no sklearn dependency).

Frame-level protocols used by the benchmarks we target:
* micro-AUC : concatenate all test frames, one ROC AUC (UCF-Crime "AUC", UAV "Micro-AUC")
* macro-AUC : mean of per-video AUCs (videos with a single class are skipped)
* AP        : average precision over all frames (XD-Violence)
* AUC_A     : micro-AUC restricted to videos that contain anomalies (less scene-biased)
* EER       : equal error rate of the micro ROC
"""
from __future__ import annotations

import numpy as np


def _rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic with average ranks for ties."""
    y = labels.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_auc(scores, labels) -> float:
    return _rank_auc(np.asarray(scores, np.float64).ravel(), np.asarray(labels).ravel())


def average_precision(scores, labels) -> float:
    """Step-wise AP (same definition as sklearn.metrics.average_precision_score)."""
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[distinct]
    fps = (distinct + 1) - tps
    precision = tps / (tps + fps)
    recall = tps / y.sum()
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def eer(scores, labels) -> float:
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    thr = np.unique(s)
    fpr = np.array([(s[~y] >= t).mean() for t in thr])
    fnr = np.array([(s[y] < t).mean() for t in thr])
    k = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[k] + fnr[k]) / 2.0)


def vad_report(per_video: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """``per_video[name] = (scores, labels)`` with one entry per frame."""
    all_s = np.concatenate([v[0] for v in per_video.values()])
    all_y = np.concatenate([v[1] for v in per_video.values()]).astype(int)
    per = {}
    for name, (s, y) in per_video.items():
        per[name] = roc_auc(s, y)
    valid = [a for a in per.values() if not np.isnan(a)]
    abn = [k for k, (_, y) in per_video.items() if np.any(y)]
    auc_a = roc_auc(np.concatenate([per_video[k][0] for k in abn]),
                    np.concatenate([per_video[k][1] for k in abn])) if abn else float("nan")
    return {
        "videos": len(per_video),
        "frames": int(all_y.size),
        "abnormal_frame_ratio": round(float(all_y.mean()), 4),
        "micro_auc": roc_auc(all_s, all_y),
        "macro_auc": float(np.mean(valid)) if valid else float("nan"),
        "ap": average_precision(all_s, all_y),
        "auc_abnormal_videos": auc_a,
        "eer": eer(all_s, all_y),
        "per_video_auc": per,
    }


def box_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def category_report(rows: list[dict]) -> dict:
    """A2Seek-style sample metrics. Each row: gt_category, pred_category, gt_box?, pred_box?, scene?"""
    if not rows:
        return {}
    n = len(rows)
    first = sum(r["pred_category"] == r["gt_category"] for r in rows) / n
    binary = sum((r["pred_category"] == "Normal") == (r["gt_category"] == "Normal") for r in rows) / n
    ious = [box_iou(r.get("gt_box"), r.get("pred_box")) for r in rows
            if r.get("gt_box") and any(r["gt_box"])]
    scenes = {}
    for r in rows:
        scenes.setdefault(r.get("scene", "all"), []).append(r["pred_category"] == r["gt_category"])
    return {
        "examples": n,
        "first_category_accuracy": first,
        "binary_anomaly_accuracy": binary,
        "scene_macro_first_category_accuracy": float(np.mean([np.mean(v) for v in scenes.values()])),
        "localized_examples": len(ious),
        "mean_first_box_iou": float(np.mean(ious)) if ious else None,
        "localization_hit_rate@0.5": float(np.mean([i >= 0.5 for i in ious])) if ious else None,
    }
