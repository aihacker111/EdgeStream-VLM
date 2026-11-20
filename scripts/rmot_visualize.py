#!/usr/bin/env python
"""Render referring-MOT results of a benchmark run as videos (or frames): GT vs prediction.

    # the 3 worst and 3 best sequences of search@L2, side by side with filter_kf@L2
    python scripts/rmot_visualize.py --run runs/am_smoke --dataset aerialmind --root data/AerialMind \
        --split in_domain --methods search,filter_kf --lag 2 --worst 3 --best 3

    # one given sequence (video+expression), every frame saved as jpg instead of mp4
    python scripts/rmot_visualize.py --run runs/am_smoke --dataset aerialmind --root data/AerialMind \
        --split in_domain --methods search --lag inf --seq "uav0000009_03358_v+Moving cars" --frames

Drawing: thin green = GT referred object, coloured thick box + id = prediction (colour per id);
a prediction that matches no GT box (IoU < 0.5) gets a red "FP" tag, a GT box no prediction
covers gets a "miss" tag. The header shows the expression, the frame and the sequence's HOTA.
Output: <run>/vis/<method>_L<lag>/<video>/<expression>.mp4 (or a folder of jpg with --frames).
With several --methods the panels are placed side by side in one video.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from edgestream.rmot import hota as H  # noqa: E402
from edgestream.rmot.datasets import load_split  # noqa: E402


def color(i: int) -> tuple:
    rng = np.random.default_rng(int(i) * 7919 + 13)
    c = rng.integers(60, 256, 3)
    return int(c[0]), int(c[1]), int(c[2])


def by_frame(rows: np.ndarray) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(int(r[0]), []).append(r)
    return out


def draw_panel(img, gt_f, pr_f, title, scale):
    vis = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else img.copy()
    gt = np.array([g[2:6] for g in gt_f]).reshape(-1, 4)
    pr = np.array([p[2:6] for p in pr_f]).reshape(-1, 4)
    iou = H.box_iou_xywh(gt, pr) if len(gt) and len(pr) else np.zeros((len(gt), len(pr)))
    for k, g in enumerate(gt_f):
        x, y, w, h = [v * scale for v in g[2:6]]
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
        if not len(pr) or iou[k].max() < 0.5:
            cv2.putText(vis, "miss", (int(x), int(y + h + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    for k, p in enumerate(pr_f):
        x, y, w, h = [v * scale for v in p[2:6]]
        c = color(int(p[1]))
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), c, 2)
        tag = f"{int(p[1])}"
        fp = not len(gt) or iou[:, k].max() < 0.5
        cv2.putText(vis, tag, (int(x), max(int(y) - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        if fp:
            cv2.putText(vis, "FP", (int(x + w) + 2, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                        cv2.LINE_AA)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(vis, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def render(video, seq: str, panels: list, out_path: str, frames_dir: bool, fps: float, width: int) -> None:
    vid, expr = seq.split("+", 1)
    W, Hh = video.image_wh
    scale = min(1.0, width / float(W * len(panels)))
    data = []
    for name, d, score in panels:
        gt = by_frame(H.read_mot(os.path.join(d, vid, expr, "gt.txt")))
        pr = by_frame(H.read_mot(os.path.join(d, vid, expr, "predict.txt")))
        data.append((name, gt, pr, score))
    writer = None
    if frames_dir:
        os.makedirs(out_path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for r in range(len(video)):
        img = cv2.imread(video.frames[r])
        if img is None:
            continue
        f = r + 1
        tiles = [draw_panel(img, gt.get(f, []), pr.get(f, []),
                            f"{name} | HOTA {score:.1f} | f{f} | {expr[:60]}", scale) for name, gt, pr, score in data]
        canvas = np.concatenate(tiles, 1)
        if frames_dir:
            cv2.imwrite(os.path.join(out_path, f"{f:06d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        else:
            if writer is None:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
    if writer is not None:
        writer.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the --out folder of rmot_benchmark.py")
    ap.add_argument("--dataset", default="aerialmind", choices=["aerialmind", "refer_kitti_v2", "refer_kitti"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="in_domain")
    ap.add_argument("--methods", default="search", help="comma list; panels side by side")
    ap.add_argument("--lag", default="2", help="lag folder suffix: 0, 2, 5, inf")
    ap.add_argument("--seq", action="append", default=[], help="video+expression (repeatable)")
    ap.add_argument("--worst", type=int, default=0, help="N worst sequences of the first method (HOTA)")
    ap.add_argument("--best", type=int, default=0, help="N best sequences of the first method (HOTA)")
    ap.add_argument("--frames", action="store_true", help="save jpg frames instead of an mp4")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--width", type=int, default=1920, help="max width of the output canvas")
    a = ap.parse_args()

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    dirs = [os.path.join(a.run, f"{m}_L{a.lag}") for m in methods]
    for d in dirs:
        if not os.path.isdir(d):
            sys.exit(f"{d} not found (run rmot_benchmark.py first; check --methods/--lag)")
    per = []
    for d in dirs:
        p = os.path.join(d, "per_seq.json")
        per.append(json.load(open(p)) if os.path.isfile(p) else {})
    seqs = list(a.seq)
    if a.worst or a.best:
        if not per[0]:
            sys.exit(f"{dirs[0]}/per_seq.json missing: the run has not been evaluated yet; use --seq")
        ranked = sorted(per[0].items(), key=lambda kv: kv[1]["HOTA"])
        seqs += [k for k, _ in ranked[: a.worst]] + [k for k, _ in ranked[::-1][: a.best]]
    if not seqs:
        sys.exit("nothing to draw: give --seq, --worst or --best")
    split = load_split(a.dataset, a.root, a.split, None)
    cfg_fps = 10.0 if "kitti" in a.dataset else 30.0
    for seq in dict.fromkeys(seqs):
        vid, expr = seq.split("+", 1)
        if vid not in split.videos:
            print(f"skip {seq}: video not in {a.dataset}/{a.split}")
            continue
        panels = [(f"{m}@L{a.lag}", d, p.get(seq, {}).get("HOTA", float("nan"))) for m, d, p in zip(methods, dirs, per)]
        tag = "+".join(f"{m}" for m in methods)
        safe = expr.replace("/", "_")[:120]
        out = os.path.join(a.run, "vis", f"{tag}_L{a.lag}", vid, safe + ("" if a.frames else ".mp4"))
        print(f"{seq}: " + ", ".join(f"{n} HOTA {s:.1f}" for n, _, s in panels) + f" -> {out}", flush=True)
        render(split.videos[vid], seq, panels, out, a.frames, a.fps or cfg_fps, a.width)


if __name__ == "__main__":
    main()
