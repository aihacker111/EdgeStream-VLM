#!/usr/bin/env python
"""Draw the ground truth of one expression on a few frames, to check frame indexing and box format.

    python scripts/rmot_check_gt.py --dataset aerialmind --root data/AerialMind --split in_domain --out runs/gt_check
    python scripts/rmot_check_gt.py --dataset aerialmind --root data/AerialMind --split cross_domain \
        --label-format cxcywh --out runs/gt_check_cd

Green = all objects of labels_with_ids in that frame, red = the objects the expression refers to.
If boxes are shifted by half their size, the box format is wrong (try the other --label-format);
if red boxes sit on the wrong objects, the frame indexing is off.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

from edgestream.rmot.datasets import check_label_format, load_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aerialmind", choices=["aerialmind", "refer_kitti_v2", "refer_kitti"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="in_domain")
    ap.add_argument("--label-format", default="auto", choices=["auto", "x1y1wh", "cxcywh"])
    ap.add_argument("--videos", type=int, default=3)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kw = {"label_format": a.label_format} if a.dataset == "aerialmind" else {}
    sp = load_split(a.dataset, a.root, a.split, None, **kw)
    os.makedirs(a.out, exist_ok=True)
    for vid in list(sp.videos)[: a.videos]:
        v = sp.videos[vid]
        chk = check_label_format(v)
        print(f"{vid}: {len(v)} frames, keys {v.frame_keys[0]}..{v.frame_keys[-1]}, format used {v.fmt}, "
              f"out-of-image x1y1wh={chk['x1y1wh']:.3f} cxcywh={chk['cxcywh']:.3f} -> looks like {chk['best']}, "
              f"{len(sp.expressions[vid])} expressions")
        ex = max(sp.expressions[vid], key=lambda e: sum(len(x) for x in e.label.values()))
        r_of = v.rank_of_key()
        ranks = sorted(r_of[k] for k in ex.label if k in r_of)
        if not ranks:
            print(f"  WARNING: no frame key of '{ex.expression_id}' matches an image of {vid}")
            continue
        picks = [ranks[int(i * (len(ranks) - 1) / max(a.frames - 1, 1))] for i in range(a.frames)]
        for r in picks:
            img = cv2.imread(v.frames[r])
            ref = v.referred_ids(ex, r)
            for oid, x, y, w, h in v.gt_boxes(r):
                c = (0, 0, 255) if int(oid) in ref else (0, 200, 0)
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), c, 2 if int(oid) in ref else 1)
            cv2.putText(img, f"{vid} frame {r + 1}: {ex.expression_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            p = os.path.join(a.out, f"{vid}_f{r + 1:06d}.jpg")
            cv2.imwrite(p, img)
            print(f"  wrote {p}  ({len(ref)} referred)")


if __name__ == "__main__":
    main()
