"""Warm-start the kinematic normality bank from NORMAL training videos (no labels, no VLM).

    python scripts/fit_normal_bank.py --videos data/UIT-ADrone/train/videos --out runs/banks/uit_normal.npz
    python scripts/fit_normal_bank.py --videos data/UCF_Crimes/Normal_train --stride 2 --max-videos 200 \
        --config configs/vad_cctv.yaml --out runs/banks/ucf_normal.npz

The bank stores scale-free motion descriptors per class group (human / two-wheel / vehicle), so a
bank fitted on one set of scenes transfers to new scenes; location statistics are NOT transferred
(they are scene-specific and are always learned online).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgestream.vad.datasets import IMG_EXT, VIDEO_EXT  # noqa: E402
from edgestream.vad.normality import KinematicBank  # noqa: E402
from edgestream.vad.pipeline import VADPipeline, load_vad_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="folder of normal videos or frame folders")
    ap.add_argument("--config", default="configs/vad_uav.yaml")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--capacity", type=int, default=8192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    cfg = load_vad_config(a.config)
    cfg["vlm"]["backend"] = "simulated"
    cfg["gate"]["calls_per_min"], cfg["gate"]["bucket_capacity"] = 0.0, 0.0     # no VLM while fitting
    cfg.setdefault("objects", {})["use_pixels"] = False
    pipe = VADPipeline(cfg)
    knn = dict(cfg.get("normality", {}).get("knn", {}))
    knn.update(capacity=a.capacity, insert_below=float("inf"))       # everything here is normal
    bank = KinematicBank(**knn)
    root = Path(a.videos)
    srcs = sorted(p for p in root.iterdir() if p.suffix.lower() in VIDEO_EXT or
                  (p.is_dir() and any(q.suffix.lower() in IMG_EXT for q in p.iterdir())))[: a.max_videos]
    for k, src in enumerate(srcs):
        r = pipe.run(str(src), fps=a.fps, stride=a.stride, max_frames=a.max_frames, bank=bank)
        print(f"{k + 1}/{len(srcs)} {src.name}: {r['processed_frames']} frames, bank {len(bank)}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    bank.save(a.out)
    print(f"-> {a.out}  ({ {g: len(v) for g, v in bank.items.items()} })")


if __name__ == "__main__":
    main()
