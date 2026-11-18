"""Run perception on a video or a folder of frames.

    python -m perception.cli --source video.mp4 --out runs/demo --save-video
    python -m perception.cli --source frames_dir/ --fps 10 --out runs/seq

Writes ``tracks.jsonl`` (one FrameOutput per line) and, optionally, an annotated video.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2

from .config import PerceptionConfig
from .perceptor import Perceptor

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _frames(source: str, fps: float | None):
    src = Path(source)
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in _IMG_EXT)
        rate = fps or 30.0
        for i, f in enumerate(files):
            yield i / rate, cv2.imread(str(f))
        return
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {source}")
    rate = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i / rate, frame
        i += 1
    cap.release()


def draw(frame, out):
    vis = frame.copy()
    for t in out.tracks:
        x, y, w, h = t.tlwh.astype(int)
        color = (0, 165, 255) if t.is_recovered else (0, 220, 0)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, f"#{t.global_id} {t.cls_name}", (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(vis, f"t={out.timestamp:.2f}s  tracks={out.density}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/perception.yaml")
    ap.add_argument("--source", required=True, help="video file or folder of frames")
    ap.add_argument("--fps", type=float, default=None, help="override stream fps")
    ap.add_argument("--out", default="runs/perception")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = PerceptionConfig.from_yaml(args.config)
    if args.fps:
        cfg.tracker.frame_rate = int(round(args.fps))
    perceptor = Perceptor(cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer, totals = None, []
    with open(out_dir / "tracks.jsonl", "w") as f:
        for i, (ts, frame) in enumerate(_frames(args.source, args.fps)):
            if args.max_frames is not None and i >= args.max_frames:
                break
            out = perceptor.process(frame, timestamp=ts)
            f.write(json.dumps(out.to_dict()) + "\n")
            totals.append(out.latency_ms["total"])
            if args.save_video:
                vis = draw(frame, out)
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(str(out_dir / "tracks.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             cfg.tracker.frame_rate, (w, h))
                writer.write(vis)
    if writer is not None:
        writer.release()
    if totals:
        print(f"{len(totals)} frames, median latency {statistics.median(totals):.1f} ms "
              f"on {perceptor.device} -> {out_dir}")


if __name__ == "__main__":
    main()
