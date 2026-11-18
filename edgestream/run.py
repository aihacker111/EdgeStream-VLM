"""Run the full pipeline on one video.

    python -m edgestream.run --video data/videos/drone_highway_1080p15.mp4 --out runs/highway
    python -m edgestream.run --video ... --vlm openai          # real VLM via vLLM/Ollama/LM Studio
    python -m edgestream.run --video ... --perception-cache runs/x/tracks.jsonl   # skip the detector
"""
from __future__ import annotations

import argparse
import json

from .pipeline import Pipeline, load_config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="runs/pipeline")
    ap.add_argument("--vlm", choices=["simulated", "openai", "transformers"], default=None)
    ap.add_argument("--perception-cache", default=None, help="tracks.jsonl from perception.cli")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.vlm:
        cfg["vlm"]["backend"] = args.vlm
    perceptor = None
    if args.perception_cache:
        from .harness.perception_cache import CachedPerception
        perceptor = CachedPerception(args.perception_cache)
    summary = Pipeline(cfg, perceptor=perceptor).run(args.video, args.out, max_frames=args.max_frames,
                                                     save_video=not args.no_video, start_s=args.start_s)
    brief = {k: summary[k] for k in ("frames", "duration_s", "wall_s", "perception_ms_median",
                                     "candidates_total", "candidates_by_type", "candidates_sent_to_vlm",
                                     "vlm_backend")}
    brief["vlm_calls"] = summary["budget"]["vlm_calls"]
    brief["events"] = [(e["event_id"], e["event_type"], e["severity"], round(e["t_start"], 1),
                        round(e["t_alert"], 1)) for e in summary["events"]]
    print(json.dumps(brief, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
