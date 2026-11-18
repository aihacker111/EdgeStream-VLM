"""Run streaming VAD on one video (or frame folder) and write frame scores + an annotated video.

    python -m edgestream.vad.run --video data/videos/drone_highway_1080p15.mp4 --out runs/vad_demo --save-video
    python -m edgestream.vad.run --video clip.mp4 --vlm openai          # Ollama / vLLM (see configs/vad_uav.yaml)
    python -m edgestream.vad.run --video clip.mp4 --labels clip.npy     # also print AUC / AP
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .metrics import vad_report
from .pipeline import VADPipeline, load_vad_config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/vad_uav.yaml")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="runs/vad")
    ap.add_argument("--vlm", choices=["simulated", "oracle", "openai", "transformers"], default=None)
    ap.add_argument("--vlm-model", default=None, help="HF id (transformers) or server model name (openai)")
    ap.add_argument("--budget", type=float, default=None, help="VLM calls per minute")
    ap.add_argument("--labels", default=None, help=".npy / .txt with one 0/1 per frame")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--perception-cache", default=None, help="tracks.jsonl recorded by an earlier run")
    ap.add_argument("--save-video", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_vad_config(a.config)
    if a.vlm:
        cfg["vlm"]["backend"] = a.vlm
    if a.vlm_model:
        key = "transformers" if (a.vlm or cfg["vlm"]["backend"]) == "transformers" else "openai"
        cfg["vlm"][key]["model_id" if key == "transformers" else "model"] = a.vlm_model
    if a.budget is not None:
        cfg["gate"]["calls_per_min"] = a.budget
    labels = None
    if a.labels:
        labels = np.load(a.labels) if a.labels.endswith(".npy") else np.loadtxt(a.labels)
        labels = labels.astype(np.uint8).ravel()
    perceptor = None
    if a.perception_cache:
        from ..harness.perception_cache import CachedPerception
        perceptor = CachedPerception(a.perception_cache)
    r = VADPipeline(cfg).run(a.video, out_dir=a.out, labels=labels, fps=a.fps, stride=a.stride,
                             max_frames=a.max_frames, save_video=a.save_video, perceptor=perceptor,
                             record_tracks=perceptor is None)
    brief = {k: r[k] for k in ("processed_frames", "duration_s", "wall_s", "perception_ms_median",
                               "candidates_by_type", "candidates_sent")}
    brief["vlm_calls"] = r["budget"]["vlm_calls"]
    brief["events"] = [(e["event_type"], round(e["t_start"], 1), round(e["t_alert"], 1)) for e in r["events"]]
    if labels is not None:
        n = min(len(labels), len(r["scores_online"]))
        for tag in ("online", "offline"):
            rep = vad_report({"v": (r[f"scores_{tag}"][:n], labels[:n])})
            brief[f"{tag}_auc"], brief[f"{tag}_ap"] = round(rep["micro_auc"], 4), round(rep["ap"], 4)
    print(json.dumps(brief, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
