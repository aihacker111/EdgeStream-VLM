"""Synthetic wrong-way scenario through the whole pipeline (ground truth: car #999 from t=3s).

    python scripts/synthetic_demo.py --out runs/synthetic_wrong_way
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edgestream.pipeline import Pipeline, load_config  # noqa: E402
from tests.synthetic import FakePerceptor, scenario, write_video  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/synthetic_wrong_way")
ap.add_argument("--vlm", default="simulated")
args = ap.parse_args()

frames = scenario(seconds=12.0, wrong_way_from=3.0)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
write_video(frames, str(out / "input.mp4"))
cfg = load_config(ROOT / "configs" / "pipeline.yaml")
cfg["vlm"]["backend"] = args.vlm
s = Pipeline(cfg, perceptor=FakePerceptor(frames)).run(str(out / "input.mp4"), str(out))
print(json.dumps({"vlm_calls": s["budget"]["vlm_calls"], "events": s["events"]}, indent=2))
