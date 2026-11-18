"""Whole pipeline on a synthetic video with one wrong-way car (simulated VLM backend)."""
import json
from pathlib import Path

from edgestream.pipeline import Pipeline, load_config

from .synthetic import FakePerceptor, scenario, write_video

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, wrong_way_from):
    frames = scenario(seconds=12.0, wrong_way_from=wrong_way_from)
    video = str(tmp_path / "synthetic.mp4")
    write_video(frames, video)
    cfg = load_config(ROOT / "configs" / "pipeline.yaml")
    cfg["vlm"]["backend"] = "simulated"
    out = tmp_path / "run"
    summary = Pipeline(cfg, perceptor=FakePerceptor(frames)).run(video, str(out), save_video=False)
    return summary, out


def test_wrong_way_car_produces_one_routed_event_with_evidence(tmp_path):
    summary, out = _run(tmp_path, wrong_way_from=3.0)
    events = summary["events"]
    assert len(events) == 1
    ev = events[0]
    assert ev["event_type"] == "counter_flow" and ev["track_ids"] == [999]
    assert ev["severity"] == "high" and ev["t_alert"] >= ev["t_start"]
    assert (out / "evidence" / ev["evidence"]["clip"]).exists()
    assert "command_center" in ev["evidence"]["channels"]
    routed = [json.loads(line) for line in open(out / "route_command_center.jsonl")]
    assert routed[0]["event_id"] == ev["event_id"]
    assert summary["budget"]["vlm_calls"] <= 1 + summary["duration_s"] / 60 * 6 + 3


def test_normal_traffic_produces_no_event(tmp_path):
    summary, _ = _run(tmp_path, wrong_way_from=None)
    assert summary["events"] == []
