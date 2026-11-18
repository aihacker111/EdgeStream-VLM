"""Replay perception from ``tracks.jsonl`` so budget sweeps don't re-run the detector.

Embeddings are not stored in the JSONL, so re-identification is off in cache mode.
"""
from __future__ import annotations

import json

import numpy as np

from perception.types import FrameOutput, TrackOutput


class CachedPerception:
    appearance = False

    def __init__(self, jsonl_path: str):
        self.frames = []
        with open(jsonl_path) as f:
            for line in f:
                self.frames.append(json.loads(line))
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    def process(self, frame_bgr=None, timestamp=None) -> FrameOutput:
        if self._i >= len(self.frames):
            raise IndexError("perception cache exhausted")
        r = self.frames[self._i]
        self._i += 1
        tracks = [TrackOutput(
            global_id=t["global_id"], track_id=t["track_id"], cls_id=t["cls_id"], cls_name=t["cls_name"],
            tlwh=np.asarray(t["tlwh"], np.float32), velocity=np.asarray(t["velocity"], np.float32),
            quality=t["quality"], det_score=t["det_score"], age=t["age"], is_recovered=t["is_recovered"],
            kappa=t.get("kappa"), embedding=None) for t in r["tracks"]]
        return FrameOutput(frame_idx=r["frame_idx"],
                           timestamp=timestamp if timestamp is not None else r["timestamp"],
                           image_hw=tuple(r["image_hw"]), tracks=tracks,
                           num_detections=r["num_detections"], latency_ms=r.get("latency_ms", {}))
