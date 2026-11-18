"""Data contracts between perception and the rest of the pipeline.

Everything downstream (scene state, candidate generator, memory, VLM prompt
builder) should depend only on these types, never on tracker internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

VISDRONE_CLASSES = (
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
)


@dataclass
class TrackOutput:
    global_id: int                 # unique per Perceptor instance (tracker ids are per class)
    track_id: int                  # raw per-class id from the tracker
    cls_id: int                    # temporal-consensus class (TCC)
    cls_name: str
    tlwh: np.ndarray               # (4,) original-image pixels
    velocity: np.ndarray           # (2,) centre velocity in px/frame (Kalman state)
    quality: float                 # AERO-MOT track quality score (TQS), in [0, 1]
    det_score: float               # detector confidence of the last matched detection
    age: int                       # frames since the track started
    is_recovered: bool             # position comes from QAM recovery, not a detection
    kappa: Optional[float] = None  # vMF appearance concentration; None without ReID
    embedding: Optional[np.ndarray] = None  # (D,) L2-normalised appearance; None without ReID

    @property
    def center(self) -> np.ndarray:
        return self.tlwh[:2] + 0.5 * self.tlwh[2:]

    def to_dict(self) -> dict:
        return {
            "global_id": self.global_id, "track_id": self.track_id, "cls_id": self.cls_id,
            "cls_name": self.cls_name, "tlwh": [round(float(v), 2) for v in self.tlwh],
            "velocity": [round(float(v), 3) for v in self.velocity],
            "quality": round(self.quality, 4), "det_score": round(self.det_score, 4),
            "age": self.age, "is_recovered": self.is_recovered,
            "kappa": None if self.kappa is None else round(self.kappa, 3),
        }


@dataclass
class FrameOutput:
    frame_idx: int
    timestamp: float               # seconds, from the caller (stream clock)
    image_hw: tuple[int, int]
    tracks: list[TrackOutput] = field(default_factory=list)
    num_detections: int = 0
    latency_ms: dict = field(default_factory=dict)

    @property
    def density(self) -> int:
        """Number of active tracks; the density axis used across the whole project."""
        return len(self.tracks)

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx, "timestamp": self.timestamp,
            "image_hw": list(self.image_hw), "num_detections": self.num_detections,
            "latency_ms": {k: round(v, 2) for k, v in self.latency_ms.items()},
            "tracks": [t.to_dict() for t in self.tracks],
        }
