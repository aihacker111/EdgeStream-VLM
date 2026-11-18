"""Perception module of EdgeStream-VLM: detector + AERO-MOT tracker.

    from perception import Perceptor
    p = Perceptor.from_yaml("configs/perception.yaml")
    out = p.process(frame_bgr, timestamp=t)   # -> FrameOutput

Ported from AERO-MOT (AGPL-3.0); see perception/NOTICE.
"""
from .config import PerceptionConfig
from .perceptor import Perceptor
from .types import FrameOutput, TrackOutput

__all__ = ["Perceptor", "PerceptionConfig", "FrameOutput", "TrackOutput"]
__version__ = "0.1.0"
