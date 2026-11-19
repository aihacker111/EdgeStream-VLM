"""Zoom re-detection: run the detector again on an up-scaled crop around a box we must output.

Used at commit time, only for objects the solver has decided are referred:

* frames inside a stitched gap (the tracker lost the object; the box is first interpolated
  between the two linked pieces, then snapped to a detection found in the zoomed crop), and
* QAM-recovered frames (the tracker coasted without a detection).

Small aerial objects are often missed at the detector's 960x544 input; at 2-3x zoom they are
found again. A zoom detection replaces the box only if it overlaps the expected box enough.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expression import FAMILY_OF
from .hota import box_iou_xywh


@dataclass
class ZoomConfig:
    enabled: bool = True
    context: float = 3.0            # crop side = context x object side (min min_crop)
    min_crop: int = 160
    min_iou: float = 0.3
    conf_thres: float = 0.25
    max_per_frame: int = 8


class ZoomRedetector:
    def __init__(self, detect_fn, cfg: ZoomConfig | None = None):
        """``detect_fn(bgr) -> [(tlwh, score, cls_name)]`` (e.g. ``Perceptor.detect``)."""
        self.detect = detect_fn
        self.cfg = cfg or ZoomConfig()
        self.calls = 0
        self.hits = 0

    def refine(self, frame: np.ndarray, box, family: str) -> tuple[np.ndarray, bool]:
        c = self.cfg
        x, y, w, h = [float(v) for v in box]
        side = max(c.min_crop, c.context * max(w, h))
        cx, cy = x + w / 2, y + h / 2
        H, W = frame.shape[:2]
        x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        x1, y1 = int(min(cx + side / 2, W)), int(min(cy + side / 2, H))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return np.asarray(box, np.float64), False
        self.calls += 1
        dets = self.detect(frame[y0:y1, x0:x1])
        best, best_iou = None, c.min_iou
        ref = np.array([[x, y, w, h]])
        for tlwh, score, name in dets:
            if score < c.conf_thres or FAMILY_OF.get(name, name) != family:
                continue
            b = np.array([tlwh[0] + x0, tlwh[1] + y0, tlwh[2], tlwh[3]], np.float64)
            iou = float(box_iou_xywh(ref, b[None])[0, 0])
            if iou > best_iou:
                best, best_iou = b, iou
        if best is None:
            return np.asarray(box, np.float64), False
        self.hits += 1
        return best, True
