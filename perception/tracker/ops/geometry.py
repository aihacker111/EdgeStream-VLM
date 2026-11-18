# AERO-MOT 🚁 AGPL-3.0 License
# Copyright (C) 2026 AERO-MOT Authors

"""Small, dependency-free geometry helpers shared across the tracker core.

These used to live as private ``_centers_of`` / ``_iou_1_to_n`` functions at the
top of ``multitracker.py``. They are pure NumPy and hold no state, so they are
pulled out here where they can be read (and unit-tested) in isolation.
"""

import numpy as np


def centers_of(items):
    """(N, 2) array of box centres for a list of tracks/detections (tlwh boxes)."""
    out = np.empty((len(items), 2), dtype=np.float32)
    for i, t in enumerate(items):
        b = t.tlwh
        out[i, 0] = b[0] + b[2] * 0.5
        out[i, 1] = b[1] + b[3] * 0.5
    return out


def iou_1_to_n(box, boxes):
    """IoU of a single tlbr ``box`` against an (N, 4) array of tlbr ``boxes``."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    a1 = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    a2 = np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 0, None)
    union = a1 + a2 - inter + 1e-6
    return (inter / union).astype(np.float32)
