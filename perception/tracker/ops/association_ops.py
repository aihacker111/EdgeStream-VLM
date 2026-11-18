# AERO-MOT 🚁 AGPL-3.0 License
# Copyright (C) 2026 AERO-MOT Authors

"""Minimal geometry matching primitives used by Falcon-MOT.

This file intentionally contains no alternative association strategy.  SPEAR
owns assignment/fusion; these helpers only produce geometry distances.
"""
import numpy as np

try:  # fast path (same numbers as the original AERO-MOT code)
    from cython_bbox import bbox_overlaps as bbox_ious
except ImportError:  # pragma: no cover - exercised on platforms without cython_bbox
    def bbox_ious(boxes, query_boxes):
        """NumPy port of ``cython_bbox.bbox_overlaps`` (keeps its +1 pixel convention)."""
        boxes = np.asarray(boxes, dtype=np.float64)
        query_boxes = np.asarray(query_boxes, dtype=np.float64)
        if boxes.size == 0 or query_boxes.size == 0:
            return np.zeros((len(boxes), len(query_boxes)), dtype=np.float64)
        q_area = ((query_boxes[:, 2] - query_boxes[:, 0] + 1)
                  * (query_boxes[:, 3] - query_boxes[:, 1] + 1))
        iw = (np.minimum(boxes[:, None, 2], query_boxes[None, :, 2])
              - np.maximum(boxes[:, None, 0], query_boxes[None, :, 0]) + 1)
        ih = (np.minimum(boxes[:, None, 3], query_boxes[None, :, 3])
              - np.maximum(boxes[:, None, 1], query_boxes[None, :, 1]) + 1)
        b_area = ((boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1))
        inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
        ua = b_area[:, None] + q_area[None, :] - inter
        return np.where(inter > 0, inter / ua, 0.0)


def ious(atlbrs, btlbrs):
    """Pairwise IoU between two tlbr box sets."""
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    return bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64),
    )


def iou_distance(atracks, btracks):
    """Return ``1 - IoU`` for track/detection lists."""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float64)
    if isinstance(atracks[0], np.ndarray) or isinstance(btracks[0], np.ndarray):
        atlbrs, btlbrs = atracks, btracks
    else:
        atlbrs = [t.tlbr for t in atracks]
        btlbrs = [t.tlbr for t in btracks]
    return 1.0 - ious(atlbrs, btlbrs)


def biou_distance(atracks, btracks, buffer_scale: float = 0.30):
    """Return ``1 - BIoU`` (Scale-Buffered IoU, C-BIoU baseline) for track/detection lists."""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float64)
    a = _tlwh_matrix(atracks)
    b = _tlwh_matrix(btracks)
    a_buf = np.column_stack([
        a[:, 0] - buffer_scale * a[:, 2],
        a[:, 1] - buffer_scale * a[:, 3],
        a[:, 0] + (1.0 + buffer_scale) * a[:, 2],
        a[:, 1] + (1.0 + buffer_scale) * a[:, 3],
    ])
    b_buf = np.column_stack([
        b[:, 0] - buffer_scale * b[:, 2],
        b[:, 1] - buffer_scale * b[:, 3],
        b[:, 0] + (1.0 + buffer_scale) * b[:, 2],
        b[:, 1] + (1.0 + buffer_scale) * b[:, 3],
    ])
    return 1.0 - ious(a_buf, b_buf)


def _tlwh_matrix(tracks):
    if len(tracks) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if isinstance(tracks[0], np.ndarray):
        return np.asarray(tracks, dtype=np.float64)
    return np.asarray([t.tlwh for t in tracks], dtype=np.float64)


def _tlwh_to_tlbr(a):
    b = a.copy()
    b[:, 2:] += b[:, :2]
    return b


def w2_iou_fusion_distance(atracks, btracks, norm_c=2.0, w2_weight=0.5):
    """Geometry distance combining IoU and normalized Wasserstein-2 distance."""
    na, nb = len(atracks), len(btracks)
    if na == 0 or nb == 0:
        return np.zeros((na, nb), dtype=np.float32)

    a = _tlwh_matrix(atracks)
    b = _tlwh_matrix(btracks)
    iou_dist = 1.0 - ious(_tlwh_to_tlbr(a), _tlwh_to_tlbr(b))

    a_cx = a[:, 0] + a[:, 2] * 0.5
    a_cy = a[:, 1] + a[:, 3] * 0.5
    b_cx = b[:, 0] + b[:, 2] * 0.5
    b_cy = b[:, 1] + b[:, 3] * 0.5

    dcx = a_cx[:, None] - b_cx[None, :]
    dcy = a_cy[:, None] - b_cy[None, :]
    dw = a[:, 2, None] - b[None, :, 2]
    dh = a[:, 3, None] - b[None, :, 3]

    w2_sq = (dcx ** 2 + dcy ** 2) + (dw ** 2 + dh ** 2) * 0.25
    norm = norm_c * (b[:, 2] * b[:, 3])[None, :] + 1e-6
    w2_dist = 1.0 - np.exp(-w2_sq / norm)
    return ((1.0 - w2_weight) * iou_dist + w2_weight * w2_dist).astype(np.float32)
