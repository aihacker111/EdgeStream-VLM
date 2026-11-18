"""Camera-motion (ego-motion) estimation from the background.

Moving objects cannot be used for this: on a fixed camera with one-way traffic the
median object velocity IS the traffic. We track corners on the background (object
boxes masked out) with Lucas-Kanade and fit a similarity transform with RANSAC.
For a fixed CCTV camera this returns ~identity; for a drifting UAV it removes the drift.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class EgoMotionEstimator:
    def __init__(self, mode: str = "flow", width: int = 480, max_corners: int = 300,
                 min_inliers: int = 12):
        if mode not in ("flow", "none"):
            raise ValueError("ego_motion mode must be 'flow' or 'none'")
        self.mode = mode
        self.width = width
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self.prev_gray = None
        self.scale = 1.0
        self.last_affine = np.array([[1, 0, 0], [0, 1, 0]], np.float64)   # prev -> curr, original px
        self.prev_boxes: list = []
        self.min_inlier_ratio = 0.5

    def update(self, frame: Optional[np.ndarray], boxes_tlwh: list) -> np.ndarray:
        identity = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
        if self.mode == "none" or frame is None:
            self.last_affine = identity
            return identity
        h, w = frame.shape[:2]
        self.scale = self.width / float(w)
        gray = cv2.cvtColor(cv2.resize(frame, (self.width, int(round(h * self.scale)))), cv2.COLOR_BGR2GRAY)
        A = identity
        if self.prev_gray is not None:
            mask = np.full(gray.shape, 255, np.uint8)
            # corners are found on the PREVIOUS frame, so mask objects where they were then
            # (and where they are now) with a margin for motion between frames
            for x, y, bw, bh in list(self.prev_boxes) + list(boxes_tlwh):
                m = 0.5 * max(bw, bh) + 4
                x0, y0 = int((x - m) * self.scale), int((y - m) * self.scale)
                x1, y1 = int((x + bw + m) * self.scale), int((y + bh + m) * self.scale)
                mask[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 0
            pts = cv2.goodFeaturesToTrack(self.prev_gray, self.max_corners, 0.01, 8, mask=mask)
            if pts is not None and len(pts) >= self.min_inliers:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None)
                good = st.reshape(-1) == 1
                if good.sum() >= self.min_inliers:
                    M, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                                         ransacReprojThreshold=1.0)
                    ok_inl = inl is not None and inl.sum() >= max(self.min_inliers,
                                                                  self.min_inlier_ratio * good.sum())
                    if M is not None and ok_inl:
                        # back to original-resolution pixels
                        M = M.astype(np.float64)
                        M[:, 2] /= self.scale
                        A = M
        self.prev_gray = gray
        self.prev_boxes = [tuple(b) for b in boxes_tlwh]
        self.last_affine = A
        return A

    @staticmethod
    def point_velocity(A: np.ndarray, x: float, y: float) -> np.ndarray:
        """Apparent motion (px/frame) that the camera alone induces at image point (x, y)."""
        p = np.array([x, y, 1.0])
        return A @ p - p[:2]
