"""Pixel-level cues for what the detector cannot see (it only knows the VisDrone classes).

* ``ResidualMotion``  - motion left after camera compensation and outside every tracked box
                        (animals, objects thrown, unconventional vehicles, water play ...).
* ``StaticChange``    - dual-background (fast/slow running averages) left-object detector;
                        active only while the camera is steady (fixed CCTV, hovering UAV).
                        Catches a dropped bag, litter, a new mark on a wall.
Both run on a small grey image (default 320 px wide), a few ms per frame on a CPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Blob:
    box: tuple          # tlwh in original pixels
    area_frac: float
    strength: float     # [0,1]
    persist_s: float = 0.0
    first_t: float = 0.0
    bid: int = 0


def _mask_boxes(mask: np.ndarray, boxes, s: float, margin: float = 0.3) -> None:
    for x, y, w, h in boxes:
        m = margin * max(w, h) + 3
        x0, y0 = int((x - m) * s), int((y - m) * s)
        x1, y1 = int((x + w + m) * s), int((y + h + m) * s)
        mask[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 0


class _BlobTracker:
    def __init__(self, iou: float = 0.2):
        self.iou = iou
        self.blobs: list[Blob] = []
        self._next = 1

    @staticmethod
    def _iou(a, b) -> float:
        ax0, ay0, aw, ah = a
        bx0, by0, bw, bh = b
        w = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
        h = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
        inter = w * h
        u = aw * ah + bw * bh - inter
        return inter / u if u > 0 else 0.0

    def update(self, t: float, found: list[Blob], dilate: float = 0.5) -> list[Blob]:
        out = []
        for b in found:
            best, bi = None, self.iou
            for old in self.blobs:
                x, y, w, h = old.box
                grown = (x - dilate * w, y - dilate * h, w * (1 + 2 * dilate), h * (1 + 2 * dilate))
                v = self._iou(grown, b.box)
                if v > bi:
                    best, bi = old, v
            if best is None:
                b.first_t, b.bid = t, self._next
                self._next += 1
            else:
                b.first_t, b.bid = best.first_t, best.bid
            b.persist_s = t - b.first_t
            out.append(b)
        self.blobs = out
        return out


class ResidualMotion:
    def __init__(self, width: int = 320, diff_thr: int = 25, ema: float = 0.3, min_area_frac: float = 4e-4,
                 max_area_frac: float = 0.15, max_global_frac: float = 0.25):
        self.width = width
        self.diff_thr = diff_thr
        self.ema = ema
        self.min_area = min_area_frac
        self.max_area = max_area_frac
        self.max_global = max_global_frac
        self.prev = None
        self.energy = None
        self.scale = 1.0
        self.global_frac = 0.0
        self.tracker = _BlobTracker()

    def update(self, frame: np.ndarray, A_prev2curr: np.ndarray, boxes) -> None:
        h, w = frame.shape[:2]
        self.scale = self.width / float(w)
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (self.width, int(round(h * self.scale))),
                                                     interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self.prev is None or self.prev.shape != g.shape:
            self.prev, self.energy = g, np.zeros(g.shape, np.float32)
            return
        A = A_prev2curr.astype(np.float32).copy()
        A[:, 2] *= self.scale
        warped = cv2.warpAffine(self.prev, A, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpAffine(np.full_like(self.prev, 255), A, (g.shape[1], g.shape[0]),
                               flags=cv2.INTER_NEAREST, borderValue=0)
        valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
        diff = cv2.absdiff(g, warped)
        moving = ((diff > self.diff_thr) & (valid > 0)).astype(np.float32)
        self.global_frac = float(moving.mean())
        _mask_boxes(moving, boxes, self.scale)
        self.energy = (1 - self.ema) * self.energy + self.ema * moving
        self.prev = g

    def blobs(self, t: float) -> list[Blob]:
        if self.energy is None or self.global_frac > self.max_global:   # registration failed / flash
            return self.tracker.update(t, [])
        m = (self.energy > 0.5).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.dilate(m, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(m)
        total = m.shape[0] * m.shape[1]
        found = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            frac = a / total
            if self.min_area <= frac <= self.max_area:
                s = 1.0 / self.scale
                strength = float(self.energy[y:y + h, x:x + w].mean())
                found.append(Blob((x * s, y * s, w * s, h * s), float(frac), min(1.0, 1.5 * strength)))
        return self.tracker.update(t, found)


class StaticChange:
    def __init__(self, width: int = 320, fast: float = 0.05, slow: float = 0.002, thr: int = 30,
                 steady_px: float = 0.5, min_area_frac: float = 3e-4, max_area_frac: float = 0.05,
                 warmup_s: float = 20.0):
        self.width = width
        self.fast_a, self.slow_a, self.thr = fast, slow, thr
        self.steady_px = steady_px
        self.min_area, self.max_area = min_area_frac, max_area_frac
        self.warmup_s = warmup_s
        self.fast = self.slow = None
        self.t0 = None
        self.scale = 1.0
        self.active = False
        self.tracker = _BlobTracker(iou=0.3)

    def update(self, t: float, frame: np.ndarray, A_prev2curr: np.ndarray, boxes) -> None:
        shift = float(np.abs(A_prev2curr[:, 2]).max() + 100 * np.abs(A_prev2curr[:, :2] - np.eye(2)).max())
        h, w = frame.shape[:2]
        self.scale = self.width / float(w)
        g = cv2.cvtColor(cv2.resize(frame, (self.width, int(round(h * self.scale))),
                                    interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if shift > self.steady_px or self.fast is None:      # camera moved: restart the models
            self.fast, self.slow, self.t0 = g.copy(), g.copy(), t
            self.active = False
            return
        upd = np.ones(g.shape, np.float32)
        _mask_boxes(upd, boxes, self.scale, margin=0.2)       # people standing still are not litter
        self.fast += self.fast_a * upd * (g - self.fast)
        self.slow += self.slow_a * upd * (g - self.slow)
        self.active = (t - self.t0) >= self.warmup_s

    def blobs(self, t: float, boxes) -> list[Blob]:
        if not self.active:
            return self.tracker.update(t, [])
        m = (np.abs(self.fast - self.slow) > self.thr).astype(np.uint8)
        _mask_boxes(m, boxes, self.scale, margin=0.5)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(m)
        total = m.shape[0] * m.shape[1]
        s = 1.0 / self.scale
        found = [Blob((x * s, y * s, w * s, h * s), a / total, 1.0)
                 for x, y, w, h, a in stats[1:] if self.min_area <= a / total <= self.max_area]
        return self.tracker.update(t, found)
