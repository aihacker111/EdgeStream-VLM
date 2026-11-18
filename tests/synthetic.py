"""Synthetic traffic: one lane flowing right plus one wrong-way vehicle (ground truth known)."""
from __future__ import annotations

import cv2
import numpy as np

from perception.types import FrameOutput, TrackOutput

W, H, FPS = 640, 360, 10
CAR = (20, 12)


def scenario(seconds: float = 12.0, wrong_way_from: float | None = 3.0, bidirectional: bool = False):
    """Yield (t, [(gid, cx, cy, vx, vy)]) per frame. Velocities in px/frame."""
    n = int(seconds * FPS)
    v = 6.0                                  # px/frame = 60 px/s ~ 3.9 car-sizes/s
    lanes = [(180, +1)] + ([(200, -1)] if bidirectional else [])
    frames = []
    for i in range(n):
        objs = []
        for li, (y, d) in enumerate(lanes):
            for k in range(10):
                x = (k * 70 + d * v * i) % (W + 140) - 70
                if 0 <= x <= W:
                    objs.append((1 + 100 * li + k, x, y, d * v, 0.0))
        if wrong_way_from is not None and i / FPS >= wrong_way_from:
            x = W - 40 - v * (i - wrong_way_from * FPS)
            if 0 <= x <= W:
                objs.append((999, x, 180 + 14, -v, 0.0))
        frames.append((i / FPS, objs))
    return frames


class FakePerceptor:
    appearance = False

    def __init__(self, frames):
        self.frames = frames
        self.i = 0
        self.age = {}

    def reset(self):
        self.i = 0
        self.age = {}

    def process(self, frame, timestamp=None):
        t, objs = self.frames[self.i]
        self.i += 1
        tracks = []
        for gid, cx, cy, vx, vy in objs:
            self.age[gid] = self.age.get(gid, -1) + 1
            tracks.append(TrackOutput(
                global_id=gid, track_id=gid, cls_id=3, cls_name="car",
                tlwh=np.array([cx - CAR[0] / 2, cy - CAR[1] / 2, *CAR], np.float32),
                velocity=np.array([vx, vy], np.float32), quality=0.9, det_score=0.9,
                age=self.age[gid], is_recovered=False))
        return FrameOutput(frame_idx=self.i - 1, timestamp=t, image_hw=(H, W), tracks=tracks,
                           num_detections=len(tracks), latency_ms={"total": 1.0})


def write_video(frames, path: str) -> None:
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _, objs in frames:
        img = np.full((H, W, 3), 90, np.uint8)
        cv2.rectangle(img, (0, 165), (W, 215), (60, 60, 60), -1)
        for gid, cx, cy, *_ in objs:
            col = (0, 0, 200) if gid == 999 else (220, 220, 220)
            cv2.rectangle(img, (int(cx - CAR[0] / 2), int(cy - CAR[1] / 2)),
                          (int(cx + CAR[0] / 2), int(cy + CAR[1] / 2)), col, -1)
        vw.write(img)
    vw.release()
