"""Synthetic surveillance scene with frame-level ground truth (fixed camera, 10 fps).

People walk both ways along a footpath (y≈180). Three anomalies:
    running   t=10-14 s   one person runs at ~3.5x walking speed
    falling   t=20-26 s   one person falls (box becomes wide and flat) and lies still
    lawn      t=30-36 s   one person walks onto the lawn (y≈300), where nobody goes
"""
from __future__ import annotations

import cv2
import numpy as np

from perception.types import FrameOutput, TrackOutput

W, H, FPS = 640, 360, 10
PW, PH = 10, 24                    # standing person box
WALK = 2.2                         # px/frame ≈ 1.4 body sizes/s
EVENTS = {"running": (10.0, 14.0), "falling": (20.0, 26.0), "lawn": (30.0, 36.0)}


def scenario(seconds: float = 40.0, anomalies: bool = True, seed: int = 0):
    """Return (frames, labels). frames[i] = (t, [(gid, cx, cy, w, h, vx, vy)])."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)
    walkers = []
    for k in range(8):
        d = 1 if k % 2 == 0 else -1
        walkers.append((1 + k, rng.uniform(0, W), 170 + 8 * (k % 3), d * WALK * rng.uniform(0.85, 1.15)))
    frames, labels = [], np.zeros(n, np.uint8)
    for i in range(n):
        t = i / FPS
        objs = []
        for gid, x0, y, v in walkers:
            x = (x0 + v * i) % (W + 40) - 20
            if 0 <= x <= W:
                objs.append((gid, x, y + rng.normal(0, 0.3), PW, PH, v, 0.0))
        if anomalies:
            a, b = EVENTS["running"]
            if a <= t < b + 6:                           # runs for 4 s, then walks on
                k = i - a * FPS
                run_k = min(k, (b - a) * FPS)
                x = 40 + 3.5 * WALK * run_k + WALK * (k - run_k)
                v = 3.5 * WALK if t < b else WALK
                objs.append((101, x, 186, PW, PH, v, 0.0))
            a, b = EVENTS["falling"]
            if a - 4 <= t < b:
                if t < a:
                    x = 200 + WALK * (i - (a - 4) * FPS)
                    objs.append((102, x, 182, PW, PH, WALK, 0.0))
                else:
                    x = 200 + WALK * 4 * FPS
                    objs.append((102, x, 190, PH, PW, 0.0, 0.0))     # lying: wide and flat
            a, b = EVENTS["lawn"]
            if a - 3 <= t < b:
                if t < a:
                    objs.append((103, 500 + WALK * (i - a * FPS), 182, PW, PH, WALK, 0.0))
                else:
                    y = min(182 + 4.0 * (i - a * FPS), 300)
                    objs.append((103, 500 - 0.5 * (i - a * FPS), y, PW, PH, -0.5, 4.0 if y < 300 else 0.0))
            for name, (a, b) in EVENTS.items():
                if a <= t < b:
                    labels[i] = 1
        frames.append((t, objs))
    return frames, labels


class FakePerceptorVAD:
    appearance = False

    def __init__(self, frames, cls_name: str = "pedestrian"):
        self.frames, self.cls_name = frames, cls_name
        self.i, self.age = 0, {}

    def reset(self):
        self.i, self.age = 0, {}

    def process(self, frame, timestamp=None):
        t, objs = self.frames[self.i]
        self.i += 1
        tracks = []
        for gid, cx, cy, w, h, vx, vy in objs:
            self.age[gid] = self.age.get(gid, -1) + 1
            tracks.append(TrackOutput(
                global_id=gid, track_id=gid, cls_id=0, cls_name=self.cls_name,
                tlwh=np.array([cx - w / 2, cy - h / 2, w, h], np.float32),
                velocity=np.array([vx, vy], np.float32), quality=0.9, det_score=0.9,
                age=self.age[gid], is_recovered=False))
        return FrameOutput(frame_idx=self.i - 1, timestamp=t, image_hw=(H, W), tracks=tracks,
                           num_detections=len(tracks), latency_ms={"total": 1.0})


def write_video(frames, path: str) -> None:
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _, objs in frames:
        img = np.full((H, W, 3), (60, 140, 60), np.uint8)              # lawn
        cv2.rectangle(img, (0, 150), (W, 215), (150, 150, 150), -1)     # footpath
        for gid, cx, cy, w, h, *_ in objs:
            cv2.rectangle(img, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)),
                          (40, 40, 200) if gid > 100 else (230, 230, 230), -1)
        vw.write(img)
    vw.release()
