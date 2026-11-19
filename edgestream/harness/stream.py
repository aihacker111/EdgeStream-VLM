"""Causal video replay with a logical clock, plus a ring buffer for evidence clips."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class FramePacket:
    frame_idx: int
    t: float            # logical time in seconds (frame_idx / fps)
    image: np.ndarray   # BGR


class VideoStream:
    """Replays a video at its native rate. Consumers only ever see the past (causal)."""

    def __init__(self, path: str, fps: Optional[float] = None, max_frames: Optional[int] = None,
                 start_s: float = 0.0):
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video {path}")
        self.fps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self.max_frames = max_frames
        self.start_frame = int(round(start_s * self.fps))

    def __iter__(self) -> Iterator[FramePacket]:
        cap = cv2.VideoCapture(self.path)
        if self.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        i = 0
        while True:
            if self.max_frames is not None and i >= self.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            yield FramePacket(i, i / self.fps, frame)
            i += 1
        cap.release()


class FrameFolderStream:
    """A folder of frames replayed at ``fps``."""

    IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif")

    def __init__(self, folder: str, fps: float = 30.0, max_frames: Optional[int] = None,
                 start_s: float = 0.0):
        import re
        from pathlib import Path
        natural = lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]  # noqa: E731
        # natural order: Drone-Anomaly names frames 0.jpg, 1.jpg, ..., 10.jpg
        self.files = sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in self.IMG_EXT), key=natural)
        if not self.files:
            raise FileNotFoundError(f"No frames in {folder}")
        self.fps = float(fps)
        self.num_frames = len(self.files)
        first = cv2.imread(str(self.files[0]))
        self.height, self.width = first.shape[:2]
        self.max_frames = max_frames
        self.start_frame = int(round(start_s * self.fps))

    def __iter__(self) -> Iterator[FramePacket]:
        files = self.files[self.start_frame:]
        for i, f in enumerate(files):
            if self.max_frames is not None and i >= self.max_frames:
                break
            yield FramePacket(i, i / self.fps, cv2.imread(str(f)))


def open_stream(source: str, fps: Optional[float] = None, max_frames: Optional[int] = None,
                start_s: float = 0.0):
    """Video file or frame folder -> causal stream."""
    import os
    if os.path.isdir(source):
        return FrameFolderStream(source, fps or 30.0, max_frames, start_s)
    # a video file knows its own frame rate: ``fps`` is only a fallback for broken headers
    cap = cv2.VideoCapture(source)
    native = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    return VideoStream(source, None if native and native > 1 else fps, max_frames, start_s)


class FrameRingBuffer:
    """Keeps the last ``seconds`` of video as JPEG bytes (cheap enough for an edge box)."""

    def __init__(self, seconds: float, fps: float, max_width: int = 1280, quality: int = 80):
        self.buf: deque = deque(maxlen=max(1, int(round(seconds * fps))))
        self.max_width = max_width
        self.quality = quality

    def push(self, t: float, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_width / float(w))
        img = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1.0 else frame
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            self.buf.append((t, scale, enc.tobytes()))

    def window(self, t0: float, t1: float) -> list[tuple[float, float, np.ndarray]]:
        out = []
        for t, scale, data in self.buf:
            if t0 <= t <= t1:
                out.append((t, scale, cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)))
        return out

    def latest(self) -> Optional[tuple[float, float, np.ndarray]]:
        if not self.buf:
            return None
        t, scale, data = self.buf[-1]
        return t, scale, cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
