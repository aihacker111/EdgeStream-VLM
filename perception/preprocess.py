"""Frame preprocessing that matches how AERO-MOT detectors were trained.

AERO-MOT trains and evaluates with ``cv2.resize(INTER_AREA) -> BGR->RGB -> /255 ->
ImageNet mean/std`` (``aeromot/data/dataset.py: preprocess_for_tracking``).
The original ``tools/inference/track.py`` skipped both the channel swap and the
normalisation, so its detections did not match the paper numbers. This module is
the single source of truth for the port.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Transform:
    """Maps original-image pixels to network pixels: x_net = x * ratio_x + pad_w."""
    ratio_x: float
    ratio_y: float
    pad_w: float = 0.0
    pad_h: float = 0.0


def preprocess(frame_bgr: np.ndarray, net_w: int, net_h: int, letterbox: bool = False,
               normalize: bool = True) -> tuple[np.ndarray, Transform]:
    """Return a (3, net_h, net_w) float32 array and the geometric transform."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 BGR frame, got shape {frame_bgr.shape}")
    orig_h, orig_w = frame_bgr.shape[:2]
    if letterbox:
        r = min(net_h / orig_h, net_w / orig_w)
        new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((net_h, net_w, 3), 114, dtype=np.uint8)
        pad_w, pad_h = (net_w - new_w) * 0.5, (net_h - new_h) * 0.5
        top, left = int(round(pad_h - 0.1)), int(round(pad_w - 0.1))
        canvas[top:top + new_h, left:left + new_w] = resized
        img, tf = canvas, Transform(r, r, pad_w, pad_h)
    else:
        img = cv2.resize(frame_bgr, (net_w, net_h), interpolation=cv2.INTER_AREA)
        tf = Transform(net_w / orig_w, net_h / orig_h)

    img = img[:, :, ::-1].astype(np.float32) / 255.0          # BGR -> RGB
    if normalize:
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(img.transpose(2, 0, 1)), tf


def to_tensor(chw: np.ndarray, device: torch.device, half: bool = False) -> torch.Tensor:
    t = torch.from_numpy(chw).unsqueeze(0).to(device, non_blocking=True)
    return t.half() if half else t
