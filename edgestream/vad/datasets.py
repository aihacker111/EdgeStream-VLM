"""Benchmark loaders -> a list of ``VADVideo`` (source + frame-level labels).

Exact formats are implemented for UCF-Crime, XD-Violence and A2Seek (formats verified
against their public annotation files / official evaluation code). UIT-ADrone,
Drone-Anomaly and MUVAD are loaded through ``generic_frame_labels``: point it at the
video (or frame-folder) directory and a directory of per-video label files; check
the release you downloaded and adjust the patterns if needed.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

VIDEO_EXT = (".mp4", ".avi", ".mkv", ".mov", ".webm")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif")


@dataclass
class VADVideo:
    name: str
    source: str                         # video file or folder of frames
    labels: Optional[np.ndarray]        # 0/1 per frame (None when unknown)
    fps: Optional[float] = None         # needed for frame folders
    probes: list = field(default_factory=list)   # A2Seek-style per-frame questions
    meta: dict = field(default_factory=dict)


def _strip_ext(name: str) -> str:
    """Drop a video extension only: XD-Violence names contain dots ("A.Beautiful.Mind__#0-17-50...")."""
    for ext in VIDEO_EXT:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


_INDEX: dict = {}


def _find_video(root: Path, stem: str) -> Optional[Path]:
    """Look a video (or frame folder) up by name without glob (names may contain [ ] or #)."""
    key = str(root.resolve())
    if key not in _INDEX:
        idx = {}
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXT:
                idx.setdefault(_strip_ext(p.name), p)
            elif p.is_dir():
                idx.setdefault(p.name, p)
        _INDEX[key] = idx
    return _INDEX[key].get(stem)


def _count_frames(source: Path) -> int:
    if source.is_dir():
        return sum(1 for p in source.iterdir() if p.suffix.lower() in IMG_EXT)
    import cv2
    cap = cv2.VideoCapture(str(source))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def intervals_to_labels(n_frames: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    y = np.zeros(n_frames, dtype=np.uint8)
    for s, e in intervals:
        if s < 0 or e < 0:
            continue
        y[max(0, int(s)):min(n_frames, int(e) + 1)] = 1
    return y


# ---------------------------------------------------------------------------- UCF-Crime
def ucf_crime(video_root: str, annotation_txt: str) -> list[VADVideo]:
    """``Temporal_Anomaly_Annotation_for_Testing_Videos.txt``:
    ``Arrest001_x264.mp4  Arrest  1185  1485  -1  -1`` (frame indices, 2 segments max)."""
    root = Path(video_root)
    _INDEX.pop(str(root.resolve()), None)
    out = []
    for line in Path(annotation_txt).read_text().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        stem = _strip_ext(Path(parts[0]).name)
        src = _find_video(root, stem)
        if src is None:
            continue
        s1, e1, s2, e2 = (int(x) for x in parts[2:6])
        n = _count_frames(src)
        out.append(VADVideo(stem, str(src), intervals_to_labels(n, [(s1, e1), (s2, e2)]),
                            meta={"class": parts[1]}))
    return out


# ---------------------------------------------------------------------------- XD-Violence
def xd_violence(video_root: str, annotation_txt: str, test_list: Optional[str] = None) -> list[VADVideo]:
    """``annotations.txt``: ``<video_name> s1 e1 [s2 e2 ...]`` (frame indices). Test videos
    absent from the file (``label_A``) are normal."""
    root = Path(video_root)
    _INDEX.pop(str(root.resolve()), None)
    ann = {}
    for line in Path(annotation_txt).read_text().splitlines():
        parts = line.split()
        if parts:
            nums = [int(x) for x in parts[1:]]
            ann[_strip_ext(Path(parts[0]).name)] = list(zip(nums[0::2], nums[1::2]))
    stems = ([_strip_ext(Path(x.strip()).name) for x in Path(test_list).read_text().splitlines() if x.strip()]
             if test_list else [_strip_ext(p.name) for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT])
    out = []
    for stem in sorted(set(stems)):
        src = _find_video(root, stem)
        if src is None:
            continue
        n = _count_frames(src)
        out.append(VADVideo(stem, str(src), intervals_to_labels(n, ann.get(stem, [])),
                            meta={"label": stem.split("label_")[-1] if "label_" in stem else ""}))
    return out


# ---------------------------------------------------------------------------- generic
def _read_labels(path: Path, n_frames: int) -> np.ndarray:
    if path.suffix == ".npy":
        y = np.load(path).astype(np.uint8).ravel()
    elif path.suffix == ".json":
        raw = json.loads(path.read_text())
        y = (intervals_to_labels(n_frames, raw["intervals"]) if isinstance(raw, dict)
             else np.asarray(raw, np.uint8))
    else:
        txt = path.read_text().split()
        y = np.asarray([int(float(v)) for v in txt], np.uint8)
    if y.size != n_frames and n_frames > 0:
        y = np.resize(y, n_frames) if y.size > n_frames else np.r_[y, np.zeros(n_frames - y.size, np.uint8)]
    return y


def generic_frame_labels(video_root: str, label_root: str, fps: float = 30.0,
                         label_glob: str = "{name}.*") -> list[VADVideo]:
    """Videos (or frame folders) under ``video_root``; labels as ``label_root/<name>.npy|txt|json``
    with one 0/1 per frame (json may hold ``{"intervals": [[s, e], ...]}``)."""
    root, lroot = Path(video_root), Path(label_root)
    sources = sorted([p for p in root.iterdir() if p.suffix.lower() in VIDEO_EXT or
                      (p.is_dir() and any(q.suffix.lower() in IMG_EXT for q in p.iterdir()))])
    out = []
    for src in sources:
        name = _strip_ext(src.name)
        hits = sorted(lroot.glob(label_glob.format(name=name)))
        if not hits:
            continue
        n = _count_frames(src)
        out.append(VADVideo(name, str(src), _read_labels(hits[0], n), fps=fps if src.is_dir() else None))
    return out


# ---------------------------------------------------------------------------- A2Seek
def a2seek(samples_jsonl: str, video_template: str, frame_scale: float = 1.0,
           dataset_root: Optional[str] = None) -> list[VADVideo]:
    """A2Seek samples (``dataset/cot_single/test.jsonl``): each sample is a frame ``f_id`` of clip
    ``g_id/v_id`` with ``anomaly_name`` and ``objects{ref, bbox}`` in image pixels.

    ``video_template`` locates the clip video, e.g. ``data/A2Seek-Videos/{g_id}/{v_id}.mp4``, or is a
    ``.json`` file mapping ``"g_id/v_id"`` to a video path (written by ``scripts/datasets/fetch.py``).
    Sample images (``images/...``) are resolved against ``dataset_root`` (default: the folder that
    contains ``dataset/``), only to read the image size for box scaling.
    ``frame_scale`` maps ``f_id`` to a video frame index (verify on your copy of the data).
    Each clip becomes one streamed video; samples become probes answered causally at their frame.
    """
    from PIL import Image
    root = Path(dataset_root) if dataset_root else Path(samples_jsonl).resolve().parents[2]
    mapping = json.loads(Path(video_template).read_text()) if video_template.endswith(".json") else None
    groups = defaultdict(list)
    with open(samples_jsonl) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                groups[(r["g_id"], r["v_id"])].append(r)
    out = []
    for (g, v), rows in sorted(groups.items()):
        if mapping is not None:
            if f"{g}/{v}" not in mapping:
                continue
            src = Path(mapping[f"{g}/{v}"])
        else:
            src = Path(video_template.format(g_id=g, v_id=v))
        probes = []
        for r in sorted(rows, key=lambda x: int(x["f_id"])):
            img_wh = None
            if r.get("images"):
                p = Path(r["images"][0])
                p = p if p.is_absolute() else root / p
                try:
                    with Image.open(p) as im:
                        img_wh = im.size
                except OSError:
                    img_wh = None
            obj = r.get("objects") or {}
            box = (obj.get("bbox") or [None])[0]
            m = re.match(r"E\d{2}S(\d{2})", r.get("code", ""))
            probes.append({"sample_id": f"{g}/{v}/{int(r['f_id']):05d}",
                           "g_id": g, "v_id": v, "f_id": r["f_id"],
                           "frame": int(round(int(r["f_id"]) * frame_scale)),
                           "gt_category": r["anomaly_name"], "gt_box": box,
                           "image_wh": img_wh, "scene": m.group(1) if m else "00"})
        out.append(VADVideo(f"{g}/{v}", str(src), None, probes=probes))
    return out
