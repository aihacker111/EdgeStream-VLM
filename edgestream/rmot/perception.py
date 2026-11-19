"""Perception for RMOT: run AERO-MOT once per video and replay it for every expression.

Cache layout (``<cache>/<video_id>/``):
    tracks.jsonl   one FrameOutput per line (+ "affine": prev->curr background affine, 2x3)
    emb.npz        ReID embeddings, key "<rank>_<global_id>" (optional; used for stitching)

``load_mot_tracks`` replays an external tracker's output (MOT txt ``frame,id,x,y,w,h,score[,cls]``)
so the method can be run on the same tracks as other papers (perception-matched comparison).
"""
from __future__ import annotations

import json
import os

import numpy as np

from perception.types import FrameOutput, TrackOutput


def cache_dir(root: str, video_id: str) -> str:
    return os.path.join(root, video_id)


def has_cache(root: str, video_id: str, n_frames: int) -> bool:
    p = os.path.join(cache_dir(root, video_id), "tracks.jsonl")
    if not os.path.isfile(p):
        return False
    with open(p) as f:
        return sum(1 for _ in f) == n_frames


def run_perception(frame_paths: list, perceptor, out_root: str, video_id: str, fps: float,
                   ego_mode: str = "flow", save_emb: bool = True, progress: bool = True) -> None:
    import cv2
    from ..scene.ego_motion import EgoMotionEstimator
    d = cache_dir(out_root, video_id)
    os.makedirs(d, exist_ok=True)
    perceptor.reset()
    ego = EgoMotionEstimator(mode=ego_mode)
    embs = {}
    tmp = os.path.join(d, "tracks.jsonl.tmp")
    with open(tmp, "w") as f:
        for r, p in enumerate(frame_paths):
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(p)
            out = perceptor.process(img, timestamp=r / fps)
            A = ego.update(img, [t.tlwh for t in out.tracks])
            rec = out.to_dict()
            rec["frame_idx"] = r
            rec["affine"] = [[round(float(v), 6) for v in row] for row in np.asarray(A)]
            f.write(json.dumps(rec) + "\n")
            if save_emb:
                for t in out.tracks:
                    if t.embedding is not None and np.size(t.embedding) > 1:
                        embs[f"{r}_{t.global_id}"] = np.asarray(t.embedding, np.float16)
            if progress and r % 200 == 0:
                print(f"    [{video_id}] perception {r}/{len(frame_paths)}", flush=True)
    if save_emb and embs:
        np.savez_compressed(os.path.join(d, "emb.npz"), **embs)
    os.replace(tmp, os.path.join(d, "tracks.jsonl"))


def load_perception(out_root: str, video_id: str, use_emb: bool = True) -> tuple[list, list]:
    d = cache_dir(out_root, video_id)
    emb = {}
    ep = os.path.join(d, "emb.npz")
    if use_emb and os.path.isfile(ep):
        z = np.load(ep)
        emb = {k: z[k].astype(np.float32) for k in z.files}
    frames, affines = [], []
    with open(os.path.join(d, "tracks.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            rank = int(r["frame_idx"])
            tracks = [TrackOutput(
                global_id=t["global_id"], track_id=t["track_id"], cls_id=t["cls_id"], cls_name=t["cls_name"],
                tlwh=np.asarray(t["tlwh"], np.float32), velocity=np.asarray(t["velocity"], np.float32),
                quality=t["quality"], det_score=t["det_score"], age=t["age"], is_recovered=t["is_recovered"],
                kappa=t.get("kappa"), embedding=emb.get(f"{rank}_{t['global_id']}")) for t in r["tracks"]]
            frames.append(FrameOutput(rank, r["timestamp"], tuple(r["image_hw"]), tracks, r.get("num_detections", 0)))
            affines.append(np.asarray(r["affine"], np.float64) if r.get("affine") is not None else None)
    return frames, affines


def load_mot_tracks(path: str, n_frames: int, image_hw: tuple, fps: float, frame_base: int = 1,
                    class_names: dict | None = None, default_cls: str = "car") -> tuple[list, list]:
    """External tracks in MOT format. ``frame_base``: 1 if frames are 1-based (MOT convention)."""
    per = [[] for _ in range(n_frames)]
    rows = np.loadtxt(path, delimiter=",", ndmin=2) if os.path.getsize(path) else np.zeros((0, 7))
    for row in rows:
        r = int(row[0]) - frame_base
        if not 0 <= r < n_frames:
            continue
        cls = default_cls
        if row.shape[0] > 7 and class_names:
            cls = class_names.get(int(row[7]), default_cls)
        score = float(row[6]) if row.shape[0] > 6 else 1.0
        per[r].append(TrackOutput(int(row[1]), int(row[1]), 0, cls, np.asarray(row[2:6], np.float32),
                                  np.zeros(2, np.float32), score, score, 1, False))
    frames = [FrameOutput(r, r / fps, image_hw, per[r], len(per[r])) for r in range(n_frames)]
    return frames, [None] * n_frames
