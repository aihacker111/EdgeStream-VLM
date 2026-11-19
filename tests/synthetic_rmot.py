"""Synthetic referring-MOT world for tests: GT objects, fragmenting tracker, moving camera."""
from __future__ import annotations

import numpy as np

from perception.types import FrameOutput, TrackOutput


class SynthVideo:
    """Duck-types the parts of ``RMOTVideo`` that GTOracle and the GT writer use."""

    def __init__(self, gt: dict, n: int, referred: dict):
        self.gt, self.n, self.referred = gt, n, referred      # rank -> (K,5); rank -> set(ids)

    def __len__(self):
        return self.n

    def gt_boxes(self, rank):
        return self.gt.get(rank, np.zeros((0, 5)))

    def referred_ids(self, expr, rank):
        return self.referred.get(rank, set())

    def gt_rows(self):
        rows = []
        for r in range(self.n):
            for oid, x, y, w, h in self.gt_boxes(r):
                if int(oid) in self.referred_ids(None, r):
                    rows.append((r + 1, int(oid), x, y, w, h))
        return rows


def make_world(seed=0, n=150, fps=10.0, n_cars=14, n_people=6, cam_speed=(3.0, 0.0), frag_rate=0.015,
               miss=(3, 8), fp_tracks=3):
    """Expression: "red cars moving" (red = hidden attribute, only the oracle knows it).

    Returns perception list, affines, SynthVideo, expression text.
    """
    rng = np.random.default_rng(seed)
    W, H = 1280, 720
    objs = []
    for i in range(n_cars + n_people):
        car = i < n_cars
        moving = rng.random() < 0.6
        v = rng.uniform(-8, 8, 2) if moving else np.zeros(2)
        if moving and np.linalg.norm(v) < 3:
            v = v / max(np.linalg.norm(v), 1e-6) * 4
        objs.append({"id": i + 1, "car": car, "red": car and rng.random() < 0.4, "moving": moving,
                     "p": rng.uniform([100, 100], [W - 100, H - 100]), "v": v,
                     "wh": np.array([34, 20]) if car else np.array([10, 16]),
                     "t0": int(rng.integers(0, n // 3)), "t1": int(rng.integers(2 * n // 3, n))})
    cam = np.zeros(2)
    gt, referred, perc, affines = {}, {}, [], []
    gid_of = {}
    next_gid = [1]
    hidden = {}

    def new_gid(oid):
        gid_of[oid] = next_gid[0]
        next_gid[0] += 1

    fps_tracks = [{"gid": 10000 + k, "p": rng.uniform([0, 0], [W, H]), "t0": int(rng.integers(0, max(n - 20, 1))),
                   "len": int(rng.integers(5, 15))} for k in range(fp_tracks)]
    for r in range(n):
        prev_cam = cam.copy()
        cam = cam + np.asarray(cam_speed)
        A = np.array([[1, 0, -(cam - prev_cam)[0]], [0, 1, -(cam - prev_cam)[1]]], np.float64)
        affines.append(A)
        rows, ref, tracks = [], set(), []
        for o in objs:
            if not (o["t0"] <= r <= o["t1"]):
                continue
            world = o["p"] + o["v"] * (r - o["t0"])
            img = world - cam
            x, y = img - o["wh"] / 2
            rows.append([o["id"], x, y, *o["wh"]])
            if o["car"] and o["red"] and o["moving"]:
                ref.add(o["id"])
            if o["id"] not in gid_of:
                new_gid(o["id"])
            if hidden.get(o["id"], 0) > 0:
                hidden[o["id"]] -= 1
                if hidden[o["id"]] == 0:
                    new_gid(o["id"])                  # tracker re-acquires with a new id
                continue
            if r > o["t0"] + 3 and rng.random() < frag_rate:
                hidden[o["id"]] = int(rng.integers(*miss))
                continue
            box = np.array([x, y, *o["wh"]], np.float32) + rng.normal(0, 1.0, 4).astype(np.float32)
            tracks.append(TrackOutput(gid_of[o["id"]], gid_of[o["id"]], 3 if o["car"] else 0,
                                      "car" if o["car"] else "pedestrian", box, np.zeros(2, np.float32),
                                      0.8, 0.8, 10, False))
        for fp in fps_tracks:
            if fp["t0"] <= r < fp["t0"] + fp["len"]:
                tracks.append(TrackOutput(fp["gid"], fp["gid"], 3, "car",
                                          np.array([*fp["p"], 30, 20], np.float32), np.zeros(2, np.float32),
                                          0.3, 0.4, 3, False))
        gt[r] = np.array(rows, np.float64).reshape(-1, 5)
        referred[r] = ref
        perc.append(FrameOutput(r, r / fps, (H, W), tracks, len(tracks)))
    return perc, affines, SynthVideo(gt, n, referred), "red cars moving"


def write_fake_aerialmind(root, video_ids=("uav0000009_03358_v",), n=60, seed=0, with_tracks=True):
    """Tiny dataset in the AerialMind layout (images, labels_with_ids, expression JSONs) plus
    MOT-format tracks of the synthetic tracker in ``root/tracks``. Returns the tracks dir."""
    import json
    import os

    import cv2
    tracks_dir = os.path.join(root, "tracks")
    os.makedirs(tracks_dir, exist_ok=True)
    for k, vid in enumerate(video_ids):
        perc, aff, sv, expr = make_world(seed + k, n=n)
        img_dir = os.path.join(root, "image_02", "Visdrone", vid)
        lab_dir = os.path.join(root, "labels_with_ids", "image_02", vid)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        rng = np.random.default_rng(seed)
        bg = (rng.integers(60, 120, (720, 1280, 3))).astype(np.uint8)
        W, Hh = 1280, 720
        lines = []
        for r in range(n):
            img = bg.copy()
            rows = []
            for oid, x, y, w, h in sv.gt_boxes(r):
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (40, 40, 200), -1)
                rows.append(f"0 {int(oid)} {x / W:.6f} {y / Hh:.6f} {w / W:.6f} {h / Hh:.6f}")
            cv2.imwrite(os.path.join(img_dir, f"{r + 1:07d}.jpg"), img)
            with open(os.path.join(lab_dir, f"{r + 1:07d}.txt"), "w") as f:
                f.write("\n".join(rows) + ("\n" if rows else ""))
            for t in perc[r].tracks:
                x, y, w, h = t.tlwh
                lines.append(f"{r + 1},{t.global_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{t.det_score:.3f}")
        with open(os.path.join(tracks_dir, f"{vid}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        ed = os.path.join(root, "expression", vid)
        os.makedirs(ed, exist_ok=True)
        lab = {str(r + 1): sorted(int(i) for i in sv.referred_ids(None, r)) for r in range(n) if sv.referred_ids(None, r)}
        with open(os.path.join(ed, "Red cars moving.json"), "w") as f:
            json.dump({"label": lab, "sentence": "red cars moving", "video_name": vid}, f)
        allc = {}
        for r in range(n):
            ids = [int(o[0]) for o in sv.gt_boxes(r) if int(o[0]) <= 14]
            if ids:
                allc[str(r + 1)] = ids
        with open(os.path.join(ed, "Cars.json"), "w") as f:
            json.dump({"label": allc, "sentence": "cars"}, f)
    return tracks_dir
