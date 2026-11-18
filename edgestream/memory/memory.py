"""Two-tier streaming memory under a visual-token budget (a main contribution).

* Scene tier:  per-second statistics (text, ~free) + K pooled keyframes.
* Entity tier: N slots, one per object that the gate cares about, each holding a
  trajectory summary, up to 3 crops (start / middle / now) and an embedding.

Budget split:  B_entity = lambda_t * B_mem,  B_scene = (1 - lambda_t) * B_mem,
lambda_t = clip(lambda0 + gamma * n_high / n_ref, lambda_min, lambda_max).
Eviction score e = p * exp(-dt / T); an evicted slot is compressed into ONE line
of template text in the scene log (no VLM call).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..scene.state import SceneState
from ..types import Candidate


def visual_tokens(w: int, h: int, patch: int = 28) -> int:
    """Qwen2.5/3-VL style estimate: one token per 28x28 pixels after 2x2 merging."""
    return int(math.ceil(w / patch) * math.ceil(h / patch))


def _encode(img: np.ndarray, quality: int = 85) -> bytes:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return enc.tobytes() if ok else b""


def _decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


@dataclass
class MemoryConfig:
    budget_tokens: int = 2048
    crop_size: int = 224                 # crops are resized to fit this box
    crops_per_slot: int = 3
    keyframe_width: int = 448
    keyframe_every_s: float = 10.0
    crop_every_s: float = 1.0
    lambda0: float = 0.5
    gamma: float = 0.3
    lambda_min: float = 0.3
    lambda_max: float = 0.85
    n_ref: int = 4
    decay_s: float = 20.0
    reid_cos: float = 0.75
    reid_window_s: float = 15.0


@dataclass
class EntitySlot:
    gid: int
    cls_name: str
    priority: float
    first_t: float
    last_t: float
    type_hints: set = field(default_factory=set)
    crops: list = field(default_factory=list)        # (t, jpeg bytes, (w, h))
    last_crop_t: float = -1e9
    embedding: Optional[np.ndarray] = None
    linked_to: Optional[int] = None                    # re-identified earlier gid

    def pick_crops(self, k: int) -> list:
        if len(self.crops) <= k:
            return list(self.crops)
        idx = np.linspace(0, len(self.crops) - 1, k).round().astype(int)
        return [self.crops[i] for i in idx]


class StreamingMemory:
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.slots: dict[int, EntitySlot] = {}
        self.keyframes: deque = deque()                # (t, jpeg, (w, h))
        self.scene_log: deque = deque(maxlen=200)      # (t, text)
        self.archive: deque = deque(maxlen=64)         # evicted (t, gid, cls, embedding)
        self.lam = cfg.lambda0
        self._last_keyframe_t = -1e9
        self.crop_tokens = visual_tokens(cfg.crop_size, cfg.crop_size)
        self.keyframe_tokens = visual_tokens(cfg.keyframe_width, int(cfg.keyframe_width * 9 / 16))

    # ------------------------------------------------------------------ budget
    @property
    def slot_cost(self) -> int:
        return self.cfg.crops_per_slot * self.crop_tokens

    def capacities(self, n_high: int) -> tuple[int, int]:
        c = self.cfg
        self.lam = float(np.clip(c.lambda0 + c.gamma * n_high / max(c.n_ref, 1), c.lambda_min, c.lambda_max))
        n_slots = int(self.lam * c.budget_tokens // self.slot_cost)
        n_keyframes = int((1 - self.lam) * c.budget_tokens // self.keyframe_tokens)
        return max(n_slots, 1), max(n_keyframes, 1)

    def used_tokens(self) -> int:
        crops = sum(min(len(s.crops), self.cfg.crops_per_slot) for s in self.slots.values())
        return crops * self.crop_tokens + len(self.keyframes) * self.keyframe_tokens

    # ------------------------------------------------------------------ update
    def observe(self, t: float, frame: np.ndarray, scene: SceneState, cands: list[Candidate],
                tau: float) -> None:
        n_high = sum(1 for c in cands if c.priority >= tau)
        n_slots, n_keyframes = self.capacities(n_high)

        if t - self._last_keyframe_t >= self.cfg.keyframe_every_s:
            h, w = frame.shape[:2]
            kw = self.cfg.keyframe_width
            kh = int(round(h * kw / w))
            self.keyframes.append((t, _encode(cv2.resize(frame, (kw, kh), interpolation=cv2.INTER_AREA)), (kw, kh)))
            self._last_keyframe_t = t
        while len(self.keyframes) > n_keyframes:
            self.keyframes.popleft()

        wanted: dict[int, tuple[float, str]] = {}
        for c in cands:
            for gid in c.track_ids[: (3 if c.type_hint == "congestion_forming" else len(c.track_ids))]:
                if gid in scene.tracks and c.priority > wanted.get(gid, (-1.0, ""))[0]:
                    wanted[gid] = (c.priority, c.type_hint)
        for gid, (p, typ) in wanted.items():
            slot = self.slots.get(gid)
            hist = scene.tracks[gid]
            if slot is None:
                slot = EntitySlot(gid, hist.cls_name, p, t, t)
                slot.embedding = hist.embedding
                slot.linked_to = self._reidentify(t, hist)
                self.slots[gid] = slot
            slot.priority = max(slot.priority * 0.9, p)
            slot.last_t = t
            slot.type_hints.add(typ)
            if hist.embedding is not None:
                slot.embedding = hist.embedding
            if gid in scene.active_ids and t - slot.last_crop_t >= self.cfg.crop_every_s:
                self._add_crop(slot, t, frame, hist)

        self._evict(t, n_slots, scene)

    def _add_crop(self, slot: EntitySlot, t: float, frame: np.ndarray, hist) -> None:
        _, cx, cy, w, h = hist.points[-1][:5]
        side = max(w, h) * 2.0 + 16
        H, W = frame.shape[:2]
        x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        x1, y1 = int(min(cx + side / 2, W)), int(min(cy + side / 2, H))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        crop = frame[y0:y1, x0:x1].copy()
        bx0, by0 = int(cx - w / 2 - x0), int(cy - h / 2 - y0)
        cv2.rectangle(crop, (bx0, by0), (bx0 + int(w), by0 + int(h)), (0, 0, 255), max(1, int(side / 100)))
        s = self.cfg.crop_size / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                          interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        slot.crops.append((t, _encode(crop), (crop.shape[1], crop.shape[0])))
        slot.last_crop_t = t
        if len(slot.crops) > 2 * self.cfg.crops_per_slot:          # thin, keep first and last
            keep = np.linspace(0, len(slot.crops) - 1, self.cfg.crops_per_slot + 1).round().astype(int)
            slot.crops = [slot.crops[i] for i in sorted(set(keep))]

    def _evict(self, t: float, n_slots: int, scene: SceneState) -> None:
        def score(s: EntitySlot) -> float:
            return s.priority * math.exp(-(t - s.last_t) / self.cfg.decay_s)

        while len(self.slots) > n_slots:
            victim = min(self.slots.values(), key=score)
            self.scene_log.append((t, self._summarize(victim, scene)))
            if victim.embedding is not None:
                self.archive.append((t, victim.gid, victim.cls_name, victim.embedding))
            del self.slots[victim.gid]

    def _summarize(self, slot: EntitySlot, scene: SceneState) -> str:
        hist = scene.tracks.get(slot.gid)
        traj = describe_trajectory(hist) if hist is not None else "trajectory unavailable"
        types = "/".join(sorted(slot.type_hints)) or "none"
        return (f"{slot.cls_name} #{slot.gid}, {slot.first_t:.1f}-{slot.last_t:.1f}s, {traj}, "
                f"suspected {types}, not confirmed")

    def _reidentify(self, t: float, hist) -> Optional[int]:
        if hist.embedding is None:
            return None
        best, best_cos = None, self.cfg.reid_cos
        for ta, gid, cls, emb in self.archive:
            if t - ta > self.cfg.reid_window_s or cls != hist.cls_name:
                continue
            cos = float(np.dot(emb, hist.embedding))
            if cos > best_cos:
                best, best_cos = gid, cos
        return best

    # ------------------------------------------------------------------ read
    def context(self, cand: Candidate, scene: SceneState, max_tracks: int = 3) -> dict:
        tracks = []
        for gid in cand.track_ids[:max_tracks]:
            slot = self.slots.get(gid)
            hist = scene.tracks.get(gid)
            tracks.append({
                "gid": gid,
                "cls_name": hist.cls_name if hist else (slot.cls_name if slot else "object"),
                "trajectory": describe_trajectory(hist) if hist else "unavailable",
                "crops": [(_decode(b), wh, tc) for tc, b, wh in (slot.pick_crops(self.cfg.crops_per_slot) if slot else [])],
                "linked_to": slot.linked_to if slot else None,
            })
        recent_log = [txt for (tl, txt) in self.scene_log if tl >= scene.t - 60.0][-3:]
        return {"stats_text": scene.stats_text(30.0), "tracks": tracks, "scene_log": recent_log}


def describe_trajectory(hist, seconds: float = 10.0) -> str:
    pts = [p for p in hist.recent(seconds) if not p[8]] or list(hist.recent(seconds))
    if len(pts) < 2:
        return "just appeared"
    size = float(np.sqrt(max(pts[-1][3] * pts[-1][4], 1.0)))
    vx = np.mean([p[5] for p in pts])
    vy = np.mean([p[6] for p in pts])
    speed = float(np.hypot(vx, vy)) / size
    dur = pts[-1][0] - pts[0][0]
    if speed < 0.15:
        motion = "almost stationary"
    else:
        ang = (math.degrees(math.atan2(-vy, vx)) + 360) % 360
        dirs = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
        motion = f"moving {dirs[int(((ang + 22.5) % 360) // 45)]} (image frame) at {speed:.1f} body-lengths/s"
    speeds = [np.hypot(p[5], p[6]) / size for p in pts]
    trend = speeds[-1] - speeds[0]
    trend_txt = "decelerating" if trend < -0.3 else ("accelerating" if trend > 0.3 else "steady")
    return f"{motion}, {trend_txt}, observed {dur:.1f}s"
