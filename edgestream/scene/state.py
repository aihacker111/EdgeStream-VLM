"""Scene state: per-track histories, camera-motion compensation, flow field, density.

Everything here is cheap and runs every frame. Velocities are in pixels/second of
the original image and are *relative to the ground*: camera motion is estimated from
the background (``EgoMotionEstimator``) and subtracted at each object's position.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from perception.types import FrameOutput

from .ego_motion import EgoMotionEstimator


@dataclass
class TrackHistory:
    gid: int
    cls_name: str
    points: deque                          # (t, cx, cy, w, h, vx_rel, vy_rel, quality, recovered)
    first_t: float
    last_t: float
    kappa: float | None = None
    embedding: np.ndarray | None = None

    def recent(self, seconds: float) -> list:
        if not self.points:
            return []
        t_end = self.points[-1][0]
        return [p for p in self.points if p[0] >= t_end - seconds]

    def mean_velocity(self, seconds: float) -> np.ndarray:
        """Mean camera-relative velocity (px/s) over the last ``seconds``."""
        pts = [p for p in self.recent(seconds) if not p[8]]
        if not pts:
            return np.zeros(2)
        return np.array([np.mean([p[5] for p in pts]), np.mean([p[6] for p in pts])])

    def speed_series(self, seconds: float) -> np.ndarray:
        return np.array([np.hypot(p[5], p[6]) for p in self.recent(seconds)])


@dataclass
class FlowCell:
    vec: np.ndarray = field(default_factory=lambda: np.zeros(2))   # EMA of relative velocity
    speed_mean: float = 0.0
    speed_var: float = 0.0
    weight: float = 0.0                                            # EMA of observation count
    speed_slow: float = 0.0                                        # long-horizon mean speed (congestion)
    dir_hist: np.ndarray = field(default_factory=lambda: np.zeros(8))  # EMA of heading histogram

    def dominant_direction(self) -> tuple[np.ndarray, float]:
        """Unit vector of the dominant heading and the share of traffic within +-45 deg of it.

        A cell crossed by two opposite carriageways has share ~0.5 and is treated as
        bidirectional, so nobody there is flagged as counter-flow by geometry alone.
        """
        total = float(self.dir_hist.sum())
        if total <= 1e-9:
            return np.zeros(2), 0.0
        k = int(np.argmax(self.dir_hist))
        share = float(self.dir_hist[[(k - 1) % 8, k, (k + 1) % 8]].sum() / total)
        ang = k * np.pi / 4
        return np.array([np.cos(ang), np.sin(ang)]), share


class SceneState:
    def __init__(self, fps: float, grid: tuple[int, int] = (8, 6), history_s: float = 20.0,
                 ema: float = 0.05, slow_ema: float = 0.005, min_age_frames: int = 5,
                 ego_motion: str = "flow"):
        self.fps = float(fps)
        self.gx, self.gy = grid
        self.history_s = history_s
        self.ema, self.slow_ema = ema, slow_ema
        self.min_age = min_age_frames
        self.ego = EgoMotionEstimator(ego_motion)
        self.tracks: dict[int, TrackHistory] = {}
        self.cells = [[FlowCell() for _ in range(self.gy)] for _ in range(self.gx)]
        self.image_hw = (1, 1)
        self.t = 0.0
        self.density = 0
        self.density_ref = 0.0          # EMA of density (ρ_ref)
        self.ego_velocity = np.zeros(2)
        self.active_ids: set[int] = set()
        self._stats_log: deque = deque(maxlen=600)

    # ------------------------------------------------------------------ update
    def cell_of(self, cx: float, cy: float) -> tuple[int, int]:
        h, w = self.image_hw
        i = int(np.clip(cx / max(w, 1) * self.gx, 0, self.gx - 1))
        j = int(np.clip(cy / max(h, 1) * self.gy, 0, self.gy - 1))
        return i, j

    def update(self, out: FrameOutput, frame=None) -> None:
        self.t = out.timestamp
        self.image_hw = out.image_hw
        tracks = out.tracks
        self.density = len(tracks)
        self.density_ref = self.density if self.density_ref == 0 else \
            0.98 * self.density_ref + 0.02 * self.density

        A = self.ego.update(frame, [t.tlwh for t in tracks])
        vel = np.zeros((len(tracks), 2))
        for k, trk in enumerate(tracks):
            cx, cy = trk.center
            vel[k] = (np.asarray(trk.velocity, np.float64)
                      - EgoMotionEstimator.point_velocity(A, cx, cy)) * self.fps
        h, w = self.image_hw
        self.ego_velocity = EgoMotionEstimator.point_velocity(A, w / 2, h / 2) * self.fps
        mature = np.array([t.age >= self.min_age and not t.is_recovered for t in tracks], bool)

        self.active_ids = set()
        maxlen = int(self.history_s * self.fps) + 1
        for k, trk in enumerate(tracks):
            cx, cy = trk.center
            v_rel = vel[k]
            hist = self.tracks.get(trk.global_id)
            if hist is None:
                hist = TrackHistory(trk.global_id, trk.cls_name, deque(maxlen=maxlen), self.t, self.t)
                self.tracks[trk.global_id] = hist
            hist.points.append((self.t, float(cx), float(cy), float(trk.tlwh[2]), float(trk.tlwh[3]),
                                float(v_rel[0]), float(v_rel[1]), float(trk.quality), bool(trk.is_recovered)))
            hist.last_t = self.t
            hist.cls_name = trk.cls_name
            if trk.kappa is not None:
                hist.kappa = trk.kappa
            if trk.embedding is not None:
                hist.embedding = trk.embedding
            self.active_ids.add(trk.global_id)
            if mature[k]:
                self._update_cell(self.cell_of(cx, cy), v_rel)

        stale = [g for g, h in self.tracks.items() if self.t - h.last_t > self.history_s]
        for g in stale:
            del self.tracks[g]

    def _update_cell(self, ij: tuple[int, int], v: np.ndarray) -> None:
        c = self.cells[ij[0]][ij[1]]
        a = self.ema
        sp = float(np.hypot(*v))
        if c.weight == 0:
            c.vec, c.speed_mean, c.speed_slow = v.copy(), sp, sp
        else:
            c.vec = (1 - a) * c.vec + a * v
            d = sp - c.speed_mean
            c.speed_mean += a * d
            c.speed_var = (1 - a) * (c.speed_var + a * d * d)
            c.speed_slow = (1 - self.slow_ema) * c.speed_slow + self.slow_ema * sp
        if sp > 1e-6:
            b = int(np.round(np.arctan2(v[1], v[0]) / (np.pi / 4))) % 8
            c.dir_hist *= (1 - a)
            c.dir_hist[b] += a
        c.weight = min(1.0, c.weight + a)

    # ------------------------------------------------------------------ queries
    def cell(self, ij: tuple[int, int]) -> FlowCell:
        return self.cells[ij[0]][ij[1]]

    def tracks_in_cell(self, ij: tuple[int, int]) -> list[TrackHistory]:
        out = []
        for gid in self.active_ids:
            h = self.tracks[gid]
            _, cx, cy = h.points[-1][:3]
            if self.cell_of(cx, cy) == ij:
                out.append(h)
        return out

    def snapshot(self) -> dict:
        """Compact per-second statistics (the scene tier of memory stores these)."""
        classes: dict[str, int] = {}
        speeds = []
        for gid in self.active_ids:
            h = self.tracks[gid]
            classes[h.cls_name] = classes.get(h.cls_name, 0) + 1
            p = h.points[-1]
            speeds.append(np.hypot(p[5], p[6]))
        snap = {
            "t": round(self.t, 2), "density": self.density, "density_ref": round(self.density_ref, 1),
            "classes": classes,
            "mean_speed_px_s": round(float(np.mean(speeds)), 1) if speeds else 0.0,
            "ego_velocity_px_s": [round(float(v), 1) for v in self.ego_velocity],
        }
        self._stats_log.append(snap)
        return snap

    def stats_text(self, seconds: float = 30.0) -> str:
        recent = [s for s in self._stats_log if s["t"] >= self.t - seconds]
        if not recent:
            return "No scene statistics yet."
        dens = [s["density"] for s in recent]
        spd = [s["mean_speed_px_s"] for s in recent]
        cls = recent[-1]["classes"]
        cls_txt = ", ".join(f"{v} {k}" for k, v in sorted(cls.items(), key=lambda kv: -kv[1]))
        return (f"Last {int(seconds)} s: {int(np.mean(dens))} objects on average (now: {cls_txt}); "
                f"mean speed {np.mean(spd):.0f} px/s, trend {spd[-1] - spd[0]:+.0f} px/s.")
