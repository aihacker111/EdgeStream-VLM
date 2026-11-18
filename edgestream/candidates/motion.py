"""Domain-agnostic candidate generator (runs once per window, e.g. every 1 s).

All motion thresholds are expressed in *object sizes per second*
(speed / sqrt(w*h)), so the same numbers work for a 6 px car seen from a UAV
and a 200 px motorbike seen from a CCTV pole.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

import numpy as np

from ..scene.state import SceneState, TrackHistory
from ..types import Candidate

SEVERITY_PRIOR = {
    "counter_flow": 0.9,
    "sudden_stop": 0.6,
    "stopped_in_flow": 0.6,
    "speeding": 0.5,
    "congestion_forming": 0.4,
}


@dataclass
class MotionRules:
    moving: float = 0.5            # sizes/s above which an object counts as moving
    stopped: float = 0.15          # sizes/s below which an object counts as stopped
    counter_cos: float = -0.5      # cos(track, local flow) below this = against the flow
    unidirectional_share: float = 0.75  # cell must carry >= this share of traffic in one heading
    min_cell_weight: float = 0.5   # flow cell must be well observed
    stop_min_s: float = 3.0        # stopped this long while the cell keeps moving
    sudden_drop_s: float = 1.5     # moving -> stopped within this window
    speeding_z: float = 3.0
    congestion_ratio: float = 0.4  # cell speed fell below 40% of its long-run mean
    congestion_min_tracks: int = 4
    window_s: float = 2.0          # velocity averaging window
    grace_s: float = 2.0           # keep a candidate open this long after its rule stops firing
    min_track_age_s: float = 2.0     # ignore tracks younger than this (edge-of-frame noise)
    enabled: tuple = field(default_factory=lambda: tuple(SEVERITY_PRIOR))
    classes: tuple = ()            # restrict to these class names (empty = all classes)


def _size(h: TrackHistory) -> float:
    p = h.points[-1]
    return float(np.sqrt(max(p[3] * p[4], 1.0)))


class MotionCandidateGenerator:
    def __init__(self, rules: MotionRules | None = None):
        self.r = rules or MotionRules()
        self.open: dict[tuple, Candidate] = {}
        self._ids = count(1)

    def step(self, scene: SceneState) -> list[Candidate]:
        t = scene.t
        fired: dict[tuple, dict] = {}
        for gid in scene.active_ids:
            h = scene.tracks[gid]
            if t - h.first_t < self.r.min_track_age_s or len(h.points) < 3:
                continue
            if self.r.classes and h.cls_name not in self.r.classes:
                continue
            for typ, info in self._track_rules(scene, h).items():
                fired[(typ, (gid,))] = info
        if "congestion_forming" in self.r.enabled:
            for key, info in self._congestion(scene).items():
                fired[key] = info

        for key, info in fired.items():
            cand = self.open.get(key)
            if cand is None:
                cand = Candidate(cand_id=next(self._ids), type_hint=key[0], track_ids=list(key[1]),
                                 t_start=t, t_last=t, s_rule=0.0, s_anom=0.0,
                                 severity_prior=SEVERITY_PRIOR[key[0]], cell=info["cell"])
                self.open[key] = cand
            cand.t_last = t
            cand.s_rule = float(np.clip(info["s_rule"], 0, 1))
            cand.s_anom = float(np.clip(info["s_anom"], 0, 1))
            cand.cell = info["cell"]
            cand.evidence = info.get("evidence", {})
            self._attach_perception_signals(scene, cand)

        for key in [k for k, c in self.open.items() if t - c.t_last > self.r.grace_s]:
            del self.open[key]
        return list(self.open.values())

    # ------------------------------------------------------------------ rules
    def _track_rules(self, scene: SceneState, h: TrackHistory) -> dict:
        r = self.r
        out = {}
        size = _size(h)
        v = h.mean_velocity(r.window_s)
        speed = float(np.hypot(*v)) / size
        _, cx, cy = h.points[-1][:3]
        ij = scene.cell_of(cx, cy)
        cell = scene.cell(ij)
        cell_speed = float(np.hypot(*cell.vec)) / size
        base = {"cell": ij, "evidence": {"speed_sizes_s": round(speed, 2),
                                         "cell_speed_sizes_s": round(cell_speed, 2)}}

        if "counter_flow" in r.enabled and cell.weight >= r.min_cell_weight and speed > r.moving \
                and cell.speed_mean / size > r.moving:
            dom, share = cell.dominant_direction()
            if share >= r.unidirectional_share:
                cos = float(np.dot(v, dom) / (np.linalg.norm(v) + 1e-9))
                if cos < r.counter_cos:
                    out["counter_flow"] = {**base, "s_rule": (-cos - 0.5) / 0.5 * share,
                                           "s_anom": min(speed / (2 * r.moving), 1.0),
                                           "evidence": {**base["evidence"], "cos_to_flow": round(cos, 2),
                                                        "flow_share": round(share, 2)}}

        speeds = h.speed_series(max(r.stop_min_s, r.sudden_drop_s) + 0.5) / size
        if speeds.size >= 3:
            if "stopped_in_flow" in r.enabled:
                stop_pts = h.recent(r.stop_min_s)
                long_enough = stop_pts and (stop_pts[-1][0] - stop_pts[0][0]) >= r.stop_min_s * 0.9
                if long_enough and np.all(speeds[-len(stop_pts):] < r.stopped) and cell_speed > r.moving:
                    out["stopped_in_flow"] = {**base, "s_rule": min(cell_speed / (2 * r.moving), 1.0),
                                              "s_anom": 1.0 - float(np.mean(speeds[-len(stop_pts):])) / r.stopped}
            if "sudden_stop" in r.enabled:
                k = max(2, int(r.sudden_drop_s * scene.fps))
                recent = speeds[-k:]
                if recent.size >= 2 and recent[0] > 2 * r.moving and recent[-1] < r.stopped:
                    out["sudden_stop"] = {**base, "s_rule": min(recent[0] / (4 * r.moving), 1.0),
                                          "s_anom": 1.0}

        if "speeding" in r.enabled and cell.weight >= r.min_cell_weight and speed > 2 * r.moving:
            sp_px = float(np.hypot(*v))
            z = (sp_px - cell.speed_mean) / (np.sqrt(cell.speed_var) + 1e-6)
            if z > r.speeding_z:
                out["speeding"] = {**base, "s_rule": min((z - r.speeding_z) / r.speeding_z, 1.0),
                                   "s_anom": min(z / (2 * r.speeding_z), 1.0),
                                   "evidence": {**base["evidence"], "speed_z": round(float(z), 2)}}
        return out

    def _congestion(self, scene: SceneState) -> dict:
        r = self.r
        out = {}
        for i in range(scene.gx):
            for j in range(scene.gy):
                c = scene.cells[i][j]
                if c.weight < r.min_cell_weight or c.speed_slow <= 1e-6:
                    continue
                if c.speed_mean < r.congestion_ratio * c.speed_slow:
                    members = scene.tracks_in_cell((i, j))
                    if len(members) >= r.congestion_min_tracks:
                        ratio = c.speed_mean / c.speed_slow
                        gids = tuple(sorted(h.gid for h in members))
                        out[("congestion_forming", ("cell", i, j))] = {
                            "cell": (i, j), "track_ids": gids,
                            "s_rule": 1.0 - ratio / r.congestion_ratio,
                            "s_anom": min(len(members) / (2 * r.congestion_min_tracks), 1.0),
                            "evidence": {"speed_ratio": round(ratio, 2), "n_tracks": len(members)}}
        return out

    # ------------------------------------------------------------------ signals
    @staticmethod
    def _attach_perception_signals(scene: SceneState, cand: Candidate) -> None:
        if cand.type_hint == "congestion_forming":
            members = scene.tracks_in_cell(cand.cell)
            cand.track_ids = sorted(h.gid for h in members)[:12]
        q, rec, kap = [], [], []
        for gid in cand.track_ids:
            h = scene.tracks.get(gid)
            if h is None:
                continue
            pts = h.recent(2.0)
            q += [p[7] for p in pts]
            rec += [float(p[8]) for p in pts]
            if h.kappa is not None:
                kap.append(h.kappa)
        cand.quality = float(np.mean(q)) if q else 0.0
        cand.recovered_frac = float(np.mean(rec)) if rec else 0.0
        cand.kappa = float(np.mean(kap)) if kap else None
