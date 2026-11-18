"""Object-centric anomaly candidates for surveillance VAD (UAV and CCTV).

Cheap tier, causal, training-free. Each rule proposes a *hypothesis* (``type_hint``)
that selects which event cards the VLM sees; the VLM may still answer any category.

    human rules     running, falling, lying_still, loitering, close_interaction, crowd_change
    any class       location_rarity (ClassLocationMap), kinematic_outlier (KinematicBank)
    no track        unexplained_motion (ResidualMotion), static_change (StaticChange)

All thresholds are scale-free: speeds in object sizes per second, distances in object sizes,
and human speed / shape references are learned online from the people in the scene, so the
same numbers work for a 12 px pedestrian seen from 80 m and a 200 px one on a CCTV pole.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from itertools import count

import numpy as np

from ..scene.state import SceneState, TrackHistory
from ..types import Candidate
from ..vad.normality import (ClassLocationMap, KinematicBank, StabilizedFrame, class_group)
from ..vad.residual import ResidualMotion, StaticChange

SEVERITY_PRIOR = {
    "running": 0.5, "falling": 0.9, "lying_still": 0.9, "loitering": 0.4, "close_interaction": 0.8,
    "crowd_change": 0.6, "location_rarity": 0.5, "kinematic_outlier": 0.4, "unexplained_motion": 0.45,
    "static_change": 0.5,
}


@dataclass
class ObjectRules:
    window_s: float = 2.0
    min_track_age_s: float = 1.0
    moving: float = 0.5              # sizes/s
    stopped: float = 0.2
    run_ratio: float = 2.2           # running = this many times the typical walking speed ...
    run_min: float = 2.0             # ... and at least this fast (sizes/s)
    fall_log_aspect: float = 0.47    # |Δ log(h/w)| >= log(1.6) within ~2 s
    lying_log_aspect: float = 0.59   # |log(h/w) - typical| >= log(1.8) ...
    lying_min_s: float = 3.0         # ... while stationary this long
    loiter_min_s: float = 10.0
    loiter_tortuosity: float = 2.5
    pair_dist: float = 1.5           # centre distance in mean object sizes
    pair_min_s: float = 1.0
    pair_jitter: float = 0.6         # sizes/s
    crowd_min: int = 4
    crowd_lookback_s: float = 4.0
    rarity_nats: float = 1.5         # ≈ 4.5x rarer than the class prior
    rarity_min_exposure: float = 60.0  # pooled cell-seconds of observation before judging
    knn_ratio: float = 2.0           # x the 95th-percentile normal k-NN distance
    knn_persist: int = 3             # windows in a row
    blob_min_s: float = 1.0
    static_min_s: float = 5.0
    grace_s: float = 2.0
    enabled: tuple = field(default_factory=lambda: tuple(SEVERITY_PRIOR))


def _tlwh(p) -> tuple:
    return (p[1] - p[3] / 2, p[2] - p[4] / 2, p[3], p[4])


def track_descriptor(h: TrackHistory, scene: SceneState, window_s: float = 2.0) -> dict | None:
    pts = [p for p in h.recent(window_s) if not p[8]]
    if len(pts) < 3:
        return None
    long_pts = [p for p in h.recent(10.0) if not p[8]] or pts
    size = float(np.median([math.sqrt(max(p[3] * p[4], 1.0)) for p in long_pts]))
    V = np.array([[p[5], p[6]] for p in pts])
    v = V.mean(0)
    speed = float(np.hypot(*v)) / size
    # robust spread (MAD): a tracker glitch of a few frames must not look like erratic motion
    jitter = float(np.hypot(*(1.4826 * np.median(np.abs(V - np.median(V, 0)), 0)))) / size
    sp = np.hypot(V[:, 0], V[:, 1])
    mv = sp / size > 0.5
    turn = 0.0
    if mv.sum() >= 3:
        ang = np.unwrap(np.arctan2(V[mv, 1], V[mv, 0]))
        tm = np.array([p[0] for p, m in zip(pts, mv) if m])
        dt = np.maximum(np.diff(tm), 1e-3)
        turn = float(np.median(np.abs(np.diff(ang)) / dt))      # rad/s, robust to single flips
    H, W = scene.image_hw
    def _cut(p):                                     # box truncated by the image border
        return p[1] - p[3] / 2 < 2 or p[2] - p[4] / 2 < 2 or p[1] + p[3] / 2 > W - 2 or p[2] + p[4] / 2 > H - 2
    inner = [p for p in pts if not _cut(p)] or pts
    la = np.log(np.array([max(p[4], 1.0) / max(p[3], 1.0) for p in inner]))
    k = max(1, len(la) // 4)
    la_now = float(np.median(la[-k:]))
    old = [p for p in h.recent(window_s + 1.5) if not p[8] and not _cut(p) and p[0] <= pts[0][0] + 0.5 * window_s]
    if _cut(pts[-1]):
        old = []                                     # do not read a shape change into a clipped box
    la_old = float(np.median([math.log(max(p[4], 1.0) / max(p[3], 1.0)) for p in old])) if old else la_now
    # ego-compensated path over a long window for tortuosity / radius of gyration
    lp = [p for p in h.recent(10.0) if not p[8]]
    if len(lp) >= 3:
        t = np.array([p[0] for p in lp])
        Vl = np.array([[p[5], p[6]] for p in lp])
        dt = np.diff(t)[:, None]
        steps = Vl[1:] * dt
        path = float(np.hypot(steps[:, 0], steps[:, 1]).sum())
        net = float(np.hypot(*steps.sum(0)))
        P = np.cumsum(steps, 0)
        rog = float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())) / size
        tort = path / max(net, size)
        span = float(t[-1] - t[0])
    else:
        tort, rog, span = 1.0, 0.0, 0.0
    _, cx, cy = h.points[-1][:3]
    cell = scene.cell(scene.cell_of(cx, cy))
    flow_dev = float(np.hypot(*(v - cell.vec))) / size if cell.weight > 0.3 else speed
    return {"size": size, "speed": speed, "jitter": jitter, "turn": turn, "log_aspect": la_now,
            "d_log_aspect": la_now - la_old, "tortuosity": tort, "rog": rog, "span": span,
            "flow_dev": flow_dev, "v": v, "cx": cx, "cy": cy,
            "x": np.array([math.log1p(speed), jitter, min(turn, 10.0), la_now, la_now - la_old,
                           math.log(max(tort, 1.0)), math.log1p(flow_dev)], np.float64)}


class ObjectAnomalyGenerator:
    def __init__(self, rules: ObjectRules | None = None, ids=None, bank: KinematicBank | None = None,
                 location_map: ClassLocationMap | None = None, use_pixels: bool = True):
        self.r = rules or ObjectRules()
        self._ids = ids or count(1)
        self.bank = bank if bank is not None else KinematicBank()   # an empty bank is falsy (__len__)
        self.locmap = location_map if location_map is not None else ClassLocationMap(min_exposure=self.r.rarity_min_exposure)
        self.stab = StabilizedFrame()
        self.use_pixels = use_pixels
        self.residual = ResidualMotion() if use_pixels else None
        self.static = StaticChange() if use_pixels else None
        self.open: dict[tuple, Candidate] = {}
        self.human_speed = 1.0           # typical walking speed (sizes/s), learned online
        self.human_aspect = None          # typical log(h/w) of people in this view
        self.pairs: dict[tuple, float] = {}
        self.crowd_hist: deque = deque()
        self.crowd_ema: dict = {}
        self.last_desc: dict[int, dict] = {}
        self.kin_scores: dict[int, float] = {}        # persistence-filtered (min over recent windows)
        self._kin_hist: dict[int, deque] = {}
        self.frozen = False               # True: do not update normality (evaluation on a fitted bank)
        self.dense_score = 0.0            # max per-object normality score this window, in [0, 1]
        self.dense_gid = None

    # ------------------------------------------------------------------ per frame
    def observe_frame(self, scene: SceneState, frame) -> None:
        A = scene.ego.last_affine
        h, w = scene.image_hw
        if self.stab.update(A, (w, h)):
            self.locmap.reset()
        boxes = [_tlwh(scene.tracks[g].points[-1]) for g in scene.active_ids]
        if frame is not None and self.use_pixels:
            self.residual.update(frame, A, boxes)
            self.static.update(scene.t, frame, A, boxes)
        if not self.frozen:
            self.locmap.observe_view(scene.t, self.stab)
            for gid in scene.active_ids:
                hist = scene.tracks[gid]
                if scene.t - hist.first_t < self.r.min_track_age_s:
                    continue
                p = hist.points[-1]
                key = self.locmap.key(*self.stab.to_ref(p[1], p[2]), w)
                self.locmap.observe(scene.t, gid, class_group(hist.cls_name), key)

    # ------------------------------------------------------------------ per window
    def step(self, scene: SceneState) -> list[Candidate]:
        r, t = self.r, scene.t
        fired: dict[tuple, dict] = {}
        descs = {}
        for gid in scene.active_ids:
            hist = scene.tracks[gid]
            if t - hist.first_t < r.min_track_age_s:
                continue
            d = track_descriptor(hist, scene, r.window_s)
            if d is not None:
                descs[gid] = d
        self.last_desc = descs
        self._update_human_refs(scene, descs)

        for gid, d in descs.items():
            hist = scene.tracks[gid]
            grp = class_group(hist.cls_name)
            if grp == "human":
                self._human_rules(t, hist, d, fired)
            self._rarity(scene, gid, grp, d, fired)
            self._kinematic(gid, grp, d, fired)
        self._dense(scene, descs)
        self._pairs(t, scene, descs, fired)
        self._crowd(t, scene, descs, fired)
        if self.use_pixels:
            self._pixel_cands(t, scene, fired)

        for key, info in fired.items():
            typ = key[0]
            if typ not in r.enabled:
                continue
            cand = self.open.get(key)
            if cand is None:
                cand = Candidate(cand_id=next(self._ids), type_hint=typ, track_ids=list(info.get("tracks", [])),
                                 t_start=t, t_last=t, s_rule=0.0, s_anom=0.0,
                                 severity_prior=SEVERITY_PRIOR[typ], cell=info.get("cell", (0, 0)))
                self.open[key] = cand
            cand.t_last = t
            cand.s_rule = float(np.clip(info["s_rule"], 0, 1))
            cand.s_anom = float(np.clip(info["s_anom"], 0, 1))
            cand.cell = info.get("cell", cand.cell)
            cand.evidence = info.get("evidence", {})
            cand.box = info.get("box")
            if "tracks" in info:
                cand.track_ids = list(info["tracks"])
            self._signals(scene, cand, info)

        for key in [k for k, c in self.open.items() if t - c.t_last > r.grace_s]:
            del self.open[key]
        live = {g for g in scene.tracks}
        for gid in [g for g in self.locmap.own if g not in live]:
            self.locmap.forget(gid)
        for gid in [g for g in self._kin_hist if g not in scene.active_ids]:
            del self._kin_hist[gid]
            self.kin_scores.pop(gid, None)
        return list(self.open.values())

    # ------------------------------------------------------------------ references
    def _update_human_refs(self, scene, descs) -> None:
        hs = [d for g, d in descs.items() if class_group(scene.tracks[g].cls_name) == "human"]
        walking = [d["speed"] for d in hs if d["speed"] > self.r.moving]
        if len(walking) >= 2:
            self.human_speed = 0.95 * self.human_speed + 0.05 * float(np.median(walking))
        if hs:
            med = float(np.median([d["log_aspect"] for d in hs]))
            self.human_aspect = med if self.human_aspect is None else 0.95 * self.human_aspect + 0.05 * med

    # ------------------------------------------------------------------ human rules
    def _human_rules(self, t, hist, d, fired) -> None:
        r = self.r
        gid = hist.gid
        base = {"tracks": [gid]}
        thr = max(r.run_min, r.run_ratio * self.human_speed)
        if d["speed"] >= thr:
            fired[("running", (gid,))] = {**base, "s_rule": 0.5 + (d["speed"] / thr - 1.0),
                                          "s_anom": min(d["speed"] / (2 * thr), 1.0),
                                          "evidence": {"speed_sizes_s": round(d["speed"], 2),
                                                       "walking_ref": round(self.human_speed, 2)}}
        if abs(d["d_log_aspect"]) >= r.fall_log_aspect and d["speed"] < 2 * r.moving:
            fired[("falling", (gid,))] = {**base, "s_rule": abs(d["d_log_aspect"]) / (2 * r.fall_log_aspect),
                                          "s_anom": 0.8,
                                          "evidence": {"aspect_change": round(math.exp(abs(d["d_log_aspect"])), 2)}}
        if self.human_aspect is not None and abs(d["log_aspect"] - self.human_aspect) >= r.lying_log_aspect:
            still = [p for p in hist.recent(r.lying_min_s) if not p[8]]
            if still and still[-1][0] - still[0][0] >= 0.9 * r.lying_min_s and d["speed"] < r.stopped:
                fired[("lying_still", (gid,))] = {**base, "s_rule": abs(d["log_aspect"] - self.human_aspect)
                                                  / (2 * r.lying_log_aspect), "s_anom": 0.9,
                                                  "evidence": {"aspect_vs_people": round(
                                                      math.exp(abs(d["log_aspect"] - self.human_aspect)), 2),
                                                      "still_s": round(still[-1][0] - still[0][0], 1)}}
        if d["span"] >= r.loiter_min_s * 0.9 and d["tortuosity"] >= r.loiter_tortuosity and d["rog"] < 6.0:
            fired[("loitering", (gid,))] = {**base, "s_rule": (d["tortuosity"] - 2.0) / 3.0,
                                            "s_anom": min(d["span"] / (2 * r.loiter_min_s), 1.0),
                                            "evidence": {"tortuosity": round(d["tortuosity"], 1),
                                                         "dwell_s": round(d["span"], 1)}}

    def _pairs(self, t, scene, descs, fired) -> None:
        r = self.r
        hum = [(g, d) for g, d in descs.items() if class_group(scene.tracks[g].cls_name) == "human"]
        seen = set()
        if 2 <= len(hum) <= 300:
            P = np.array([[d["cx"], d["cy"]] for _, d in hum])
            S = np.array([d["size"] for _, d in hum])
            D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
            near = D < r.pair_dist * 0.5 * (S[:, None] + S[None, :])
            ii, jj = np.where(np.triu(near, 1))
            for i, j in zip(ii, jj):
                (ga, da), (gb, db) = hum[i], hum[j]
                key = (min(ga, gb), max(ga, gb))
                seen.add(key)
                self.pairs.setdefault(key, t)
                if t - self.pairs[key] < r.pair_min_s:
                    continue
                jit = 0.5 * (da["jitter"] + db["jitter"])
                rel = float(np.hypot(*(da["v"] - db["v"]))) / (0.5 * (da["size"] + db["size"]))
                if jit >= r.pair_jitter or (rel >= 2 * r.pair_jitter and max(da["turn"], db["turn"]) > 2.0):
                    fired[("close_interaction", key)] = {
                        "tracks": list(key), "s_rule": jit / (2 * r.pair_jitter), "s_anom": min(rel / 4.0, 1.0),
                        "evidence": {"jitter": round(jit, 2), "relative_speed": round(rel, 2),
                                     "together_s": round(t - self.pairs[key], 1)}}
        for key in [k for k in self.pairs if k not in seen]:
            del self.pairs[key]

    def _crowd(self, t, scene, descs, fired) -> None:
        r = self.r
        h, w = scene.image_hw
        counts, members = {}, {}
        for g, d in descs.items():
            if class_group(scene.tracks[g].cls_name) != "human":
                continue
            ij = (int(np.clip(d["cx"] / w * 4, 0, 3)), int(np.clip(d["cy"] / h * 3, 0, 2)))
            counts[ij] = counts.get(ij, 0) + 1
            members.setdefault(ij, []).append(g)
        self.crowd_hist.append((t, counts, members))
        while self.crowd_hist and t - self.crowd_hist[0][0] > r.crowd_lookback_s:
            self.crowd_hist.popleft()
        if t - self.crowd_hist[0][0] < 0.75 * r.crowd_lookback_s:
            return
        _, before, before_m = self.crowd_hist[0]
        t_before = self.crowd_hist[0][0]
        before_ids = {g for ms in before_m.values() for g in ms}
        for ij in set(counts) | set(before):
            now, old = counts.get(ij, 0), before.get(ij, 0)
            # arrivals = people who already existed elsewhere and walked in (not people who just appeared)
            arrived = [g for g in members.get(ij, []) if g in before_ids and g not in before_m.get(ij, [])
                       and scene.tracks[g].first_t <= t_before]
            ema = self.crowd_ema.get(ij, float(now))
            self.crowd_ema[ij] = 0.98 * ema + 0.02 * now
            if now >= max(r.crowd_min, 2 * ema + 2) and len(arrived) >= max(3, old):
                fired[("crowd_change", ("cell",) + ij)] = {
                    "tracks": sorted(members[ij])[:8], "cell": ij, "s_rule": len(arrived) / max(now, 1),
                    "s_anom": min(now / (2 * r.crowd_min), 1.0),
                    "evidence": {"gathering": f"{old}->{now} people, {len(arrived)} walked in"}}
            elif old >= r.crowd_min:
                still = [g for g in before_m.get(ij, []) if g in descs]
                if len(still) >= 3:
                    V = np.array([descs[g]["v"] / descs[g]["size"] for g in still])
                    sp = np.hypot(V[:, 0], V[:, 1])
                    coh = float(np.hypot(*V.sum(0)) / (sp.sum() + 1e-6))
                    if np.median(sp) > r.run_ratio * self.human_speed and coh < 0.5:
                        fired[("crowd_change", ("cell",) + ij)] = {
                            "tracks": sorted(still)[:8], "cell": ij, "s_rule": 1.0 - coh,
                            "s_anom": min(float(np.median(sp)) / (4 * self.human_speed), 1.0),
                            "evidence": {"dispersal": f"{len(still)} people scattering"}}

    # ------------------------------------------------------------------ normality models
    def _rarity(self, scene, gid, grp, d, fired) -> None:
        r = self.r
        w = scene.image_hw[1]
        key = self.locmap.key(*self.stab.to_ref(d["cx"], d["cy"]), w)
        sur, mass = self.locmap.neighbourhood_surprise(gid, grp, key)
        if mass >= r.rarity_min_exposure and sur >= r.rarity_nats:
            fired[("location_rarity", (gid,))] = {
                "tracks": [gid], "s_rule": 0.5 + (sur - r.rarity_nats) / 3.0,
                "s_anom": min(sur / (2 * r.rarity_nats), 1.0),
                "evidence": {"rarer_than_usual_x": round(math.exp(sur), 1), "class_group": grp,
                             "cell_watched_s": int(mass)}}

    def _kinematic(self, gid, grp, d, fired) -> None:
        raw = self.bank.score(grp, d["x"])
        if not self.frozen:
            self.bank.insert(grp, d["x"], raw)
        hist = self._kin_hist.setdefault(gid, deque(maxlen=self.r.knn_persist))
        hist.append(raw if raw is not None else 0.0)
        # a box glitch lasts one window; an anomalous motion persists: require it in every recent window
        s = min(hist) if len(hist) == hist.maxlen else 0.0
        self.kin_scores[gid] = s
        if raw is not None and s >= self.r.knn_ratio:
            fired[("kinematic_outlier", (gid,))] = {
                "tracks": [gid], "s_rule": 0.5 + (s - self.r.knn_ratio) / (2 * self.r.knn_ratio),
                "s_anom": min((s - 1.0) / 3.0, 1.0),
                "evidence": {"knn_ratio": round(s, 2), "class_group": grp}}

    def _dense(self, scene, descs) -> None:
        """Soft per-object anomaly score for every track (not only rule hits): frame scores need it."""
        r, w = self.r, scene.image_hw[1]
        best, arg = 0.0, None
        for gid, d in descs.items():
            grp = class_group(scene.tracks[gid].cls_name)
            kin = self.kin_scores.get(gid, 0.0)
            s_k = float(np.clip((kin - 1.0) / (2 * (r.knn_ratio - 1.0)), 0, 1))
            sur, mass = self.locmap.neighbourhood_surprise(
                gid, grp, self.locmap.key(*self.stab.to_ref(d["cx"], d["cy"]), w))
            s_l = float(np.clip(sur / (2 * r.rarity_nats), 0, 1)) if mass >= r.rarity_min_exposure else 0.0
            s = max(s_k, s_l)
            if s > best:
                best, arg = s, gid
        self.dense_score, self.dense_gid = best, arg

    def _pixel_cands(self, t, scene, fired) -> None:
        r = self.r
        for b in self.residual.blobs(t):
            if b.persist_s >= r.blob_min_s:
                cx, cy = b.box[0] + b.box[2] / 2, b.box[1] + b.box[3] / 2
                fired[("unexplained_motion", ("blob", b.bid))] = {
                    "tracks": [], "box": b.box, "cell": scene.cell_of(cx, cy), "s_rule": b.strength,
                    "s_anom": min(b.persist_s / 4.0, 1.0), "quality": b.strength,
                    "evidence": {"untracked_motion_area_pct": round(100 * b.area_frac, 2),
                                 "persist_s": round(b.persist_s, 1)}}
        boxes = [_tlwh(scene.tracks[g].points[-1]) for g in scene.active_ids]
        for b in self.static.blobs(t, boxes):
            if b.persist_s >= r.static_min_s:
                cx, cy = b.box[0] + b.box[2] / 2, b.box[1] + b.box[3] / 2
                fired[("static_change", ("blob", b.bid))] = {
                    "tracks": [], "box": b.box, "cell": scene.cell_of(cx, cy), "s_rule": 0.7,
                    "s_anom": min(b.persist_s / (3 * r.static_min_s), 1.0), "quality": 0.8,
                    "evidence": {"new_static_area_pct": round(100 * b.area_frac, 3),
                                 "persist_s": round(b.persist_s, 1)}}

    @staticmethod
    def _signals(scene, cand, info) -> None:
        if not cand.track_ids:
            cand.quality = float(info.get("quality", 0.7))
            cand.recovered_frac, cand.kappa = 0.0, None
            return
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
