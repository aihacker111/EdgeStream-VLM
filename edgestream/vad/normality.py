"""Training-free, online normality models for object-centric anomaly scoring.

Two complementary models, both updated causally and both able to warm-start from
normal reference videos (``save`` / ``load``):

1. ``ClassLocationMap`` - where does each class usually appear?  Counts live in
   *ego-stabilised* coordinates (the UAV drift is integrated so a lawn stays the same
   cells while the drone moves) and the score is the pointwise surprise
       r = log p(g) - log p(g | cell)   (Dirichlet-smoothed, own counts removed)
   i.e. "how much rarer is a pedestrian HERE than anywhere". Catches trespassing on a
   lawn, jaywalking, a bicycle on a footpath, a vehicle on a plaza.

2. ``KinematicBank`` - per class group, a memory of normal motion descriptors
   (scale-free: object sizes/s, jitter, turning, aspect, tortuosity, deviation from
   the local flow). The score is the k-NN distance in robust-standardised space,
   divided by the bank's own typical k-NN distance. Only low-score samples are
   inserted (self-gated), so an ongoing anomaly does not become "normal".
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

HUMAN = frozenset({"pedestrian", "people", "person"})
TWO_WHEEL = frozenset({"bicycle", "motor", "motorcycle"})


def class_group(cls_name: str) -> str:
    if cls_name in HUMAN:
        return "human"
    if cls_name in TWO_WHEEL:
        return "two_wheel"
    return "vehicle"


GROUPS = ("human", "two_wheel", "vehicle")


class StabilizedFrame:
    """Integrates per-frame affines (prev->curr) into a current->reference transform."""

    def __init__(self, reset_frac: float = 1.5):
        self.M = np.eye(3)              # current -> reference
        self.reset_frac = reset_frac
        self.resets = 0
        self.image_wh = (1, 1)

    def update(self, A_prev2curr: np.ndarray, image_wh: tuple) -> bool:
        """Returns True when the reference was reset (camera moved to a new area)."""
        self.image_wh = image_wh
        A3 = np.vstack([A_prev2curr, [0, 0, 1]])
        try:
            self.M = self.M @ np.linalg.inv(A3)
        except np.linalg.LinAlgError:
            return False
        w, h = image_wh
        cx, cy = self.to_ref(w / 2, h / 2)
        if abs(cx - w / 2) > self.reset_frac * w or abs(cy - h / 2) > self.reset_frac * h:
            self.M = np.eye(3)
            self.resets += 1
            return True
        return False

    def to_ref(self, x: float, y: float) -> tuple[float, float]:
        p = self.M @ np.array([x, y, 1.0])
        return float(p[0]), float(p[1])


class ClassLocationMap:
    """Per-cell Poisson occupancy rates in ego-stabilised coordinates.

    n[c, g]  object-seconds of group g observed in cell c (each track sampled once per second)
    E[c]     exposure: seconds the cell was inside the field of view
    rate     lambda[c, g] = (n + alpha * lambda_bar[g]) / (E + alpha)     (Gamma prior at the global rate)
    surprise r = log lambda_bar[g] - log lambda[c, g]       (3x3 neighbourhood pooled, own counts removed)

    "A pedestrian where, in 2 minutes of watching, no pedestrian has been" gets r ≈ log(E / alpha).
    """

    def __init__(self, cell_frac: float = 1 / 16, alpha: float = 25.0, min_exposure: float = 20.0,
                 sample_every_s: float = 1.0):
        self.cell_frac = cell_frac
        self.alpha = alpha
        self.min_exposure = min_exposure
        self.sample_every_s = sample_every_s
        self.reset()

    def reset(self) -> None:
        self.counts: dict = defaultdict(lambda: np.zeros(len(GROUPS)))
        self.exposure: dict = defaultdict(float)
        self.total = np.zeros(len(GROUPS))
        self.total_exposure = 0.0
        self.own: dict = defaultdict(lambda: defaultdict(lambda: np.zeros(len(GROUPS))))
        self._last_sample: dict = {}
        self._last_exposure_t = -1e9

    def cell_size(self, image_w: float) -> float:
        return max(image_w * self.cell_frac, 1.0)

    def key(self, x_ref: float, y_ref: float, image_w: float) -> tuple[int, int]:
        c = self.cell_size(image_w)
        return int(np.floor(x_ref / c)), int(np.floor(y_ref / c))

    def observe_view(self, t: float, stab: "StabilizedFrame") -> None:
        """Add exposure to every cell inside the current field of view (once per sample period)."""
        if t - self._last_exposure_t < self.sample_every_s:
            return
        dt = self.sample_every_s if self._last_exposure_t > -1e8 else self.sample_every_s
        self._last_exposure_t = t
        w, h = stab.image_wh
        c = self.cell_size(w)
        corners = np.array([stab.to_ref(x, y) for x, y in ((0, 0), (w, 0), (0, h), (w, h))])
        (x0, y0), (x1, y1) = corners.min(0), corners.max(0)
        for i in range(int(np.floor(x0 / c)) + 1, int(np.floor(x1 / c))):     # fully visible cells only
            for j in range(int(np.floor(y0 / c)) + 1, int(np.floor(y1 / c))):
                self.exposure[(i, j)] += dt
                self.total_exposure += dt

    def observe(self, t: float, gid: int, group: str, key: tuple) -> None:
        if t - self._last_sample.get(gid, -1e9) < self.sample_every_s:
            return
        self._last_sample[gid] = t
        g = GROUPS.index(group)
        self.counts[key][g] += self.sample_every_s
        self.total[g] += self.sample_every_s
        self.own[gid][key][g] += self.sample_every_s

    def surprise(self, gid: int, group: str, key: tuple) -> tuple[float, float]:
        """(surprise in nats, pooled exposure in seconds). Pooled over the 3x3 neighbourhood."""
        g = GROUPS.index(group)
        own_map = self.own.get(gid, {})
        n, E = 0.0, 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                k = (key[0] + di, key[1] + dj)
                w = 1.0 if (di, dj) == (0, 0) else 0.5
                cnt = self.counts.get(k)
                if cnt is not None:
                    own = own_map.get(k)
                    n += w * (cnt[g] - (own[g] if own is not None else 0.0))
                E += w * self.exposure.get(k, 0.0)
        own_tot = sum(v[g] for v in own_map.values()) if own_map else 0.0
        if self.total_exposure <= 0:
            return 0.0, 0.0
        lam_bar = (self.total[g] - own_tot + 1.0) / (self.total_exposure + 1.0)
        lam = (max(n, 0.0) + self.alpha * lam_bar) / (E + self.alpha)
        return float(np.log(lam_bar) - np.log(lam)), float(E)

    neighbourhood_surprise = surprise

    def forget(self, gid: int) -> None:
        self.own.pop(gid, None)
        self._last_sample.pop(gid, None)


FEATURE_NAMES = ("log_speed", "jitter", "turn_rate", "log_aspect", "d_log_aspect",
                 "log_tortuosity", "flow_dev")
# measurement-noise floor of each feature's robust scale: a bank of near-identical walkers must not
# turn a 2 % speed difference into a 10-sigma outlier
FEATURE_FLOOR = np.array([0.15, 0.15, 0.4, 0.1, 0.08, 0.1, 0.15])


class KinematicBank:
    def __init__(self, k: int = 5, capacity: int = 4096, min_items: int = 64, insert_below: float = 1.5,
                 ref_quantile: float = 0.95, seed: int = 0):
        self.k = k
        self.capacity = capacity
        self.min_items = min_items
        self.insert_below = insert_below
        self.ref_quantile = ref_quantile
        self.rng = np.random.default_rng(seed)
        self.items: dict[str, list] = {g: [] for g in GROUPS}
        self.seen = {g: 0 for g in GROUPS}
        self._stats: dict = {}

    def __len__(self) -> int:
        return sum(len(v) for v in self.items.values())

    def _calibrate(self, g: str) -> None:
        X = np.asarray(self.items[g], np.float64)
        med = np.median(X, axis=0)
        mad = np.maximum(np.median(np.abs(X - med), axis=0) * 1.4826, FEATURE_FLOOR)
        Z = (X - med) / mad
        n = len(Z)
        idx = self.rng.choice(n, size=min(n, 512), replace=False)
        d = np.sqrt(((Z[idx, None, :] - Z[None, :, :]) ** 2).sum(-1))
        d.sort(axis=1)
        kk = min(self.k, n - 1)
        # leave-one-out k-NN distances of normal samples; their upper quantile is the unit, so a score
        # of 1 means "as far from the bank as the 95th-percentile normal sample" (conformal-style)
        ref = max(float(np.quantile(d[:, kk], self.ref_quantile)), 1.0)   # column 0 is the point itself
        self._stats[g] = (med, mad, Z, ref, len(self.items[g]))

    def score(self, g: str, x: np.ndarray) -> float | None:
        """k-NN distance / 95th-percentile normal k-NN distance (<1 for most normal samples). None while cold."""
        if len(self.items[g]) < self.min_items:
            return None
        st = self._stats.get(g)
        if st is None or st[4] != len(self.items[g]) and (len(self.items[g]) - st[4]) > 0.1 * st[4]:
            self._calibrate(g)
            st = self._stats[g]
        med, mad, Z, ref, _ = st
        z = (np.asarray(x, np.float64) - med) / mad
        d = np.sqrt(((Z - z) ** 2).sum(-1))
        kk = min(self.k, len(d)) - 1
        return float(np.partition(d, kk)[kk] / ref)

    def insert(self, g: str, x: np.ndarray, score: float | None) -> None:
        if score is not None and score > self.insert_below:
            return
        self.seen[g] += 1
        x = np.asarray(x, np.float32)
        if len(self.items[g]) < self.capacity:
            self.items[g].append(x)
        else:                                              # reservoir sampling
            j = int(self.rng.integers(0, self.seen[g]))
            if j < self.capacity:
                self.items[g][j] = x

    def save(self, path: str) -> None:
        np.savez_compressed(path, **{g: np.asarray(v, np.float32).reshape(-1, len(FEATURE_NAMES))
                                     for g, v in self.items.items()})

    def load(self, path: str) -> "KinematicBank":
        data = np.load(path)
        for g in GROUPS:
            if g in data:
                self.items[g] = [r for r in data[g]]
                self.seen[g] = len(self.items[g])
        self._stats.clear()
        return self
