"""Density-aware gate with a token-bucket VLM budget (a main contribution of the paper).

Per window:
    p_i   = sigmoid(w . x_i + b)                          (priority)
    tau_t = tau0 * (1 + alpha * rho_t / rho_ref) * (1 + beta * (1 - b_t / B))
    send candidates with p_i >= tau_t, best p/cost first, while the bucket allows.
Candidates that are not sent stay open and are reconsidered in later windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import Candidate

FEATURES = ("s_rule", "s_anom", "severity_prior", "log_duration", "quality",
            "recovered_frac", "kappa_norm")


def features(c: Candidate, kappa_max: float = 300.0) -> np.ndarray:
    kappa = 0.5 if c.kappa is None else float(np.clip(c.kappa / kappa_max, 0.0, 1.0))
    return np.array([c.s_rule, c.s_anom, c.severity_prior, np.log1p(c.duration),
                     c.quality, c.recovered_frac, kappa], dtype=np.float64)


@dataclass
class PriorityModel:
    """Logistic priority. Default weights are hand-set; ``fit`` learns them from labels."""
    # persistence (log_duration) dominates: a rule that fired once is rarely worth a VLM call
    weights: np.ndarray = field(default_factory=lambda: np.array(
        [1.5, 1.0, 1.5, 2.5, 1.0, -2.0, 0.3]))
    bias: float = -4.5

    def __call__(self, c: Candidate) -> float:
        return float(1.0 / (1.0 + np.exp(-(features(c) @ self.weights + self.bias))))

    def fit(self, cands: list[Candidate], labels: list[int], l2: float = 1e-2,
            lr: float = 0.1, steps: int = 2000) -> "PriorityModel":
        X = np.stack([features(c) for c in cands])
        y = np.asarray(labels, dtype=np.float64)
        w, b = self.weights.copy(), self.bias
        for _ in range(steps):
            p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
            g = p - y
            w -= lr * (X.T @ g / len(y) + l2 * w)
            b -= lr * g.mean()
        self.weights, self.bias = w, float(b)
        return self


def uncertainty(p: float) -> float:
    """1 at p=0.5, 0 at p in {0, 1}."""
    return 4.0 * p * (1.0 - p)


def value_of_information(c: Candidate) -> float:
    """Expected benefit of a VLM call for this candidate.

    p * severity is the expected cost of missing it; the uncertainty term is how much the
    answer can still change our belief (1 if never verified, else 4 q (1-q) of the last
    VLM probability q). Persistence enters through p (log_duration feature).
    """
    u = 1.0 if c.vlm_p is None else uncertainty(c.vlm_p)
    return c.priority * (0.5 + c.severity_prior) * u


class TokenBucket:
    """VLM-call budget: refills at ``rate_per_min``; ``reserve`` is kept for critical calls."""

    def __init__(self, rate_per_min: float, capacity: float, reserve_frac: float = 0.2):
        self.rate = rate_per_min / 60.0
        self.capacity = float(capacity)
        self.reserve = reserve_frac * self.capacity
        self.tokens = float(capacity)
        self.t = None

    def refill(self, t: float) -> None:
        if self.t is not None:
            self.tokens = min(self.capacity, self.tokens + (t - self.t) * self.rate)
        self.t = t

    def can_spend(self, cost: float, critical: bool) -> bool:
        floor = 0.0 if critical else self.reserve
        return self.tokens - cost >= floor - 1e-9

    def spend(self, cost: float) -> None:
        self.tokens -= cost

    @property
    def level(self) -> float:
        return self.tokens / self.capacity if self.capacity else 0.0


@dataclass
class GateConfig:
    calls_per_min: float = 6.0
    bucket_capacity: float = 3.0
    reserve_frac: float = 0.2
    tau0: float = 0.5
    alpha: float = 0.5           # density sensitivity
    beta: float = 0.5            # budget sensitivity
    critical_types: tuple = ("counter_flow",)
    critical_priority: float = 0.8
    resend_after_s: float = 8.0  # re-verify an open candidate only after this long ...
    resend_delta_p: float = 0.15  # ... or when its priority rose by this much
    max_sends: int = 2           # per candidate
    voi: bool = False            # value-of-information ordering / re-verification (VAD mode)
    voi_resend_uncertainty: float = 0.5   # re-verify only if the last answer was this uncertain


class DensityGate:
    def __init__(self, cfg: GateConfig, priority: PriorityModel | None = None):
        self.cfg = cfg
        self.priority = priority or PriorityModel()
        self.bucket = TokenBucket(cfg.calls_per_min, cfg.bucket_capacity, cfg.reserve_frac)
        self.last_tau = cfg.tau0

    def threshold(self, density: float, density_ref: float) -> float:
        c = self.cfg
        rho = density / max(density_ref, 1.0)
        return c.tau0 * (1 + c.alpha * rho) / (1 + c.alpha) * (1 + c.beta * (1 - self.bucket.level))

    def step(self, t: float, cands: list[Candidate], density: float, density_ref: float,
             cost_fn=lambda c: 1.0) -> tuple[list[Candidate], dict]:
        c = self.cfg
        self.bucket.refill(t)
        tau = self.threshold(density, density_ref)
        self.last_tau = tau
        for cand in cands:
            cand.priority = self.priority(cand)

        eligible = []
        for cand in cands:
            if cand.times_sent >= c.max_sends:
                continue
            if cand.last_sent_t is not None:
                fresh = (t - cand.last_sent_t) >= c.resend_after_s
                rose = cand.priority - cand.last_sent_priority >= c.resend_delta_p
                if c.voi and cand.vlm_p is not None:
                    # re-ask only if the last answer was uncertain, or if the evidence grew after a "no"
                    fresh = fresh and uncertainty(cand.vlm_p) >= c.voi_resend_uncertainty
                    rose = rose and cand.vlm_p < 0.5
                if not (fresh or rose):
                    continue
            if cand.priority >= tau:
                eligible.append(cand)

        if c.voi:
            eligible.sort(key=lambda x: value_of_information(x) / max(cost_fn(x), 1e-6), reverse=True)
        else:
            eligible.sort(key=lambda x: x.priority / max(cost_fn(x), 1e-6), reverse=True)
        selected = []
        for cand in eligible:
            critical = cand.type_hint in c.critical_types and cand.priority >= c.critical_priority
            if self.bucket.can_spend(1.0, critical):
                self.bucket.spend(1.0)
                cand.times_sent += 1
                cand.last_sent_t = t
                cand.last_sent_priority = cand.priority
                selected.append(cand)
        info = {"t": round(t, 2), "tau": round(tau, 3), "bucket": round(self.bucket.tokens, 3),
                "density": density, "n_candidates": len(cands), "n_eligible": len(eligible),
                "n_selected": len(selected)}
        return selected, info
