"""Frame-level anomaly score from object-level beliefs (what VAD benchmarks evaluate).

For each candidate c (an object / pair / region hypothesis):
    unverified   b_c = w_u * q_c                                 q_c = gate priority
    verified     b_c = sigmoid(logit(p_vlm) + eta * logit(q_c))  product of experts
    closed       b_c decays with exp(-(t - t_last) / decay_s)
Frame score:
    s(t) = 1 - (1 - max_c b_c(t)) * (1 - w_d * dense(t))         dense = per-object normality
then an attack/release filter (fast rise, slow fall).

Online scores are causal: a VLM answer affects frames only from t_alert = t_call + latency.
Offline scores (the usual VAD protocol) apply each candidate's final belief to its whole
interval [t_start - pre_s, t_last + post_s]. Both are saved; report which one you use.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class ScorerConfig:
    w_unverified: float = 0.4
    eta: float = 0.5
    w_dense: float = 0.25
    decay_s: float = 1.0
    hold_s: float = 0.5               # keep full belief this long after the last rule hit (one window)
    tau_up_s: float = 0.25
    tau_down_s: float = 0.75
    pre_s: float = 1.0
    post_s: float = 1.0
    probe_threshold: float = 0.5


@dataclass
class CandState:
    cand_id: int
    type_hint: str
    t_start: float
    t_last: float
    q: float = 0.0
    p_vlm: float | None = None
    category: str | None = None
    hypothesis: str | None = None
    track_ids: list = field(default_factory=list)
    box: tuple | None = None
    open: bool = True

    def belief(self, t: float, cfg: ScorerConfig) -> float:
        if self.p_vlm is None:
            b = cfg.w_unverified * self.q
        else:
            b = _sigmoid(_logit(self.p_vlm) + cfg.eta * _logit(max(self.q, 1e-3)))
        if t > self.t_last + cfg.hold_s:          # the rule stopped firing: let the belief fade
            b *= math.exp(-(t - self.t_last - cfg.hold_s) / cfg.decay_s)
        return b


def vlm_probability(verdict: str, confidence: float | None) -> float:
    """P(anomaly) from a parsed verdict. All backends report ``confidence`` = P(yes) (normalised over
    yes/no/uncertain at the verdict token) when they can; otherwise fall back to the hard verdict."""
    if confidence is not None:
        return float(min(max(confidence, 0.0), 1.0))
    return {"yes": 0.9, "no": 0.1}.get(verdict, 0.5)


class FrameScorer:
    def __init__(self, cfg: ScorerConfig | None = None):
        self.cfg = cfg or ScorerConfig()
        self.cands: dict[int, CandState] = {}
        self.pending: list = []                       # (t_avail, seq, cand_id, p, category)
        self._seq = 0
        self.dense = 0.0
        self.smooth = 0.0
        self.last_t = None
        self.times: list[float] = []
        self.raw: list[float] = []
        self.online: list[float] = []
        self.dense_trace: list[float] = []

    # ------------------------------------------------------------------ inputs
    def update_candidates(self, t: float, cands, hypothesis_of=None) -> None:
        live = set()
        for c in cands:
            st = self.cands.get(c.cand_id)
            if st is None:
                st = CandState(c.cand_id, c.type_hint, c.t_start, c.t_last,
                               hypothesis=hypothesis_of(c) if hypothesis_of else None)
                self.cands[c.cand_id] = st
            st.t_last, st.q, st.open = c.t_last, float(c.priority), True
            st.track_ids, st.box = list(c.track_ids), c.box
            live.add(c.cand_id)
        for cid, st in self.cands.items():
            if cid not in live:
                st.open = False

    def set_dense(self, value: float) -> None:
        self.dense = float(value)

    def add_verdict(self, t_avail: float, cand_id: int, p: float, category: str | None) -> None:
        heapq.heappush(self.pending, (t_avail, self._seq, cand_id, p, category))
        self._seq += 1

    # ------------------------------------------------------------------ per processed frame
    def frame(self, t: float) -> float:
        while self.pending and self.pending[0][0] <= t + 1e-9:
            _, _, cid, p, cat = heapq.heappop(self.pending)
            st = self.cands.get(cid)
            if st is not None:
                st.p_vlm, st.category = p, cat
        best = max((st.belief(t, self.cfg) for st in self.cands.values()), default=0.0)
        s = 1.0 - (1.0 - best) * (1.0 - self.cfg.w_dense * self.dense)
        dt = 0.0 if self.last_t is None else max(t - self.last_t, 0.0)
        tau = self.cfg.tau_up_s if s > self.smooth else self.cfg.tau_down_s
        a = 1.0 if self.last_t is None else 1.0 - math.exp(-dt / max(tau, 1e-6))
        self.smooth += a * (s - self.smooth)
        self.last_t = t
        self.times.append(t)
        self.raw.append(s)
        self.dense_trace.append(self.dense)
        self.online.append(self.smooth)
        return self.smooth

    def flush(self) -> None:
        """Apply verdicts that would arrive after the stream ended (offline scores only)."""
        while self.pending:
            _, _, cid, p, cat = heapq.heappop(self.pending)
            if cid in self.cands:
                self.cands[cid].p_vlm, self.cands[cid].category = p, cat

    def offline_scores(self) -> np.ndarray:
        self.flush()
        t = np.asarray(self.times)
        best = np.zeros_like(t)
        cfg = self.cfg
        for st in self.cands.values():
            b_final = st.belief(st.t_last, cfg)
            m = (t >= st.t_start - cfg.pre_s) & (t <= st.t_last + cfg.post_s)
            best[m] = np.maximum(best[m], b_final)
        return 1.0 - (1.0 - best) * (1.0 - cfg.w_dense * np.asarray(self.dense_trace))

    # ------------------------------------------------------------------ probes (A2Seek)
    def top_hypothesis(self, t: float):
        best, arg = 0.0, None
        for st in self.cands.values():
            b = st.belief(t, self.cfg)
            if b > best:
                best, arg = b, st
        return best, arg
