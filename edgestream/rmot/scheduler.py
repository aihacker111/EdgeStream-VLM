"""Deadline-aware scheduling of VLM requests on the logical clock.

The VLM is a pool of ``n_workers`` servers (vLLM batches concurrent requests). A request started
at logical time ``t`` returns at ``t + latency``; its answer can only influence frames that are
still uncommitted then (commit time of frame f = t_f + L). Each scheduling tick the queue picks,
for every free worker, the request of highest

    value = sum_i  4 p_i (1 - p_i)  x  frames_i(uncommitted at completion)  /  (1 + n_asked_i)

where p_i is the chain-aware posterior of member i. Requests whose members would all be
committed before the answer arrives have value 0 and are dropped. Dividing by the time left
before the first member's deadline gives urgency, so near-deadline uncertainty is resolved first.

The backend call itself runs synchronously when the request starts (deterministic replay); its
measured latency (x ``latency_scale``) or the oracle's latency model sets the completion time.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field


@dataclass
class SchedulerConfig:
    n_workers: int = 2              # concurrent VLM requests (vLLM batch slots)
    latency_scale: float = 1.0      # measured latency -> logical latency
    fixed_latency_s: float | None = None
    eps_s: float = 0.25             # urgency floor
    future_s: float = 1.0           # frames an open segment is still expected to add
    max_calls: int | None = None    # hard budget per stream (compute matching)
    offline_drain: bool = True      # L = inf: keep asking after the stream ends until nothing is worth it
    min_value: float = 1e-3


@dataclass(order=True)
class _Pending:
    t_avail: float
    seq: int
    answer: object = field(compare=False)
    request: object = field(compare=False)


@dataclass
class Request:
    kind: str                       # refer | link | single
    key: tuple                      # identity for de-duplication
    members: list                   # segment ids (refer) or [(i, j)] (link)
    anchor_rank: int = -1
    value: float = 0.0
    payload: dict = field(default_factory=dict)


class AsyncVLM:
    def __init__(self, cfg: SchedulerConfig):
        self.cfg = cfg
        self.free_at = [0.0] * cfg.n_workers
        self.pending: list = []
        self.calls = 0
        self.busy_s = 0.0
        self.lat_ema = {"refer": 0.6, "link": 0.4, "single": 0.5}
        self.log: list = []
        self._seq = itertools.count()

    def free_worker(self, now: float) -> int | None:
        i = min(range(len(self.free_at)), key=lambda k: self.free_at[k])
        return i if self.free_at[i] <= now + 1e-9 else None

    def budget_left(self) -> bool:
        return self.cfg.max_calls is None or self.calls < self.cfg.max_calls

    def expected_latency(self, kind: str) -> float:
        if self.cfg.fixed_latency_s is not None:
            return self.cfg.fixed_latency_s
        return self.lat_ema.get(kind, 0.6)

    def submit(self, now: float, req: Request, fn) -> float:
        return self.submit_many(now, [(req, fn)], parallel=False)[0]

    def submit_many(self, now: float, items: list, parallel: bool = True) -> list:
        """Start several requests at logical time ``now``. With ``parallel`` the backend calls really
        run concurrently (threads), so a vLLM server batches them; the logical schedule is the
        same as submitting them one by one."""
        if not items:
            return []
        if parallel and len(items) > 1:
            if getattr(self, "_pool", None) is None:
                from concurrent.futures import ThreadPoolExecutor
                self._pool = ThreadPoolExecutor(max_workers=max(self.cfg.n_workers, 1))
            answers = list(self._pool.map(lambda it: it[1](), items))
        else:
            answers = [fn() for _, fn in items]
        return [self._account(now, req, ans) for (req, _), ans in zip(items, answers)]

    def _account(self, now: float, req: Request, ans) -> float:
        w = self.free_worker(now)
        start = now if w is not None else min(self.free_at)
        w = w if w is not None else self.free_at.index(start)
        lat = self.cfg.fixed_latency_s if self.cfg.fixed_latency_s is not None else ans.latency_s * self.cfg.latency_scale
        lat = max(lat, 1e-3)
        self.lat_ema[req.kind] = 0.8 * self.lat_ema.get(req.kind, lat) + 0.2 * lat
        t_avail = start + lat
        self.free_at[w] = t_avail
        self.calls += 1
        self.busy_s += lat
        heapq.heappush(self.pending, _Pending(t_avail, next(self._seq), ans, req))
        self.log.append({"t_start": round(start, 3), "t_avail": round(t_avail, 3), "kind": req.kind,
                         "n": ans.n_cands, "value": round(req.value, 4), "prompt_tokens": ans.prompt_tokens,
                         "visual_tokens": ans.visual_tokens, "latency_s": round(lat, 3)})
        return t_avail

    def collect(self, now: float) -> list:
        out = []
        while self.pending and self.pending[0].t_avail <= now + 1e-9:
            p = heapq.heappop(self.pending)
            out.append((p.t_avail, p.answer, p.request))
        return out

    def close(self) -> None:
        if getattr(self, "_pool", None) is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def drain_time(self) -> float:
        return max([p.t_avail for p in self.pending] + [max(self.free_at)])

    def stats(self) -> dict:
        n = max(self.calls, 1)
        return {"calls": self.calls, "busy_s": round(self.busy_s, 2),
                "prompt_tokens": sum(e["prompt_tokens"] for e in self.log),
                "visual_tokens": sum(e["visual_tokens"] for e in self.log),
                "mean_latency_s": round(self.busy_s / n, 3),
                "by_kind": {k: sum(1 for e in self.log if e["kind"] == k) for k in ("refer", "link", "single")}}
