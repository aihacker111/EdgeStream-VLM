"""Online referring MOT by receding-horizon search ("search, don't filter").

``OnlineRMOT`` runs the full method on one (video, expression) stream:

    perception (AERO-MOT tracks, ego affine)
      -> incremental tracklet hypothesis graph (cut early)
      -> provisional prior (class family + measured kinematics [+ CLIP])
      -> deadline-aware scheduler -> async VLM pool (set-level refer prompts, pairwise link prompts)
      -> window solver (labels + links jointly, frozen committed boundary)   every reopt_s
      -> commit frames older than t - L (+ stitched-gap interpolation and zoom re-detection)

``lag_s = inf`` is the offline special case: nothing is committed before the end, the queue
keeps working after the stream ends until no request is worth it (or ``max_calls``), then one
final solve. The filter baselines (YesTrack-style keyframe filter, every-frame filter) share the
same perception, prompts, VLM pool and lag semantics so that comparisons are compute-matched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .expression import Query, parse
from .hota import box_iou_xywh
from .scheduler import AsyncVLM, Request, SchedulerConfig
from .solver import SolverConfig, WindowSolver
from .tracklets import GraphConfig, Segment, TrackletGraph, logit, sigmoid
from .verifier import FrameSource
from .zoom import ZoomConfig, ZoomRedetector

INF = float("inf")


@dataclass
class RMOTConfig:
    method: str = "search"          # search | filter | prior
    lag_s: float = 2.0              # commit lag L (inf = offline)
    window_s: float = 12.0          # solver window W (open pieces)
    ghost_s: float = 3.0            # fully committed pieces stay as variables this long after their commit
    reopt_s: float = 0.5            # re-optimisation period
    stride: int = 1
    # verifier usage
    group_size: int = 8
    rounds: int = 2                 # R: partitions of each anchor pool (test-time scaling)
    k_answers: int = 2              # K: answers wanted per segment (test-time scaling)
    anchor_s: float = 0.5
    min_ask_s: float = 0.3          # an open segment is asked once it is this long
    link_verify: bool = True
    link_band: tuple = (0.15, 0.85)  # stitch edges with prior p in this band are asked
    # prior
    prior_base: float = -1.0
    class_mismatch: float = -8.0
    kin_weight: float = 1.0
    unmeasured_cap: float = -0.5    # prior ceiling when the query has content the parser cannot measure
    clip_weight: float = 0.0
    # output
    interp_max_s: float = 2.0
    zoom: bool = True
    # filter baselines
    keyframe_every: int = 10        # frames between keyframes (1 = every-frame filter)
    filter_prompt: str = "single"   # single (YesTrack-style) | set
    filter_facts: bool = False      # YesTrack does not see measured kinematics
    facts: bool | None = None       # search: override the verifier's measured-facts text (None = keep)
    # Platt calibration of the VLM's P(yes) per prompt kind, fitted on the TRAIN split with
    # scripts/rmot_diagnose.py: calibrated logit = a * logit(p) + b
    calib: dict = field(default_factory=dict)       # {"set": [a, b], "single": [a, b]}
    filter_propagate: str = "forward"   # forward (online) | nearest (offline TRP)
    graph: GraphConfig = field(default_factory=GraphConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    sched: SchedulerConfig = field(default_factory=SchedulerConfig)
    zoom_cfg: ZoomConfig = field(default_factory=ZoomConfig)

    def cal_logit(self, p: float, kind: str) -> float:
        z = logit(p, 0.02)
        ab = (self.calib or {}).get(kind)
        return z if not ab else float(ab[0]) * z + float(ab[1])

    @classmethod
    def from_dict(cls, d: dict) -> "RMOTConfig":
        d = dict(d or {})
        sub = {"graph": GraphConfig, "solver": SolverConfig, "sched": SchedulerConfig, "zoom_cfg": ZoomConfig}
        kw = {k: v for k, v in d.items() if k not in sub and k in cls.__dataclass_fields__}
        if "lag_s" in kw and kw["lag_s"] in ("inf", None):
            kw["lag_s"] = INF
        if "link_band" in kw:
            kw["link_band"] = tuple(kw["link_band"])
        for k, c in sub.items():
            kw[k] = c(**{a: b for a, b in (d.get(k) or {}).items() if a in c.__dataclass_fields__})
        return cls(**kw)


# ---------------------------------------------------------------------- ground-truth oracle
class GTOracle:
    """Truth functions for :class:`OracleVerifier` (and for diagnostics). Not used by the method."""

    def __init__(self, video, expr, iou: float = 0.5):
        self.video, self.expr, self.iou = video, expr, iou
        self._gt: dict = {}

    def _gt_at(self, rank: int):
        if rank not in self._gt:
            self._gt[rank] = self.video.gt_boxes(rank)
        return self._gt[rank]

    def gt_id(self, rank: int, box) -> int | None:
        gt = self._gt_at(rank)
        if not len(gt):
            return None
        iou = box_iou_xywh(np.asarray(box)[None], gt[:, 1:5])[0]
        k = int(np.argmax(iou))
        return int(gt[k, 0]) if iou[k] >= self.iou else None

    def refer(self, s: Segment) -> float:
        hit = 0
        for o in s.obs:
            gid = self.gt_id(o.rank, o.box)
            if gid is not None and gid in self.video.referred_ids(self.expr, o.rank):
                hit += 1
        return hit / max(len(s.obs), 1)

    def same(self, a: Segment, b: Segment):
        ga, gb = self.gt_id(a.obs[-1].rank, a.obs[-1].box), self.gt_id(b.obs[0].rank, b.obs[0].box)
        if ga is None or gb is None:
            return None
        return ga == gb


# ---------------------------------------------------------------------- optional CLIP prior
class ClipPrior:
    """Provisional crop-text similarity until the VLM answers (optional; transformers CLIP/SigLIP)."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "auto"):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        self.device = torch.device(dev)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self._text = {}
        self.stats = [0.0, 0.0, 0]

    def score(self, crop_bgr: np.ndarray, text: str) -> float:
        import cv2
        from PIL import Image
        t = self.torch
        with t.inference_mode():
            if text not in self._text:
                ti = self.proc(text=[f"a drone photo of {text}"], return_tensors="pt", padding=True).to(self.device)
                e = self.model.get_text_features(**ti)
                self._text[text] = e / e.norm(dim=-1, keepdim=True)
            ii = self.proc(images=[Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))],
                           return_tensors="pt").to(self.device)
            e = self.model.get_image_features(**ii)
            e = e / e.norm(dim=-1, keepdim=True)
            s = float((e @ self._text[text].T)[0, 0])
        m, v, n = self.stats
        n += 1
        d = s - m
        m += d / n
        v += d * (s - m)
        self.stats = [m, v, n]
        sd = math.sqrt(v / max(n - 1, 1)) if n > 2 else 1.0
        return (s - m) / max(sd, 1e-3) if n > 2 else 0.0


# ---------------------------------------------------------------------- the method
class OnlineRMOT:
    def __init__(self, cfg: RMOTConfig, verifier, zoom_detect=None, clip: ClipPrior | None = None):
        self.cfg = cfg
        self.verifier = verifier
        self.zoom = ZoomRedetector(zoom_detect, cfg.zoom_cfg) if (zoom_detect and cfg.zoom) else None
        self.clip = clip

    # ................................................................ helpers
    def _prior(self, g: TrackletGraph, s: Segment, q: Query) -> float:
        c = self.cfg
        if not q.class_ok(s.cls_name):
            return c.class_mismatch
        z = c.prior_base + c.kin_weight * g.kinematic_prior(s, q)
        if not q.measurable:
            z = min(z, c.unmeasured_cap)
        if self.clip is not None and c.clip_weight and s.n >= 3 and self._frames is not None:
            from .verifier import crop
            f = self._frames.get(s.mid().rank)
            if f is not None:
                z += c.clip_weight * self.clip.score(crop(f, s.mid().box), q.text)
        return z

    def _uncommitted_frames(self, s: Segment, tau: float) -> float:
        L = self.cfg.lag_s
        if L == INF:
            n = s.n
        else:
            n = sum(1 for o in s.obs[s.committed_upto + 1:] if o.t + L > tau)
        if not s.closed:
            n += self.cfg.sched.future_s * self.fps / self.cfg.stride
        return n

    def _deadline(self, s: Segment) -> float:
        if self.cfg.lag_s == INF:
            return INF
        i = s.committed_upto + 1
        t = s.obs[i].t if i < s.n else s.t1 + 1.0 / self.fps
        return t + self.cfg.lag_s

    # ................................................................ requests
    def _anchors_of(self, s: Segment) -> list:
        a = self.cfg.anchor_s
        lo, hi = math.ceil(s.t0 / a), math.floor(s.t1 / a)
        grid = [k * a for k in range(lo, hi + 1)]
        if not grid:
            grid = [round(s.mid().t / a) * a]
        return grid

    def _anchor_rank(self, t: float) -> int:
        st = self.cfg.stride
        return int(round(t * self.fps / st)) * st

    def _make_requests(self, g: TrackletGraph, now: float, q: Query) -> list:
        g.clear_memo()
        try:
            return self._make_requests_inner(g, now, q)
        finally:
            g._memo = None

    def _make_requests_inner(self, g: TrackletGraph, now: float, q: Query) -> list:
        c = self.cfg
        tau = now + self.vlm.expected_latency("refer")
        tol = 0.5 * c.anchor_s + 1e-6
        # pools: (anchor_t, family) -> [segments with an observation near the anchor]
        pools: dict = {}
        horizon = now - c.window_s
        for sid in self.live:
            s = g.segs[sid]
            if s.t1 < horizon:
                continue
            for at in self._anchors_of(s):
                if at > now + 1e-9 or at < horizon:
                    continue
                o = min(s.obs, key=lambda x: abs(x.t - at))
                if abs(o.t - at) <= tol:
                    pools.setdefault((at, s.family), []).append((o.box[0] + o.box[2] / 2, sid))
        reqs = []
        for (at, fam), members in pools.items():
            members.sort()
            sids = [m[1] for m in members]
            G = max(1, c.group_size)
            for r in range(max(1, c.rounds)):
                off = (r * G) // max(1, c.rounds)
                order = sids[off:] + sids[:off] if off < len(sids) else sids
                for k in range(0, len(order), G):
                    grp = order[k:k + G]
                    key = ("refer", round(at, 3), fam, r, k // G)
                    if key in self.issued:
                        continue
                    v, dl = 0.0, INF
                    for sid in grp:
                        s = g.segs[sid]
                        if len(s.vlm_logits) + s.n_asked_pending >= c.k_answers:
                            continue
                        if not s.closed and s.duration(self.fps / c.stride) < c.min_ask_s:
                            continue
                        p = sigmoid(g.effective_logit(sid))
                        u = 4 * p * (1 - p)
                        a = self._uncommitted_frames(s, tau)
                        v += u * a / (1 + len(s.vlm_logits) + s.n_asked_pending)
                        dl = min(dl, self._deadline(s))
                    if v <= c.sched.min_value:
                        continue
                    urg = 1.0 if dl == INF else 1.0 / max(dl - tau, c.sched.eps_s)
                    reqs.append(Request("refer", key, grp, self._anchor_rank(at), v * urg))
        if c.link_verify:
            lo, hi = c.link_band
            for key, e in g.edges.items():
                if e.kind != "stitch" or e.fixed is not None or e.n_asked or ("link",) + key in self.issued:
                    continue
                if e.i not in self.live_all or e.j not in self.live_all:
                    continue
                p = sigmoid(g.edge_logit(e))
                if not (lo <= p <= hi):
                    continue
                sj = g.segs[e.j]
                rel = max(sigmoid(g.effective_logit(e.i)), sigmoid(g.effective_logit(e.j)))
                a = self._uncommitted_frames(sj, tau)
                v = 4 * p * (1 - p) * rel * a
                dl = self._deadline(sj)
                urg = 1.0 if dl == INF else 1.0 / max(dl - tau, c.sched.eps_s)
                if v * urg > c.sched.min_value:
                    reqs.append(Request("link", ("link",) + key, [key], -1, v * urg))
        reqs.sort(key=lambda r: -r.value)
        return reqs

    def _dispatch(self, g: TrackletGraph, now: float, expr: str) -> None:
        """Start one request per free worker; the calls of one tick run concurrently."""
        free = sum(1 for f in self.vlm.free_at if f <= now + 1e-9)
        batch = []
        while self.pool and free > 0 and (self.vlm.cfg.max_calls is None or
                                          self.vlm.calls + len(batch) < self.vlm.cfg.max_calls):
            req = self.pool.pop(0)
            if req.key in self.issued:
                continue
            self.issued.add(req.key)
            if req.kind == "refer":
                grp = [g.segs[sid] for sid in req.members]
                for s in grp:
                    s.n_asked_pending += 1
                fn = (lambda grp=grp, ar=req.anchor_rank: self.verifier.refer(g, grp, ar, expr, self._frames))
            else:
                i, j = req.members[0]
                g.edges[(i, j)].n_asked += 1
                fn = (lambda i=i, j=j: self.verifier.link(g, g.segs[i], g.segs[j], self._frames))
            batch.append((req, fn))
            free -= 1
        self.vlm.submit_many(now, batch, parallel=getattr(self.verifier, "parallel", False))

    def _apply(self, g: TrackletGraph, now: float) -> None:
        for t_avail, ans, req in self.vlm.collect(now):
            self.dirty = True
            self.touched.update(req.members if req.kind == "refer" else [k for m in req.members for k in m])
            if req.kind == "refer":
                for sid in req.members:
                    s = g.segs[sid]
                    s.n_asked_pending = max(0, s.n_asked_pending - 1)
                kind = "single" if len(req.members) == 1 else "set"
                for sid, p in ans.probs.items():
                    g.segs[sid].vlm_logits.append(self.cfg.cal_logit(p, kind))
            else:
                for key, p in ans.probs.items():
                    if key in g.edges:
                        g.edges[key].vlm_logits.append(logit(p, 0.02))

    # ................................................................ solve & commit
    def _solve(self, g: TrackletGraph, now: float, only: set | None = None) -> None:
        c = self.cfg
        horizon = now - c.window_s
        if c.lag_s == INF:
            free = list(self.compatible)
        else:
            ghost = now - c.lag_s - c.ghost_s
            free = [sid for sid in self.compatible
                    if (sid in self.live and g.segs[sid].t1 >= horizon) or g.segs[sid].t1 >= ghost]
        if not free:
            return
        y, x = self.solver.solve(g, free, only=only)
        self.touched = set()
        self.y_sol.update(y)
        self.x_sol.update(x)
        self.n_solves += 1
        self.solve_status[self.solver.last_status] = self.solve_status.get(self.solver.last_status, 0) + 1

    def _label(self, g: TrackletGraph, s: Segment) -> int:
        if s.sid in self.y_sol:
            return self.y_sol[s.sid]
        # not solved yet (appeared since the last solve): chain-aware evidence, so a new piece
        # of an already-verified object keeps its label
        return int(g.effective_logit(s.sid) > self.cfg.solver.tau)

    def _commit(self, g: TrackletGraph, tc: float) -> None:
        done = []
        for sid in sorted(self.live, key=lambda k: (g.segs[k].t0, k)):
            s = g.segs[sid]
            i0 = s.committed_upto + 1
            if i0 >= s.n or s.obs[i0].t > tc:
                if s.closed and i0 >= s.n:
                    done.append(sid)
                continue
            if s.out_id is None:
                self._assign_identity(g, s)
            y = self._label(g, s)
            i = i0
            while i < s.n and s.obs[i].t <= tc:
                o = s.obs[i]
                if y:
                    box = o.box
                    if o.recovered and self.zoom is not None:
                        f = self._frames.get(o.rank)
                        if f is not None:
                            box, _ = self.zoom.refine(f, box, s.family)
                    self.rows.append((o.rank + 1, s.out_id, *[float(v) for v in box]))
                i += 1
            s.committed_upto = i - 1
            s.y_committed = y
            if s.closed and s.committed_upto >= s.n - 1:
                done.append(sid)
        for sid in done:
            self.live.discard(sid)

    def _assign_identity(self, g: TrackletGraph, s: Segment) -> None:
        chosen = None
        for key in g.in_edges[s.sid]:
            e = g.edges[key]
            src = g.segs[e.i]
            if e.fixed is None and key not in self.x_sol:
                xv = int(g.edge_logit(e) > 0)            # not solved yet (new piece): link prior
            else:
                xv = self.x_sol.get(key, 0)
            ok = e.fixed == 1 or (e.fixed is None and xv == 1)
            if ok and src.out_id is not None and src.committed_upto >= src.n - 1 and e.i in self.compatible:
                # the source must not already continue elsewhere
                if any(g.edges[k].fixed == 1 for k in g.out_edges[e.i] if k != key):
                    continue
                if chosen is None or g.edge_logit(e) > g.edge_logit(g.edges[chosen]):
                    chosen = key
        for key in g.in_edges[s.sid]:
            g.edges[key].fixed = 1 if key == chosen else 0
        if chosen is None:
            s.out_id = self.next_id
            self.next_id += 1
            return
        e = g.edges[chosen]
        for k in g.out_edges[e.i]:
            if k != chosen:
                g.edges[k].fixed = 0
        src = g.segs[e.i]
        s.out_id = src.out_id
        # stitched gap: interpolate when both sides are referred
        y_now = self._label(g, s)
        if e.kind == "stitch" and src.y_committed and y_now and e.gap_s <= self.cfg.interp_max_s:
            a, b = src.obs[-1], s.obs[0]
            for r in range(a.rank + 1, b.rank):
                w = (r - a.rank) / (b.rank - a.rank)
                box = (1 - w) * a.box + w * b.box
                if self.zoom is not None:
                    f = self._frames.get(r)
                    if f is not None:
                        box, _ = self.zoom.refine(f, box, s.family)
                self.rows.append((r + 1, s.out_id, *[float(v) for v in box]))

    # ................................................................ main loop
    def run(self, perception: list, expr: str, fps: float, image_wh: tuple, frames: FrameSource | None = None,
            affines: list | None = None) -> dict:
        """``perception[rank]`` = FrameOutput (or None for skipped ranks); ``affines[rank]`` = 2x3."""
        c = self.cfg
        if c.method == "filter":
            return self._run_filter(perception, expr, fps, image_wh, frames, affines)
        old_facts = getattr(self.verifier, "facts", None)
        if c.facts is not None and old_facts is not None:
            self.verifier.facts = c.facts
        try:
            return self._run_search(perception, expr, fps, image_wh, frames, affines)
        finally:
            if c.facts is not None and old_facts is not None:
                self.verifier.facts = old_facts

    def _run_search(self, perception, expr, fps, image_wh, frames, affines) -> dict:
        c = self.cfg
        self.fps = fps
        self._frames = frames or FrameSource()
        q = parse(expr)
        gcfg = GraphConfig(**{**c.graph.__dict__, "fps": fps / c.stride})
        g = TrackletGraph(gcfg, q)
        self.vlm = AsyncVLM(c.sched)
        self.solver = WindowSolver(c.solver)
        self.y_sol, self.x_sol = {}, {}
        self.rows, self.pool, self.issued = [], [], set()
        self.live, self.live_all, self.compatible = set(), set(), set()
        self.next_id, self.n_solves, self.solve_status = 1, 0, {}
        self.dirty = False
        self.touched = set()
        use_vlm = c.method == "search" and self.verifier is not None
        N = len(perception)
        t_next = 0.0
        t = 0.0
        for rank in range(0, N, c.stride):
            fo = perception[rank]
            if fo is None:
                continue
            t = rank / fps
            A = affines[rank] if affines is not None else None
            new = g.update(rank, t, fo.tracks, image_wh, A)
            for sid in new:
                s = g.segs[sid]
                s.n_asked_pending = 0
                if q.class_ok(s.cls_name):
                    self.live.add(sid)
                    self.live_all.add(sid)
                    self.compatible.add(sid)
            if use_vlm:
                self._apply(g, t)
            if t >= t_next:
                for sid in self.live:
                    s = g.segs[sid]
                    if not s.closed or s.prior_logit == 0.0:
                        s.prior_logit = self._prior(g, s, q)
                if use_vlm:
                    self.pool = self._make_requests(g, t, q)
                if c.lag_s != INF:                   # offline: one exact solve at the end
                    self._solve(g, t)
                    self.dirty = False
                t_next = t + c.reopt_s
            elif self.dirty and c.lag_s != INF:      # new answers: re-solve the touched components
                self._solve(g, t, only=set(self.touched))
                self.dirty = False
            if use_vlm:
                self._dispatch(g, t, expr)
            if c.lag_s != INF:
                self._commit(g, t - c.lag_s)
        t_end = t
        # ---- after the stream: finite L commits the tail on schedule; L = inf drains the queue
        if c.lag_s != INF:
            tt = t_end
            while self.live and tt <= t_end + c.lag_s + c.reopt_s:
                tt += c.reopt_s
                if use_vlm:
                    self._apply(g, tt)
                    self.pool = self._make_requests(g, tt, q)
                self._solve(g, tt)
                if use_vlm:
                    self._dispatch(g, tt, expr)
                self._commit(g, tt - c.lag_s)
            self._commit(g, INF)
        else:
            tt = t_end
            if use_vlm and c.sched.offline_drain:
                for _ in range(100000):
                    self._apply(g, tt)
                    self.pool = self._make_requests(g, tt, q)
                    if not self.pool and not self.vlm.pending:
                        break
                    if not self.vlm.budget_left() and not self.vlm.pending:
                        break
                    self._dispatch(g, tt, expr)
                    nxt = min([self.vlm.pending[0].t_avail] if self.vlm.pending else [tt + c.reopt_s])
                    tt = max(tt + 1e-3, min(nxt, min(self.vlm.free_at)) if self.pool else nxt)
                self._apply(g, INF)
            for sid in self.live:
                g.segs[sid].prior_logit = self._prior(g, g.segs[sid], q)
            self.solver.cfg.time_limit_s = max(self.solver.cfg.time_limit_s, 5.0)
            self._solve(g, tt)
            self._commit(g, INF)
        self.vlm.close()
        rows = _finalize(self.rows, N, c.stride)
        return {"rows": rows, "stats": {
            "vlm": self.vlm.stats(), "segments": len(g.segs), "edges": len(g.edges),
            "stitch_edges": sum(1 for e in g.edges.values() if e.kind == "stitch"),
            "solves": self.n_solves, "solver": self.solve_status,
            "zoom_calls": self.zoom.calls if self.zoom else 0, "zoom_hits": self.zoom.hits if self.zoom else 0,
            "out_ids": self.next_id - 1}, "graph": g}

    # ................................................................ filter baselines
    def _run_filter(self, perception, expr, fps, image_wh, frames, affines) -> dict:
        """YesTrack-style keyframe filter (keyframe_every=k) / every-frame filter (k=1).

        At each keyframe every active class-compatible track is judged independently (or in
        set prompts with ``filter_prompt=set``); a track keeps the label of its last answered
        keyframe (``forward``) or of the nearest answered keyframe (``nearest``, offline TRP).
        No stitching, no joint inference. Same VLM pool, prompts and lag rule as the method.
        """
        c = self.cfg
        self.fps = fps
        self._frames = frames or FrameSource()
        q = parse(expr)
        gcfg = GraphConfig(**{**c.graph.__dict__, "fps": fps / c.stride, "max_len_s": 1e9})
        g = TrackletGraph(gcfg, q)
        self.vlm = AsyncVLM(c.sched)
        answers: dict = {}                       # gid -> [(kf_rank, t_avail, p)]
        queue: list = []                         # FIFO of (kf_rank, deadline, [segments])
        prior_of: dict = {}
        N = len(perception)
        t = 0.0
        L = c.lag_s
        k = max(1, c.keyframe_every)

        def apply(now):
            for t_avail, ans, req in self.vlm.collect(now):
                for sid, p in ans.probs.items():
                    answers.setdefault(g.segs[sid].gid, []).append((req.anchor_rank, t_avail, p))

        def dispatch(now):
            free = sum(1 for f in self.vlm.free_at if f <= now + 1e-9)
            single = c.filter_prompt == "single"
            batch = []
            while queue and free > 0 and (self.vlm.cfg.max_calls is None or
                                          self.vlm.calls + len(batch) < self.vlm.cfg.max_calls):
                kf, dl, grp = queue.pop(0)
                req = Request("single" if single else "refer", ("kf", kf, grp[0].sid), [s.sid for s in grp], kf, 1.0)
                batch.append((req, lambda grp=grp, kf=kf: self.verifier.refer(g, grp, kf, expr, self._frames,
                                                                               single=single)))
                free -= 1
            self.vlm.submit_many(now, batch, parallel=getattr(self.verifier, "parallel", False))

        def enqueue(kf, dl, grp):
            # a newer keyframe supersedes a still-pending older one of the same track(s): an
            # answer keeps labelling the track until a newer one arrives, so nothing is dropped
            # for being late, but stale work is never done
            gids = {s.gid for s in grp}
            for n_, q_ in enumerate(queue):          # replace in place: keep its turn in the queue
                if {s.gid for s in q_[2]} <= gids:
                    queue[n_] = (kf, dl, grp)
                    return
            queue.append((kf, dl, grp))

        old_facts = getattr(self.verifier, "facts", None)
        if old_facts is not None:
            self.verifier.facts = c.filter_facts
        try:
            for rank in range(0, N, c.stride):
                fo = perception[rank]
                if fo is None:
                    continue
                t = rank / fps
                g.update(rank, t, fo.tracks, image_wh, affines[rank] if affines is not None else None)
                apply(t)
                if (rank // c.stride) % k == 0 and self.verifier is not None and c.method == "filter":
                    act = [g.segs[g.open_of_gid[int(tr.global_id)]] for tr in fo.tracks
                           if q.class_ok(tr.cls_name)]
                    dl = INF if L == INF else t + k * c.stride / fps + L
                    if c.filter_prompt == "single":
                        for s in act:
                            enqueue(rank, dl, [s])
                    else:
                        act.sort(key=lambda s: s.obs[-1].box[0])
                        G = max(1, c.group_size)
                        for i in range(0, len(act), G):
                            enqueue(rank, dl, act[i:i + G])
                for tr in fo.tracks:
                    gid = int(tr.global_id)
                    if gid not in prior_of and q.class_ok(tr.cls_name):
                        s = g.segs[g.open_of_gid[gid]]
                        if s.n >= 3:
                            prior_of[gid] = self._prior(g, s, q)
                dispatch(t)
            tt = t
            while queue or self.vlm.pending:
                if L != INF and tt > t + L + 1.0:
                    break
                tt = max(tt + 1e-3, min(self.vlm.free_at)) if queue else (
                    self.vlm.pending[0].t_avail if self.vlm.pending else tt)
                apply(tt)
                dispatch(tt)
                if not self.vlm.budget_left() and not self.vlm.pending:
                    break
            apply(INF)
        finally:
            self.vlm.close()
            if old_facts is not None:
                self.verifier.facts = old_facts
        rows = []
        for sid, s in g.segs.items():
            if not q.class_ok(s.cls_name):
                continue
            ans = sorted(answers.get(s.gid, []))
            for o in s.obs:
                tc = o.t + L if L != INF else INF
                av = [(kf, p) for kf, ta, p in ans if ta <= tc]
                if c.filter_propagate == "forward":
                    av = [(kf, p) for kf, p in av if kf <= o.rank]
                    p = av[-1][1] if av else None
                else:
                    p = min(av, key=lambda a: abs(a[0] - o.rank))[1] if av else None
                if p is None:
                    ok = prior_of.get(s.gid, c.prior_base) > c.solver.tau
                else:
                    ok = c.cal_logit(p, "single" if c.filter_prompt == "single" else "set") > c.solver.tau
                if ok:
                    rows.append((o.rank + 1, s.gid, *[float(v) for v in o.box]))
        return {"rows": _finalize(rows, N, c.stride),
                "stats": {"vlm": self.vlm.stats(), "segments": len(g.segs), "out_ids": len(g.track_of)},
                "graph": g}


def _finalize(rows: list, n_frames: int, stride: int) -> list:
    """Sort, drop duplicates (frame, id), fill skipped frames when running at a stride."""
    best = {}
    for r in rows:
        best.setdefault((int(r[0]), int(r[1])), r)
    rows = sorted(best.values(), key=lambda r: (r[1], r[0]))
    if stride > 1 and rows:
        extra = []
        for a, b in zip(rows, rows[1:]):
            if a[1] == b[1] and 1 < b[0] - a[0] <= stride:
                for f in range(int(a[0]) + 1, int(b[0])):
                    w = (f - a[0]) / (b[0] - a[0])
                    extra.append((f, a[1], *[(1 - w) * x + w * y for x, y in zip(a[2:], b[2:])]))
        rows = sorted(rows + extra, key=lambda r: (r[1], r[0]))
    return sorted([r for r in rows if 1 <= r[0] <= n_frames], key=lambda r: (r[0], r[1]))
