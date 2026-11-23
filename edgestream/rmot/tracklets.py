"""Incremental hypothesis graph over over-segmented tracklets ("cut early, decide late").

Nodes are short pieces of tracker tracks (``Segment``). A track is cut

* every ``max_len_s`` seconds, so a referring label can change over time (temporal validity:
  "the car that is turning" is only referred while it turns);
* where the tracker itself is unsure: a run of QAM-recovered boxes, a jump the Kalman velocity
  does not explain, an appearance drop or a class flip. These cuts get a lower link prior, so
  the solver may break the track there if the VLM evidence disagrees.

Edges are link hypotheses between consecutive pieces of the same track (``cont``) and between the
end of a track and the start of another one (``stitch``: the tracker lost the object and
started a new id), gated by an ego-motion-compensated motion prediction and, when available,
ReID cosine similarity.

All motion is measured in ego-stabilised coordinates (camera motion removed with the background
affine of :mod:`edgestream.scene.ego_motion`), which is what makes "moving", "parked", "turning
left" and "same direction as us" measurable from a moving drone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .expression import FAMILY_OF, Query


@dataclass
class GraphConfig:
    fps: float = 30.0
    max_len_s: float = 1.0          # segment length cap
    min_len_s: float = 0.2          # a cut never makes a piece shorter than this
    end_after_s: float = 0.5        # track considered ended when unseen for this long
    recovered_cut: int = 6          # consecutive QAM-recovered frames that trigger a cut
    jump_sigma: float = 3.0         # residual / size beyond which a jump cut is made
    app_cut: float = 0.45           # cosine to the segment mean below which an appearance cut is made
    p_cont: float = 0.97            # link prior of a length cut
    p_cont_weak: float = 0.80       # recovered / class flip cut
    p_cont_suspect: float = 0.50    # jump or appearance cut (possible identity switch)
    stitch_gap_s: float = 3.0       # max gap of a stitch hypothesis
    stitch_gate: float = 2.5        # gate radius in units of (object size + motion uncertainty)
    stitch_max_in: int = 4          # keep the best k stitch hypotheses per new track
    kin_window_s: float = 2.0       # track history used for kinematic features
    moving_bl_s: float = 0.35       # body lengths per second above which an object is "moving"
    ego_min_px_s: float = 8.0       # drone apparent speed below which "relative to us" is undefined
    # how a VLM answer reaches the pieces nobody asked about:
    #   identity: along links, weighted by P(same object) of every link on the way; a cut made only
    #             because a piece reached max_len_s is artificial (p_prop_length), a suspicious cut
    #             (jump, appearance, class, recovered, resume, stitch) passes its own link belief.
    #             The answer of one piece then covers the whole stretch of track between two
    #             suspicious cuts, as a filter baseline does, while an identity switch still stops it.
    #   hops:     legacy, the answer decays 0.5 per piece over at most 3 pieces (ablation).
    propagation: str = "identity"
    p_prop_length: float = 0.995
    prop_max_hops: int = 90
    prop_min_w: float = 0.05


@dataclass
class Obs:
    rank: int
    t: float
    box: np.ndarray                 # tlwh, image px
    g: np.ndarray                   # centre in ego-stabilised reference coordinates
    epoch: int                      # stabiliser epoch (reference resets break motion continuity)
    score: float
    quality: float
    recovered: bool
    emb: np.ndarray | None = None


@dataclass
class Segment:
    sid: int
    gid: int                        # tracker global id
    cls_name: str
    family: str
    obs: list = field(default_factory=list)
    closed: bool = False
    track_ended: bool = False       # closed because the track disappeared (stitch source)
    first_of_track: bool = False    # first piece of a track (stitch target)
    cut_reason: str = "start"
    # evidence
    prior_logit: float = 0.0
    vlm_logits: list = field(default_factory=list)
    n_asked: int = 0
    n_asked_pending: int = 0
    # commit state
    committed_upto: int = -1        # index into obs of the last committed observation
    y_committed: int | None = None
    out_id: int | None = None
    in_edge_fixed: tuple | None = None   # (i, j) fixed once this node has committed frames
    _emb_sum: np.ndarray | None = None
    _v: np.ndarray | None = None          # EMA velocity (ref px/s), for the jump test

    @property
    def t0(self) -> float:
        return self.obs[0].t

    @property
    def t1(self) -> float:
        return self.obs[-1].t

    @property
    def n(self) -> int:
        return len(self.obs)

    def duration(self, fps: float) -> float:
        return len(self.obs) / fps

    def emb_mean(self) -> np.ndarray | None:
        if self._emb_sum is None:
            return None
        v = self._emb_sum
        nv = np.linalg.norm(v)
        return v / nv if nv > 0 else None

    def obs_near(self, t: float) -> Obs:
        return min(self.obs, key=lambda o: abs(o.t - t))

    def mid(self) -> Obs:
        return self.obs[len(self.obs) // 2]


@dataclass
class Edge:
    i: int
    j: int
    kind: str                       # cont | stitch
    prior_logit: float
    vlm_logits: list = field(default_factory=list)
    n_asked: int = 0
    gap_s: float = 0.0
    fixed: int | None = None        # 1/0 once decided by a commit


def logit(p: float, eps: float = 1e-3) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 30.0), -30.0)))


class TrackletGraph:
    def __init__(self, cfg: GraphConfig | None = None, query: Query | None = None):
        self.cfg = cfg or GraphConfig()
        self.query = query
        self.segs: dict[int, Segment] = {}
        self.edges: dict[tuple, Edge] = {}
        self.out_edges: dict[int, list] = {}
        self.in_edges: dict[int, list] = {}
        self.open_of_gid: dict[int, int] = {}      # gid -> sid of its open segment
        self.last_seen: dict[int, float] = {}
        self.track_of: dict[int, list] = {}        # gid -> [sid] in order
        self._vcache: dict = {}
        self._next = 0
        # ego motion
        self.M = np.eye(3)                         # current -> reference
        self.epoch = 0
        self.ego_vel = np.zeros(2)                 # drone apparent velocity in image px/s (EMA)
        self.image_wh = (1, 1)
        self.t = 0.0

    # ------------------------------------------------------------------ ego motion
    def _update_ego(self, A: np.ndarray | None, image_wh: tuple, dt: float) -> None:
        self.image_wh = image_wh
        if A is None:
            return
        A3 = np.vstack([np.asarray(A, np.float64).reshape(2, 3), [0, 0, 1]])
        try:
            self.M = self.M @ np.linalg.inv(A3)
        except np.linalg.LinAlgError:
            return
        w, h = image_wh
        c = self.M @ np.array([w / 2, h / 2, 1.0])
        if abs(c[0] - w / 2) > 1.5 * w or abs(c[1] - h / 2) > 1.5 * h:
            self.M = np.eye(3)
            self.epoch += 1
        # apparent velocity of the background at the image centre = -(drone motion in image)
        p = np.array([w / 2, h / 2, 1.0])
        bg = (A3 @ p)[:2] - p[:2]
        v = -bg / max(dt, 1e-6)
        self.ego_vel = 0.9 * self.ego_vel + 0.1 * v

    def to_ref(self, x: float, y: float) -> np.ndarray:
        p = self.M @ np.array([x, y, 1.0])
        return p[:2].copy()

    # ------------------------------------------------------------------ building
    def _new_seg(self, gid: int, cls_name: str, reason: str, first: bool) -> Segment:
        s = Segment(self._next, gid, cls_name, FAMILY_OF.get(cls_name, cls_name), cut_reason=reason,
                    first_of_track=first)
        self._next += 1
        self.segs[s.sid] = s
        self.out_edges[s.sid] = []
        self.in_edges[s.sid] = []
        self.track_of.setdefault(gid, []).append(s.sid)
        self.open_of_gid[gid] = s.sid
        return s

    def add_edge(self, i: int, j: int, kind: str, p: float, gap_s: float = 0.0) -> Edge:
        e = Edge(i, j, kind, logit(p), gap_s=gap_s)
        self.edges[(i, j)] = e
        self.out_edges[i].append((i, j))
        self.in_edges[j].append((i, j))
        return e

    def _cut(self, s: Segment, reason: str, p: float) -> Segment:
        s.closed = True
        ns = self._new_seg(s.gid, s.cls_name, reason, first=False)
        ns.prior_logit = s.prior_logit
        ns._v = s._v
        self.add_edge(s.sid, ns.sid, "cont", p)
        return ns

    def update(self, rank: int, t: float, tracks: list, image_wh: tuple, A: np.ndarray | None = None) -> list:
        """Add one frame. ``tracks`` are perception ``TrackOutput``s. Returns new segment ids."""
        cfg = self.cfg
        dt = t - self.t if rank > 0 else 1.0 / cfg.fps
        self.t = t
        self._update_ego(A, image_wh, dt)
        new, starts = [], []
        seen = set()
        for tr in tracks:
            gid = int(tr.global_id)
            seen.add(gid)
            box = np.asarray(tr.tlwh, np.float64)
            c = box[:2] + box[2:] / 2
            emb = None if getattr(tr, "embedding", None) is None else np.asarray(tr.embedding, np.float64)
            o = Obs(rank, t, box, self.to_ref(*c), self.epoch, float(tr.det_score), float(tr.quality),
                    bool(tr.is_recovered), emb)
            sid = self.open_of_gid.get(gid)
            s = self.segs.get(sid) if sid is not None else None
            if s is not None and s.closed:
                s = None
            if s is None:
                prev = self.track_of.get(gid)
                if prev:                       # the tracker re-found its own id after we closed it
                    last = prev[-1]            # read BEFORE _new_seg appends the new piece to this list
                    s = self._new_seg(gid, tr.cls_name, "resume", first=False)
                    self.add_edge(last, s.sid, "cont", cfg.p_cont_weak, gap_s=t - self.segs[last].t1)
                else:
                    s = self._new_seg(gid, tr.cls_name, "start", first=True)
                    starts.append((s, o))
                s.prior_logit = 0.0
                new.append(s.sid)
            else:
                reason, p = self._cut_reason(s, o, tr.cls_name)
                if reason:
                    s = self._cut(s, reason, p)
                    new.append(s.sid)
            if s.obs and s.obs[-1].epoch == o.epoch and o.t > s.obs[-1].t:
                vi = (o.g - s.obs[-1].g) / (o.t - s.obs[-1].t)
                s._v = vi if s._v is None else 0.7 * s._v + 0.3 * vi
            elif s.obs and s.obs[-1].epoch != o.epoch:
                s._v = None                      # stabiliser reset: velocity no longer comparable
            s.obs.append(o)
            if emb is not None and emb.size > 1:
                s._emb_sum = emb.copy() if s._emb_sum is None else s._emb_sum + emb
            self.last_seen[gid] = t
        for s, o in starts:                    # after all tracks of this frame are placed
            self._stitch_candidates(s, o, seen)
        # close tracks that disappeared
        for gid, sid in list(self.open_of_gid.items()):
            s = self.segs[sid]
            if s.closed or gid in seen:
                continue
            if t - self.last_seen.get(gid, t) >= cfg.end_after_s:
                s.closed = True
                s.track_ended = True
        return new

    def _cut_reason(self, s: Segment, o: Obs, cls_name: str) -> tuple:
        cfg = self.cfg
        dur = o.t - s.t0
        if dur < cfg.min_len_s:
            return None, None
        if FAMILY_OF.get(cls_name, cls_name) != s.family:
            return "class", cfg.p_cont_weak
        k = cfg.recovered_cut
        if not o.recovered and len(s.obs) >= k and all(x.recovered for x in s.obs[-k:]):
            return "recovered", cfg.p_cont_weak      # re-detected after a long QAM run
        last = s.obs[-1]
        if last.epoch == o.epoch and len(s.obs) >= 3 and s._v is not None:
            pred = last.g + s._v * (o.t - last.t)
            size = math.sqrt(max(o.box[2] * o.box[3], 1.0))
            if np.linalg.norm(o.g - pred) > cfg.jump_sigma * size:
                return "jump", cfg.p_cont_suspect
        if o.emb is not None and o.emb.size > 1:
            m = s.emb_mean()
            if m is not None and len(s.obs) >= 5 and float(m @ o.emb) / max(np.linalg.norm(o.emb), 1e-9) < cfg.app_cut:
                return "appearance", cfg.p_cont_suspect
        if dur >= cfg.max_len_s:
            return "length", cfg.p_cont
        return None, None

    @staticmethod
    def _velocity(obs: list) -> np.ndarray:
        """Least-squares velocity (ref px / s) of the observations of one epoch."""
        if not obs:
            return np.zeros(2)
        ep = obs[-1].epoch
        ts, xs, ys = [], [], []
        for o in obs:
            if o.epoch == ep:
                ts.append(o.t)
                xs.append(o.g[0])
                ys.append(o.g[1])
        n = len(ts)
        if n < 2:
            return np.zeros(2)
        tm, xm, ym = sum(ts) / n, sum(xs) / n, sum(ys) / n
        den = sum((t - tm) ** 2 for t in ts)
        if den <= 0:
            return np.zeros(2)
        return np.array([sum((t - tm) * (x - xm) for t, x in zip(ts, xs)) / den,
                         sum((t - tm) * (y - ym) for t, y in zip(ts, ys)) / den])

    def _end_velocity(self, a: Segment) -> np.ndarray:
        key = (a.sid, a.n)
        if self._vcache.get(a.sid, (None,))[0] != key:
            self._vcache[a.sid] = (key, self._velocity(self.history(a, self.cfg.kin_window_s)))
        return self._vcache[a.sid][1]

    def _stitch_candidates(self, s: Segment, o: Obs, alive: set) -> None:
        cfg = self.cfg
        cands = []
        for gid, t_last in self.last_seen.items():
            if gid == s.gid or gid in alive or o.t - t_last > cfg.stitch_gap_s:
                continue
            sids = self.track_of.get(gid)
            if not sids:
                continue
            a = self.segs[sids[-1]]
            if a.family != s.family or not a.obs:
                continue
            gap = o.t - a.t1
            if gap <= 0 or gap > cfg.stitch_gap_s:
                continue
            size0 = math.sqrt(max(a.obs[-1].box[2] * a.obs[-1].box[3], 1.0))
            if a.obs[-1].epoch == o.epoch and np.abs(o.g - a.obs[-1].g).max() > cfg.stitch_gate * (
                    size0 + 2.0) + 400.0 * gap:           # far beyond any plausible motion: skip cheaply
                continue
            v = self._end_velocity(a) if a.obs[-1].epoch == o.epoch else np.zeros(2)
            if a.obs[-1].epoch == o.epoch:
                pred = a.obs[-1].g + v * gap
                d = float(np.linalg.norm(o.g - pred))
            else:                                   # stabiliser reset in between: image distance
                pa = a.obs[-1].box[:2] + a.obs[-1].box[2:] / 2
                d = float(np.linalg.norm(o.box[:2] + o.box[2:] / 2 - pa))
            size = math.sqrt(max(a.obs[-1].box[2] * a.obs[-1].box[3], 1.0))
            scale = size + 0.5 * float(np.linalg.norm(v)) * gap + 2.0
            dn = d / scale
            if dn > cfg.stitch_gate:
                continue
            z = 2.0 - 1.6 * dn - 0.4 * gap
            ma, mb = a.emb_mean(), (o.emb / max(np.linalg.norm(o.emb), 1e-9)) if (
                o.emb is not None and o.emb.size > 1) else None
            if ma is not None and mb is not None:
                z += 4.0 * (float(ma @ mb) - 0.6)
            sr = min(a.obs[-1].box[2] * a.obs[-1].box[3], o.box[2] * o.box[3]) / max(
                a.obs[-1].box[2] * a.obs[-1].box[3], o.box[2] * o.box[3], 1.0)
            z -= 1.5 * (1 - sr)
            cands.append((z, a.sid, gap))
        cands.sort(reverse=True)
        for z, sid, gap in cands[:cfg.stitch_max_in]:
            self.add_edge(sid, s.sid, "stitch", sigmoid(z), gap_s=gap)

    # ------------------------------------------------------------------ kinematics
    def history(self, s: Segment, seconds: float) -> list:
        """Observations of the same track in the last ``seconds`` before the end of ``s``."""
        out = []
        for sid in self.track_of[s.gid]:
            seg = self.segs[sid]
            if seg.t0 > s.t1:
                break
            out.extend(o for o in seg.obs if s.t1 - seconds <= o.t <= s.t1)
        return out

    def kinematics(self, s: Segment) -> dict:
        cfg = self.cfg
        hist = self.history(s, cfg.kin_window_s)
        last = s.obs[-1]
        size = math.sqrt(max(last.box[2] * last.box[3], 1.0))
        v = self._velocity(hist)
        speed = float(np.linalg.norm(v))
        bl = speed / size
        moving = bl > cfg.moving_bl_s
        k = {"speed_bl_s": bl, "moving": moving, "v": v, "size": size}
        k["heading"] = math.degrees(math.atan2(-v[1], v[0])) if moving else None
        # turning: heading change between the first and second half of the history
        turn = None
        if moving and len(hist) >= 6:
            h1, h2 = hist[:len(hist) // 2], hist[len(hist) // 2:]
            v1, v2 = self._velocity(h1), self._velocity(h2)
            if np.linalg.norm(v1) / size > cfg.moving_bl_s and np.linalg.norm(v2) / size > cfg.moving_bl_s:
                a1, a2 = math.atan2(-v1[1], v1[0]), math.atan2(-v2[1], v2[0])
                turn = math.degrees((a2 - a1 + math.pi) % (2 * math.pi) - math.pi)   # + = left (CCW)
        k["turn_deg"] = turn
        ev = self.ego_vel
        es = float(np.linalg.norm(ev))
        k["ego_moving"] = es > cfg.ego_min_px_s
        k["rel_ego_cos"] = float(v @ ev / (speed * es)) if (moving and es > cfg.ego_min_px_s) else None
        W, H = self.image_wh
        cx, cy = last.box[0] + last.box[2] / 2, last.box[1] + last.box[3] / 2
        k["pos"] = (cx / W, cy / H)
        return k

    def kinematic_prior(self, s: Segment, q: Query | None) -> float:
        """Log-odds contribution of the measurable predicates of the query."""
        if q is None or not q.has_kinematics() or q.negated:
            return 0.0
        k = self.kinematics(s)
        z = 0.0

        def vote(ok: bool | None, w: float = 1.0):
            nonlocal z
            if ok is None:
                return
            z += w if ok else -1.5 * w
        if q.moving is not None:
            vote(k["moving"] == q.moving, 1.2)
        if q.turning is not None:
            t = k["turn_deg"]
            if t is None:
                vote(False if k["moving"] is False else None, 0.8)
            elif q.turning == "any":
                vote(abs(t) > 20, 0.8)
            else:
                vote((t > 20) if q.turning == "left" else (t < -20), 0.8)
        if q.rel_ego is not None and k["rel_ego_cos"] is not None:
            vote(k["rel_ego_cos"] > 0.5 if q.rel_ego == "same" else k["rel_ego_cos"] < -0.5, 0.8)
        if q.lateral is not None and k["moving"]:
            vx = k["v"][0]
            vote(vx > 0 if q.lateral == "left_to_right" else vx < 0, 0.8)
        if q.toward is not None and k["moving"]:
            vy = k["v"][1]                        # oblique view: down in the image = toward the camera
            vote(vy > 0 if q.toward == "toward" else vy < 0, 0.5)
        return z

    # ------------------------------------------------------------------ evidence
    def seg_logit(self, s: Segment, k0: float = 0.5, temp: float = 1.0) -> float:
        if not s.vlm_logits:
            return s.prior_logit
        return (k0 * s.prior_logit + sum(v / temp for v in s.vlm_logits)) / (k0 + len(s.vlm_logits))

    def edge_logit(self, e: Edge, k0: float = 1.0) -> float:
        if not e.vlm_logits:
            return e.prior_logit
        return (k0 * e.prior_logit + sum(e.vlm_logits)) / (k0 + len(e.vlm_logits))

    def chain_neighbors(self, sid: int, depth: int = 3) -> list:
        """(sid, distance) of segments reachable through likely links (p > 0.5), both directions."""
        out, frontier, seen = [], [sid], {sid}
        for d in range(1, depth + 1):
            nxt = []
            for u in frontier:
                for key in self.out_edges.get(u, []) + self.in_edges.get(u, []):
                    e = self.edges[key]
                    if self.edge_logit(e) <= 0:
                        continue
                    w = e.j if e.i == u else e.i
                    if w not in seen:
                        seen.add(w)
                        nxt.append(w)
                        out.append((w, d))
            frontier = nxt
        return out

    def clear_memo(self) -> None:
        self._memo = {}

    def effective_logit(self, sid: int, gamma: float = 0.5) -> float:
        """Belief that piece ``sid`` is referred, using the answers about the same object."""
        return self._cached_belief(sid, gamma)[0]

    def support(self, sid: int) -> float:
        """How much of the belief of an unasked piece comes from answers about the same object
        (0 = prior only, 1 = as good as asked). 1.0 for an answered piece."""
        return self._cached_belief(sid)[1]

    def _cached_belief(self, sid: int, gamma: float = 0.5) -> tuple:
        memo = getattr(self, "_memo", None)
        if memo is not None and sid in memo:
            return memo[sid]
        if self.cfg.propagation == "hops":
            v = (self._effective_logit(sid, gamma), 1.0 if self.segs[sid].vlm_logits else 0.0)
        else:
            v = self._identity_belief(sid)
        if memo is not None:
            memo[sid] = v
        return v

    def prop_weight(self, e: Edge) -> float:
        """P(the two pieces of ``e`` are the same object), as used for propagating answers."""
        if e.fixed == 0:
            return 0.0
        if e.kind == "cont" and not e.vlm_logits and self.segs[e.j].cut_reason == "length":
            return self.cfg.p_prop_length
        return sigmoid(self.edge_logit(e))

    def _identity_belief(self, sid: int) -> tuple:
        s = self.segs[sid]
        if s.vlm_logits:
            return max(min(self.seg_logit(s), 12.0), -12.0), 1.0
        # best-path P(same object) to the nearest answered pieces, both directions; the search
        # stops at an answered piece (it already summarises what lies beyond it)
        best = {sid: 1.0}
        frontier = [sid]
        found = {}
        for _ in range(self.cfg.prop_max_hops):
            nxt = []
            for u in frontier:
                wu = best[u]
                for key in self.out_edges.get(u, []) + self.in_edges.get(u, []):
                    e = self.edges[key]
                    pe = self.prop_weight(e)
                    if pe <= 0.5:
                        continue
                    v = e.j if e.i == u else e.i
                    w = wu * pe
                    if w < self.cfg.prop_min_w or w <= best.get(v, 0.0):
                        continue
                    best[v] = w
                    if self.segs[v].vlm_logits:
                        found[v] = w
                    else:
                        nxt.append(v)
            if not nxt:
                break
            frontier = nxt
        if not found:
            return max(min(s.prior_logit, 12.0), -12.0), 0.0
        tot = sum(found.values())
        z_ev = sum(w * self.seg_logit(self.segs[k]) for k, w in found.items()) / tot
        miss = 1.0
        for w in found.values():
            miss *= 1.0 - w
        sup = 1.0 - miss                                  # P(at least one answer is about this object)
        z = sup * z_ev + (1.0 - sup) * s.prior_logit
        return max(min(z, 12.0), -12.0), sup

    def _effective_logit(self, sid: int, gamma: float = 0.5) -> float:
        s = self.segs[sid]
        z = self.seg_logit(s)
        for w, d in self.chain_neighbors(sid):
            z += (gamma ** d) * self.seg_logit(self.segs[w]) * min(1.0, self.segs[w].n / max(s.n, 1))
        return max(min(z, 12.0), -12.0)
