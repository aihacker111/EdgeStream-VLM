"""Referring belief over the tracklet hypothesis graph: one probabilistic model in place of
hand-set propagation weights, unverified-piece weights and coverage heuristics.

Model
-----
Every piece ``i`` (<= 1 s of one tracker track) has a hidden state ``r_i in {0, 1}``: the piece
shows an object the expression refers to. Pieces are joined by hypothesis edges ``e = (i -> j)``
(same tracker id after a cut, or a stitch after a gap); each piece keeps its most probable
predecessor, so the pieces of one object hypothesis form a tree (a chain, branching where a
track may continue in several ways).

    P(r_root = 1)              = rho                                         base rate of the expression
    P(r_j = 1 | r_i)           = rho + lambda_e (r_i - rho)                  edge channel
    lambda_e                   = pi_e * exp(-h * dt_e)
        pi_e   = P(i and j show the same object)        cut reason / stitch evidence (fitted on train)
        h      = rate at which the referred state of ONE object changes      (0 for "red car",
                 > 0 for "car turning left"; per expression type, fitted on train)
        dt_e   = time between the centres of the two pieces
    log P(obs_i | r_i = 1) / P(obs_i | r_i = 0)
               = w_kin * kin_i + sum_k  c_k * LLR(answer_k)                  measured motion + VLM answers
        LLR    = calibrated logit - logit(base rate of the calibration set); c_k = 1 / (1 + kappa k)
                 discounts the k-th answer about the same piece (answers are correlated)

When ``pi_e = 1`` and ``h = 0`` the edge copies the state (a filter propagating along a track); when
``pi_e = 0`` it is independent of the past (a fresh object). Every value in between is a mixture,
so an identity switch, a state change and an unreliable answer are all handled by the same rule.

Inference: exact sum-product on each tree (linear time) gives the marginals ``mu_i``.
``rho`` is estimated per stream by empirical Bayes (Beta prior fitted on train, updated with the
marginals of the answered pieces).

Decisions
---------
Output a piece iff ``mu_i > theta``. For a Jaccard-type score (DetA) the expected score is
maximised by ``theta = J / (1 + J)``, ``J`` the attainable score (the F-measure analogue of
"threshold = F*/2"); ``J`` is estimated from the marginals by a fixed point.

Questions (value of information)
--------------------------------
Asking piece ``i`` returns yes / no with ``P(yes | r = 1) = TPR``, ``P(yes | r = 0) = FPR``. The
value of the question is the expected number of frame errors it removes, counted only on frames
that are still undecided when the answer arrives:

    VOI(i) = sum_j f_j ( l(mu_j) - sum_a P(a) l(mu_j | a) ),     l(m) = min(theta (1 - m), (1 - theta) m)

with ``mu_j | a`` from the exact conditionals ``P(r_j | r_i)`` of the tree. Questions still in flight
are integrated out (their outcomes enumerated). An object nobody has asked about is uncertain over
many frames and is asked first; a known object is asked again only where its answers disagree or
where its state may have changed; a piece whose frames will be committed before the answer comes
back is worth nothing. VOI of a set of questions is adaptive submodular, so greedy selection is
within (1 - 1/e) of the best adaptive policy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BeliefConfig:
    enabled: bool = False
    # Beta prior on the base rate rho, per expression type (static attributes / motion)
    rho_prior: dict = field(default_factory=lambda: {"static": [1.0, 3.0], "motion": [1.0, 6.0]})
    # rate (1/s) at which the referred state of one object changes
    hazard: dict = field(default_factory=lambda: {"static": 0.01, "motion": 0.2})
    # P(same object) across a cut, by cut reason; stitch edges: sigmoid(a * edge_logit + b)
    pi_reason: dict = field(default_factory=lambda: {"length": 0.995, "jump": 0.5, "appearance": 0.6,
                                                     "class": 0.8, "recovered": 0.8, "resume": 0.8})
    stitch_ab: list = field(default_factory=lambda: [1.0, 0.0])
    kin_w: float = 1.0              # default: measured-motion log-likelihood ratio = kin_w * score
    # fitted on train: P(referred | score) = sigmoid(a * score + b) over pieces of motion expressions
    # whose base rate is kin_base, so the likelihood ratio is a * score + b - logit(kin_base)
    kin_ab: list | None = None
    kin_base: float = 0.15
    ans_corr: float = 0.5           # kappa: the k-th answer about the same piece counts 1 / (1 + kappa k)
    # verifier calibration used ONLY by this model (baselines keep their own decision rule):
    # LLR = calib_ab[0] * stored logit + calib_ab[1] - logit(calib_base_rate)
    calib_ab: list | None = None
    calib_base_rate: float = 0.5    # base rate of the set the verifier was calibrated on
    tpr: float = 0.85               # verifier P(yes | referred)        (fitted on held-out train)
    fpr: float = 0.12               # verifier P(yes | not referred)
    llr_cap: float = 8.0
    theta: float | None = None      # None: expected-DetA-optimal threshold J / (1 + J)
    theta_range: list = field(default_factory=lambda: [0.1, 0.7])
    memory_s: float = 60.0          # pieces this old still carry evidence (online)
    voi_topk: int = 160             # VOI evaluated for at most this many candidates per round
    max_pending_per_tree: int = 2   # questions in flight per object tree that VOI integrates out
    lam: float = 0.1                # solver: label change along a chosen link (smoothing is in mu)


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 40.0), -40.0)))


def _norm(a: float, b: float) -> tuple:
    s = a + b
    if s <= 0:
        return 0.5, 0.5
    return a / s, b / s


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def query_type(q) -> str:
    return "motion" if q is not None and q.has_kinematics() else "static"


class BeliefModel:
    """Trees of pieces with exact marginals and the conditionals needed for VOI.

    ``llr(sid) -> float``: total log-likelihood ratio of the answers about piece ``sid``.
    ``kin(sid) -> float``: measured-motion score of the piece (multiplied by ``kin_w``).
    """

    def __init__(self, g, nodes, cfg: BeliefConfig, qtype: str, rho: float, llr, kin):
        self.g, self.cfg = g, cfg
        self.rho = min(max(rho, 1e-3), 1 - 1e-3)
        self.h = float(cfg.hazard.get(qtype, 0.0))
        self.nodes = list(nodes)
        ns = set(self.nodes)
        self.parent: dict = {}
        self.ab: dict = {}                 # child -> (P(r_c=1 | r_p=0), P(r_c=1 | r_p=1))
        self.children: dict = {n: [] for n in self.nodes}
        for j in self.nodes:
            best, bp = None, 0.0
            for key in g.in_edges.get(j, []):
                e = g.edges[key]
                if e.i not in ns or e.fixed == 0:
                    continue
                pi = self.edge_pi(e)
                if pi > bp:
                    best, bp = e.i, pi
            if best is None:
                continue
            self.parent[j] = best
            self.children[best].append(j)
            dt = max(0.0, self._mid(j) - self._mid(best))
            lam = min(bp * math.exp(-self.h * dt), 1 - 1e-6)
            r = self.rho
            self.ab[j] = (r * (1 - lam), r + lam * (1 - r))
        if cfg.kin_ab and qtype == "motion":
            ka, kb = cfg.kin_ab
            off = kb - _logit(cfg.kin_base)
            self.L = {n: ka * kin(n) + off + llr(n) for n in self.nodes}
        else:
            self.L = {n: kin(n) * cfg.kin_w + llr(n) for n in self.nodes}
        # trees
        self.root_of: dict = {}
        self.trees: dict = {}
        for r in (n for n in self.nodes if n not in self.parent):
            order, stack = [], [r]
            while stack:
                u = stack.pop()
                order.append(u)
                self.root_of[u] = r
                stack.extend(self.children[u])
            self.trees[r] = order               # pre-order: parents before children
        self.mu: dict = {}
        self._bp: dict = {}
        for r, order in self.trees.items():
            bp = self._run_bp(order, {})
            self._bp[r] = bp
            self.mu.update(bp["mu"])

    # ------------------------------------------------------------------ structure
    def _mid(self, sid: int) -> float:
        s = self.g.segs[sid]
        return 0.5 * (s.t0 + s.t1)

    def edge_pi(self, e) -> float:
        if e.fixed == 1:
            return 1.0
        if e.vlm_logits:                                   # a link answer: posterior of the edge
            return _sig(self.g.edge_logit(e))
        if e.kind == "cont":
            reason = self.g.segs[e.j].cut_reason
            if reason in self.cfg.pi_reason:
                return float(self.cfg.pi_reason[reason])
            return _sig(e.prior_logit)
        a, b = self.cfg.stitch_ab
        return _sig(a * e.prior_logit + b)

    # ------------------------------------------------------------------ sum-product on one tree
    def _run_bp(self, order: list, extra: dict) -> dict:
        """Messages, inside/outside terms and marginals of one tree. ``extra``: virtual evidence
        (sid -> log-likelihood ratio) for questions whose answer is being enumerated."""
        phi = {}
        for n in order:
            p1 = _sig(self.L[n] + extra.get(n, 0.0))
            phi[n] = (1 - p1, p1)
        up, inside = {}, {}
        for c in reversed(order):                          # children before parents
            i0, i1 = phi[c]
            for k in self.children[c]:
                u0, u1 = up[k]
                i0, i1 = i0 * u0, i1 * u1
            i0, i1 = _norm(i0, i1)
            inside[c] = (i0, i1)
            if c in self.parent:
                a, b = self.ab[c]
                up[c] = _norm((1 - a) * i0 + a * i1, (1 - b) * i0 + b * i1)
        top, full, mu = {}, {}, {}
        for n in order:                                    # parents before children
            if n in self.parent:
                p = self.parent[n]
                f0, f1 = full[p]
                u0, u1 = up[n]
                o0, o1 = _norm(f0 / max(u0, 1e-300), f1 / max(u1, 1e-300))   # p's belief without n
                a, b = self.ab[n]
                top[n] = _norm(o0 * (1 - a) + o1 * (1 - b), o0 * a + o1 * b)
            else:
                top[n] = (1 - self.rho, self.rho)
            t0, t1 = top[n]
            i0, i1 = inside[n]
            full[n] = _norm(t0 * i0, t1 * i1)
            mu[n] = full[n][1]
        return {"up": up, "inside": inside, "full": full, "mu": mu}

    def conditionals(self, i: int, bp: dict | None = None) -> dict:
        """``j -> (P(r_j = 1 | r_i = 0), P(r_j = 1 | r_i = 1))`` for every piece of ``i``'s tree,
        given all current evidence (exact: the tree is Markov)."""
        bp = bp or self._bp[self.root_of[i]]
        up, inside, full = bp["up"], bp["inside"], bp["full"]
        out = {i: (0.0, 1.0)}
        stack = [i]
        while stack:
            u = stack.pop()
            m0, m1 = out[u]
            nbrs = [(k, "child") for k in self.children[u]]
            if u in self.parent:
                nbrs.append((self.parent[u], "parent"))
            for j, rel in nbrs:
                if j in out:
                    continue
                if rel == "child":
                    a, b = self.ab[j]
                    i0, i1 = inside[j]
                    c0 = a * i1 / max((1 - a) * i0 + a * i1, 1e-300)
                    c1 = b * i1 / max((1 - b) * i0 + b * i1, 1e-300)
                else:                                      # j is u's parent: j's belief without u
                    a, b = self.ab[u]
                    f0, f1 = full[j]
                    u0, u1 = up[u]
                    o0, o1 = f0 / max(u0, 1e-300), f1 / max(u1, 1e-300)
                    w0, w1 = o0 * (1 - a), o1 * (1 - b)    # r_u = 0
                    c0 = w1 / max(w0 + w1, 1e-300)
                    w0, w1 = o0 * a, o1 * b                # r_u = 1
                    c1 = w1 / max(w0 + w1, 1e-300)
                out[j] = ((1 - m0) * c0 + m0 * c1, (1 - m1) * c0 + m1 * c1)
                stack.append(j)
        return out

    # ------------------------------------------------------------------ decisions
    def theta(self, weight: dict) -> float:
        """Expected-DetA-optimal threshold J / (1 + J) (fixed point), clipped to theta_range."""
        if self.cfg.theta is not None:
            return float(self.cfg.theta)
        lo, hi = self.cfg.theta_range
        th = 0.5
        for _ in range(20):
            tp = fp = fn = 0.0
            for n, m in self.mu.items():
                w = weight.get(n, 0.0)
                if m > th:
                    tp += w * m
                    fp += w * (1 - m)
                else:
                    fn += w * m
            J = tp / max(tp + fp + fn, 1e-9)
            nt = min(max(J / (1 + J), lo), hi)
            if abs(nt - th) < 1e-4:
                th = nt
                break
            th = nt
        return th

    @staticmethod
    def loss(m: float, th: float) -> float:
        return min(th * (1 - m), (1 - th) * m)

    def voi(self, i: int, f: dict, th: float, w_i: float, tpr: float, fpr: float,
            pending: dict | None = None) -> float:
        """Expected frame errors removed by asking piece ``i`` (one answer, weight ``w_i``).
        ``f``: undecided frames per piece at answer time; ``pending``: sid -> (weight, tpr, fpr)
        of questions in flight in this tree, whose outcomes are integrated out."""
        root = self.root_of[i]
        order = self.trees[root]
        pend = [(s, v) for s, v in (pending or {}).items() if s in self.root_of and self.root_of[s] == root]
        pend = pend[: self.cfg.max_pending_per_tree]
        total = 0.0
        for prob, extra in self._scenarios(order, pend):
            bp = self._run_bp(order, extra) if extra else self._bp[root]
            total += prob * self._voi_one(i, bp, f, th, w_i, tpr, fpr)
        return total

    def _scenarios(self, order, pend, extra=None, prob=1.0):
        extra = dict(extra or {})
        if not pend:
            yield prob, extra
            return
        (s, (w, tp, fp)), rest = pend[0], pend[1:]
        bp = self._run_bp(order, extra) if extra else self._bp[self.root_of[s]]
        m = bp["mu"][s]
        py = tp * m + fp * (1 - m)
        for pa, llr in ((py, math.log(tp / fp)), (1 - py, math.log((1 - tp) / (1 - fp)))):
            if pa <= 1e-9:
                continue
            e2 = dict(extra)
            e2[s] = e2.get(s, 0.0) + w * llr
            yield from self._scenarios(order, rest, e2, prob * pa)

    def _voi_one(self, i, bp, f, th, w_i, tpr, fpr) -> float:
        mu = bp["mu"]
        m = mu[i]
        cond = self.conditionals(i, bp)
        py = tpr * m + fpr * (1 - m)
        out = 0.0
        lm = _logit(m)
        post_y = _sig(lm + w_i * math.log(tpr / fpr))
        post_n = _sig(lm + w_i * math.log((1 - tpr) / (1 - fpr)))
        for j, (c0, c1) in cond.items():
            fj = f.get(j, 0.0)
            if fj <= 0:
                continue
            my = (1 - post_y) * c0 + post_y * c1
            mn = (1 - post_n) * c0 + post_n * c1
            out += fj * (self.loss(mu[j], th) - py * self.loss(my, th) - (1 - py) * self.loss(mn, th))
        return max(out, 0.0)

    def subtree_mass(self, j: int, f: dict, th: float) -> float:
        """Expected frame errors of ``j`` and its descendants: what a link question on ``j``'s
        incoming edge can at most remove."""
        out, stack = 0.0, [j]
        while stack:
            u = stack.pop()
            out += f.get(u, 0.0) * self.loss(self.mu[u], th)
            stack.extend(self.children.get(u, []))
        return out

    def tree_mass(self, root: int, f: dict, th: float) -> float:
        return sum(f.get(n, 0.0) * self.loss(self.mu[n], th) for n in self.trees[root])
