"""Window solver: joint referring labels + associations over the tracklet hypothesis graph.

    maximise   sum_i  n_i (z_i - tau) y_i                       referring evidence (seconds x log-odds)
             + kappa sum_e  clip(l_e) x_e                       link evidence
             - lam   sum_e  d_e   - lam sum_i c_i               label changes along chosen links /
                                                                w.r.t. the node's own committed prefix
    s.t.       sum_out x <= 1,  sum_in x <= 1                    each piece continues at most one way
               d_e >= |y_i - y_j| - (1 - x_e)
               decided edges are constants; committed frames are never re-output (frozen boundary),
               but committed pieces inside the window stay as variables so their evidence keeps
               propagating to the pieces that follow ("frozen outputs, live beliefs")

``y`` = referred, ``x`` = link chosen. Solved exactly with OR-Tools CP-SAT; without OR-Tools a
greedy link choice + per-chain 2-state Viterbi is used (exact for the labels given the links).
"""
from __future__ import annotations

from dataclasses import dataclass

from .tracklets import TrackletGraph


@dataclass
class SolverConfig:
    tau: float = 0.0                # decision threshold on the log-odds
    lam: float = 1.0                # cost of a label change along a link (seconds of logit-1 evidence)
    kappa: float = 0.5              # weight of link evidence
    clip: float = 6.0
    min_link_logit: float = -2.0    # edges below this are never chosen
    time_limit_s: float = 0.2
    workers: int = 4                # CP-SAT threads for large components
    backend: str = "auto"           # auto | cpsat | greedy
    scale: int = 100
    future_s: float = 1.0           # an open piece is expected to last this much longer
    unverified_w: float = 0.3       # weight of the prior of a piece the VLM has not answered yet
                                    # (raised towards 1 by the support of answers about the same object)


def _fully_committed(s) -> bool:
    return s.closed and s.committed_upto >= s.n - 1


def _have_ortools() -> bool:
    try:
        import ortools.sat.python.cp_model  # noqa: F401
        return True
    except Exception:
        return False


class WindowSolver:
    def __init__(self, cfg: SolverConfig | None = None):
        self.cfg = cfg or SolverConfig()
        self.backend = self.cfg.backend
        if self.backend == "auto":
            self.backend = "cpsat" if _have_ortools() else "greedy"
        self.last_status = ""
        self.hint_y: dict = {}

    def _gains(self, g: TrackletGraph, nodes: list, z: dict, support: dict | None = None) -> dict:
        c = self.cfg
        out = {}
        for sid in nodes:
            s = g.segs[sid]
            # evidence is about the whole piece, committed or not ("frozen outputs, live beliefs")
            answered = bool(s.vlm_logits)
            sup = 1.0 if answered else (support or {}).get(sid, 0.0)
            w = s.n / g.cfg.fps + (c.future_s if (sup > 0.5 and not s.closed) else 0.0)
            if not answered:
                w *= c.unverified_w + (1.0 - c.unverified_w) * sup
            zz = max(min(z[sid], c.clip), -c.clip)
            out[sid] = w * (zz - c.tau)
        return out

    def solve(self, g: TrackletGraph, free: list, z: dict | None = None, only: set | None = None,
              support: dict | None = None) -> tuple[dict, dict]:
        """``free`` = segment ids whose label is still open. Returns (y, x) for the free nodes and
        every edge touching them. ``only``: re-solve just the components containing these nodes."""
        c = self.cfg
        if z is None:
            z = {sid: g.seg_logit(g.segs[sid]) for sid in free}
        free_set = set(free)
        edges = {}
        for sid in free:
            for key in g.out_edges[sid] + g.in_edges[sid]:
                e = g.edges[key]
                if e.fixed is None and g.edge_logit(e) < c.min_link_logit:
                    continue
                edges[key] = e
        gains = self._gains(g, free, z, support)
        if self.backend != "cpsat" or not free:
            return self._greedy(g, free_set, edges, gains)
        # independent connected components (through candidate edges) are solved separately
        parent = {sid: sid for sid in free}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for (i, j) in edges:
            if i in parent and j in parent:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
        comps: dict = {}
        for sid in free:
            comps.setdefault(find(sid), []).append(sid)
        ys, xs, status = {}, {}, set()
        for nodes in comps.values():
            ns = set(nodes)
            if only is not None and not (ns & only):
                continue                                  # nothing changed in this component
            ce = {k: e for k, e in edges.items() if k[0] in ns or k[1] in ns}
            if len(nodes) == 1 and not ce:
                ys[nodes[0]] = int(gains[nodes[0]] > 0)
                continue
            chain = self._as_chain(g, ns, ce)
            if chain is not None:                         # exact DP, no MIP needed
                y, x = self._chain_dp(g, chain, ns, ce, gains)
                ys.update(y)
                xs.update(x)
                status.add("OPTIMAL")
                continue
            try:
                y, x = self._cpsat(g, ns, ce, gains)
            except Exception as ex:                       # pragma: no cover - solver failure
                self.last_status = f"cpsat failed: {ex}"
                y, x = self._greedy(g, ns, ce, gains)
            status.add(self.last_status)
            ys.update(y)
            xs.update(x)
        self.last_status = "OPTIMAL" if status <= {"OPTIMAL"} else "/".join(sorted(status))
        return ys, xs

    # ------------------------------------------------------------------ exact
    def _const_y(self, g, sid):
        s = g.segs[sid]
        return s.y_committed if s.y_committed is not None else 0

    def _cpsat(self, g, free, edges, gains):
        from ortools.sat.python import cp_model
        c = self.cfg
        S = c.scale
        m = cp_model.CpModel()
        y = {sid: m.NewBoolVar(f"y{sid}") for sid in free}
        obj = []
        for sid in free:
            obj.append(int(round(S * gains[sid])) * y[sid])
            s = g.segs[sid]
            if s.y_committed is not None and not _fully_committed(s):   # own committed prefix (anti-flicker)
                if s.y_committed:
                    obj.append(-int(round(S * c.lam)) * (1 - y[sid]))
                else:
                    obj.append(-int(round(S * c.lam)) * y[sid])
        x, outs, ins = {}, {}, {}
        for key, e in edges.items():
            if e.fixed is not None:
                xv = int(e.fixed)
            else:
                xv = m.NewBoolVar(f"x{e.i}_{e.j}")
                l = max(min(g.edge_logit(e), c.clip), -c.clip)
                obj.append(int(round(S * c.kappa * l)) * xv)
            x[key] = xv
            outs.setdefault(e.i, []).append(xv)
            ins.setdefault(e.j, []).append(xv)
            yi = y.get(e.i, None)
            yj = y.get(e.j, None)
            yi = yi if yi is not None else self._const_y(g, e.i)
            yj = yj if yj is not None else self._const_y(g, e.j)
            if isinstance(yi, int) and isinstance(yj, int):
                if yi != yj and not isinstance(xv, int):
                    obj.append(-int(round(S * c.lam)) * xv)
                continue
            d = m.NewBoolVar(f"d{e.i}_{e.j}")
            one_minus_x = 1 - xv if not isinstance(xv, int) else 1 - xv
            m.Add(d >= yi - yj - one_minus_x)
            m.Add(d >= yj - yi - one_minus_x)
            obj.append(-int(round(S * c.lam)) * d)
        # degree constraints include edges not in the window (fixed elsewhere)
        for sid, lst in outs.items():
            extra = sum(1 for k in g.out_edges.get(sid, []) if k not in edges and g.edges[k].fixed == 1)
            m.Add(sum(lst) <= 1 - extra)
        for sid, lst in ins.items():
            extra = sum(1 for k in g.in_edges.get(sid, []) if k not in edges and g.edges[k].fixed == 1)
            m.Add(sum(lst) <= 1 - extra)
        for sid, v in y.items():                          # warm start from the previous solve
            if sid in self.hint_y:
                m.AddHint(v, self.hint_y[sid])
        m.Maximize(sum(obj))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = c.time_limit_s
        solver.parameters.num_workers = c.workers if len(free) > 200 else 1
        st = solver.Solve(m)
        self.last_status = solver.StatusName(st)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._greedy(g, free, edges, gains)
        ys = {sid: int(solver.Value(v)) for sid, v in y.items()}
        xs = {k: (v if isinstance(v, int) else int(solver.Value(v))) for k, v in x.items()}
        self.hint_y.update(ys)
        return ys, xs

    # ------------------------------------------------------------------ exact chain DP
    def _as_chain(self, g, ns: set, ce: dict):
        """Node order if the component is a simple path through its candidate edges, else None.
        Edges to nodes outside the component are allowed only at the two ends."""
        out_n, in_n = {}, {}
        for (i, j) in ce:
            if i in ns and j in ns:
                if i in out_n or j in in_n:
                    return None
                out_n[i], in_n[j] = j, i
        heads = [n for n in ns if n not in in_n]
        if len(heads) != 1:
            return None
        order = [heads[0]]
        while order[-1] in out_n:
            order.append(out_n[order[-1]])
            if len(order) > len(ns):
                return None
        if len(order) != len(ns):
            return None
        # boundary edges (to/from outside nodes): at most one in at the head, one out at the tail
        ext_in = [k for k in ce if k[1] in ns and k[0] not in ns]
        ext_out = [k for k in ce if k[0] in ns and k[1] not in ns]
        if len(ext_in) > 1 or len(ext_out) > 1:
            return None
        if ext_in and ext_in[0][1] != order[0]:
            return None
        if ext_out and ext_out[0][0] != order[-1]:
            return None
        if ext_in and order[0] in in_n or ext_out and order[-1] in out_n:
            return None
        return order

    def _trans(self, g, e, u: int, v: int) -> tuple:
        """Best (reward, x) of edge e between labels u -> v."""
        c = self.cfg
        pen = c.lam if u != v else 0.0
        if e.fixed is not None:
            return (-pen if e.fixed else 0.0), int(e.fixed)
        keep = c.kappa * max(min(g.edge_logit(e), c.clip), -c.clip) - pen
        return (keep, 1) if keep > 0 else (0.0, 0)

    def _chain_dp(self, g, order, ns, ce, gains):
        c = self.cfg
        def unary(n, v):
            s = g.segs[n]
            r = gains[n] if v else 0.0
            if s.y_committed is not None and not _fully_committed(s) and v != s.y_committed:
                r -= c.lam
            return r
        ext_in = [k for k in ce if k[1] in ns and k[0] not in ns]
        ext_out = [k for k in ce if k[0] in ns and k[1] not in ns]
        score = []
        for v in (0, 1):
            r = unary(order[0], v)
            if ext_in:
                r += self._trans(g, ce[ext_in[0]], self._const_y(g, ext_in[0][0]), v)[0]
            score.append(r)
        back = [None]
        for idx in range(1, len(order)):
            e = ce[(order[idx - 1], order[idx])]
            row, bk = [], []
            for v in (0, 1):
                cand = [score[u] + self._trans(g, e, u, v)[0] for u in (0, 1)]
                u = 0 if cand[0] >= cand[1] else 1
                row.append(cand[u] + unary(order[idx], v))
                bk.append(u)
            score, _ = row, back.append(bk)
        if ext_out:
            score = [score[v] + self._trans(g, ce[ext_out[0]], v, self._const_y(g, ext_out[0][1]))[0] for v in (0, 1)]
        v = 0 if score[0] >= score[1] else 1
        ys = {}
        for idx in range(len(order) - 1, -1, -1):
            ys[order[idx]] = v
            if idx > 0:
                v = back[idx][v]
        xs = {}
        for k, e in ce.items():
            u = ys[k[0]] if k[0] in ys else self._const_y(g, k[0])
            w = ys[k[1]] if k[1] in ys else self._const_y(g, k[1])
            xs[k] = self._trans(g, e, u, w)[1]
        return ys, xs

    # ------------------------------------------------------------------ fallback
    def _greedy(self, g, free, edges, gains):
        c = self.cfg
        xs, out_used, in_used = {}, set(), set()
        for k, e in g.edges.items():                      # degrees taken by decided edges
            if e.fixed == 1:
                out_used.add(e.i)
                in_used.add(e.j)
        for k, e in edges.items():
            if e.fixed is not None:
                xs[k] = int(e.fixed)
        order = sorted((k for k, e in edges.items() if e.fixed is None), key=lambda k: -g.edge_logit(edges[k]))
        for k in order:
            e = edges[k]
            if g.edge_logit(e) <= 0 or e.i in out_used or e.j in in_used:
                xs[k] = 0
                continue
            xs[k] = 1
            out_used.add(e.i)
            in_used.add(e.j)
        nxt = {k[0]: k[1] for k, v in xs.items() if v}
        prv = {k[1]: k[0] for k, v in xs.items() if v}
        ys, done = {}, set()
        for sid in free:
            if sid in done:
                continue
            head = sid                                     # walk back to the start of the free chain
            while head in prv and prv[head] in free and prv[head] not in done:
                head = prv[head]
                if head == sid:
                    break
            chain = [head]
            while chain[-1] in nxt and nxt[chain[-1]] in free and nxt[chain[-1]] not in chain:
                chain.append(nxt[chain[-1]])
            done.update(chain)
            # boundary constants before/after the chain
            pre = prv.get(chain[0])
            pre_y = self._const_y(g, pre) if (pre is not None and pre not in free) else None
            post = nxt.get(chain[-1])
            post_y = self._const_y(g, post) if (post is not None and post not in free) else None
            # Viterbi over 2 states
            INF = float("-inf")
            score = [[INF, INF]]
            back = []
            for idx, n in enumerate(chain):
                s = g.segs[n]
                unary = [0.0, gains[n]]
                if s.y_committed is not None and not _fully_committed(s):
                    unary[1 - s.y_committed] -= c.lam
                if idx == 0:
                    row = [unary[v] - (c.lam if (pre_y is not None and v != pre_y) else 0) for v in (0, 1)]
                    score = [row]
                    back.append([0, 0])
                    continue
                prev = score[-1]
                row, bk = [], []
                for v in (0, 1):
                    cand = [prev[u] - (c.lam if u != v else 0) for u in (0, 1)]
                    u = 0 if cand[0] >= cand[1] else 1
                    row.append(cand[u] + unary[v])
                    bk.append(u)
                score.append(row)
                back.append(bk)
            last = score[-1]
            if post_y is not None:
                last = [last[v] - (c.lam if v != post_y else 0) for v in (0, 1)]
            v = 0 if last[0] >= last[1] else 1
            for idx in range(len(chain) - 1, -1, -1):
                ys[chain[idx]] = v
                v = back[idx][v] if idx > 0 else v
        self.last_status = "greedy"
        return ys, xs
