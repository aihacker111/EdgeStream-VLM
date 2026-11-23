"""Belief model: exact inference and value of information, checked against brute-force enumeration."""
import itertools
import math
import random
from types import SimpleNamespace as NS

from edgestream.rmot.belief import BeliefConfig, BeliefModel


def _random_graph(n, seed):
    rnd = random.Random(seed)
    segs, edges, in_edges = {}, {}, {k: [] for k in range(n)}
    for k in range(n):
        segs[k] = NS(t0=float(k), t1=k + 0.9, cut_reason=rnd.choice(["length", "jump", "appearance", "resume"]))
    for k in range(1, n):
        for _ in range(rnd.randint(1, 2)):
            p = rnd.randrange(0, k)
            e = NS(i=p, j=k, kind=rnd.choice(["cont", "stitch"]), prior_logit=rnd.uniform(-2, 3), vlm_logits=[], fixed=None)
            edges[(p, k)] = e
            in_edges[k].append((p, k))
    g = NS(segs=segs, edges=edges, in_edges=in_edges, edge_logit=lambda e: e.prior_logit)
    L = {k: rnd.uniform(-3, 3) * (rnd.random() < 0.6) for k in range(n)}
    return g, L


def _brute(m, extra=None):
    extra = extra or {}
    nodes = m.nodes
    Z, marg, joint = 0.0, {n: 0.0 for n in nodes}, {}
    for bits in itertools.product([0, 1], repeat=len(nodes)):
        r = dict(zip(nodes, bits))
        p = 1.0
        for n in nodes:
            lo = m.L[n] + extra.get(n, 0.0)
            p *= 1 / (1 + math.exp(lo)) if r[n] == 0 else 1 / (1 + math.exp(-lo))
            if n in m.parent:
                a, b = m.ab[n]
                q1 = b if r[m.parent[n]] else a
                p *= q1 if r[n] else 1 - q1
            else:
                p *= m.rho if r[n] else 1 - m.rho
        Z += p
        for n in nodes:
            if r[n]:
                marg[n] += p
        joint[bits] = p
    return {n: v / Z for n, v in marg.items()}, joint


def test_marginals_and_conditionals_are_exact():
    for seed in range(20):
        g, L = _random_graph(7, seed)
        m = BeliefModel(g, range(7), BeliefConfig(), "motion", 0.3, llr=lambda s: L[s], kin=lambda s: 0.0)
        mb, joint = _brute(m)
        assert max(abs(mb[k] - m.mu[k]) for k in range(7)) < 1e-9
        i = seed % 7
        cond = m.conditionals(i)
        for v in (0, 1):
            num, den = {j: 0.0 for j in m.nodes}, 0.0
            for bits, p in joint.items():
                r = dict(zip(m.nodes, bits))
                if r[i] != v:
                    continue
                den += p
                for j in m.nodes:
                    num[j] += p * r[j]
            for j, cv in cond.items():
                assert abs(num[j] / den - cv[v]) < 1e-9


def test_voi_is_exact_with_and_without_a_question_in_flight():
    tpr, fpr, th, w = 0.85, 0.12, 0.3, 0.7
    ly, ln = math.log(tpr / fpr), math.log((1 - tpr) / (1 - fpr))
    for seed in range(12):
        g, L = _random_graph(7, 100 + seed)
        m = BeliefModel(g, range(7), BeliefConfig(), "static", 0.25, llr=lambda s: L[s], kin=lambda s: 0.0)
        f = {k: 1.0 + k for k in range(7)}
        i = seed % 7
        tree = m.trees[m.root_of[i]]

        def brute_voi(extra):
            base, _ = _brute(m, extra)
            py = tpr * base[i] + fpr * (1 - base[i])
            my, _ = _brute(m, {**extra, i: extra.get(i, 0.0) + w * ly})
            mn, _ = _brute(m, {**extra, i: extra.get(i, 0.0) + w * ln})
            v = sum(f[j] * (m.loss(base[j], th) - py * m.loss(my[j], th) - (1 - py) * m.loss(mn[j], th)) for j in tree)
            return max(v, 0.0)
        assert abs(brute_voi({}) - m.voi(i, f, th, w, tpr, fpr)) < 1e-9
        others = [k for k in tree if k != i]
        if others:
            s = others[0]
            base, _ = _brute(m)
            ps = tpr * base[s] + fpr * (1 - base[s])
            want = ps * brute_voi({s: ly}) + (1 - ps) * brute_voi({s: ln})
            assert abs(want - m.voi(i, f, th, w, tpr, fpr, pending={s: (1.0, tpr, fpr)})) < 1e-9


def test_voi_prefers_unknown_objects_and_ignores_committed_frames():
    """A chain already answered 'yes' is worth less than a fresh chain; frames that will be
    committed before the answer returns are worth nothing."""
    segs = {k: NS(t0=float(k % 5), t1=k % 5 + 0.9, cut_reason="length") for k in range(10)}
    edges, in_edges = {}, {k: [] for k in range(10)}
    for k in list(range(1, 5)) + list(range(6, 10)):
        e = NS(i=k - 1, j=k, kind="cont", prior_logit=3.0, vlm_logits=[], fixed=None)
        edges[(k - 1, k)] = e
        in_edges[k].append((k - 1, k))
    g = NS(segs=segs, edges=edges, in_edges=in_edges, edge_logit=lambda e: e.prior_logit)
    L = {k: 0.0 for k in range(10)}
    L[0] = 4.0                                              # chain A: piece 0 answered yes
    m = BeliefModel(g, range(10), BeliefConfig(), "static", 0.3, llr=lambda s: L[s], kin=lambda s: 0.0)
    f = {k: 30.0 for k in range(10)}
    th = 0.4
    assert m.voi(7, f, th, 1.0, 0.85, 0.12) > m.voi(3, f, th, 1.0, 0.85, 0.12)
    assert m.voi(7, {k: 0.0 for k in range(10)}, th, 1.0, 0.85, 0.12) == 0.0
