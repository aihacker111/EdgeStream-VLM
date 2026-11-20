"""Tests for online referring MOT: metric, datasets, parser, verifier plumbing, solver, pipeline."""
from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from synthetic_rmot import make_world, write_fake_aerialmind  # noqa: E402

from edgestream.rmot import hota as H  # noqa: E402
from edgestream.rmot.datasets import aerialmind, check_label_format  # noqa: E402
from edgestream.rmot.expression import parse  # noqa: E402
from edgestream.rmot.pipeline import INF, GTOracle, OnlineRMOT, RMOTConfig  # noqa: E402
from edgestream.rmot.scheduler import AsyncVLM, Request, SchedulerConfig  # noqa: E402
from edgestream.rmot.solver import SolverConfig, WindowSolver  # noqa: E402
from edgestream.rmot.tracklets import GraphConfig, TrackletGraph, logit  # noqa: E402
from edgestream.rmot.verifier import CachedVerifier, FrameSource, OracleVerifier, VLMVerifier  # noqa: E402
from edgestream.vlm.backends import ChatResult, field_yes_probability  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKEVAL = os.environ.get("TRACKEVAL_PATH", "/home/claude/refs/JonathonLuiten_TrackEval")


# ---------------------------------------------------------------------- metric
def test_hota_known_values():
    gt = np.array([[f, 1, 10, 10, 20, 20] for f in range(1, 5)], float)
    same = gt.copy()
    r = H.summarize(H.hota_sequence(H.SeqData.from_rows(gt, same, 4)))
    assert abs(r["HOTA"] - 100) < 1e-6 and abs(r["LocA"] - 100) < 1e-6
    sw = gt.copy()
    sw[2:, 1] = 2                                # id switch half way
    r = H.summarize(H.hota_sequence(H.SeqData.from_rows(gt, sw, 4)))
    assert abs(r["DetA"] - 100) < 1e-6 and abs(r["AssA"] - 50) < 1e-6
    assert abs(r["HOTA"] - 100 * math.sqrt(0.5)) < 1e-6
    r = H.summarize(H.hota_sequence(H.SeqData.from_rows(gt, np.zeros((0, 6)), 4)))
    assert r["HOTA"] == 0
    with pytest.raises(ValueError):
        H.SeqData.from_rows(gt, np.array([[5, 1, 0, 0, 1, 1]], float), 4)        # frame out of range
    with pytest.raises(ValueError):
        H.SeqData.from_rows(gt, np.array([[1, 1, 0, 0, 1, 1], [1, 1, 3, 3, 1, 1]], float), 4)   # duplicate id


@pytest.mark.skipif(not os.path.isdir(os.path.join(TRACKEVAL, "trackeval")), reason="TrackEval checkout not available")
def test_hota_matches_trackeval():
    if not hasattr(np, "float"):
        np.float = float
    sys.path.insert(0, TRACKEVAL)
    from trackeval.datasets._base_dataset import _BaseDataset
    from trackeval.metrics.hota import HOTA
    rng = np.random.default_rng(1)
    mine, ref = [], []
    for _ in range(5):
        T = 30
        gt, tr = [], []
        for g in range(int(rng.integers(1, 5))):
            s = int(rng.integers(0, T // 2))
            x, y = rng.uniform(0, 300, 2)
            for f in range(s, T):
                gt.append([f + 1, g + 1, x + 2 * f, y, 25, 30])
                if rng.random() < 0.85:
                    tr.append([f + 1, 10 * g + (f > 20), x + 2 * f + rng.normal(0, 5), y + rng.normal(0, 5), 25, 30])
        gt, tr = np.array(gt), np.array(tr).reshape(-1, 6)
        d = H.SeqData.from_rows(gt, tr, T)
        m = H.hota_sequence(d)
        data = {"gt_ids": d.gt_ids, "tracker_ids": d.tr_ids, "num_gt_dets": len(gt), "num_tracker_dets": len(tr),
                "num_gt_ids": max(int(x.max()) + 1 for x in d.gt_ids if len(x)),
                "num_tracker_ids": max(int(x.max()) + 1 for x in d.tr_ids if len(x)),
                "similarity_scores": [_BaseDataset._calculate_box_ious(g.copy(), t.copy())
                                      for g, t in zip(d.gt_dets, d.tr_dets)]}
        r = HOTA().eval_sequence(data)
        for k in ("HOTA", "DetA", "AssA", "LocA"):
            assert np.allclose(m[k], r[k])
        mine.append(m)
        ref.append(r)
    cm, cr = H.combine(mine), HOTA().combine_sequences({str(i): r for i, r in enumerate(ref)})
    for k in ("HOTA", "DetA", "AssA", "LocA"):
        assert np.allclose(cm[k], cr[k])


# ---------------------------------------------------------------------- datasets
def test_aerialmind_loader_and_gt(tmp_path):
    write_fake_aerialmind(str(tmp_path), ("uav0000009_03358_v",), n=20)
    sp = aerialmind(str(tmp_path), "uav0000009_03358_v")
    v = sp.videos["uav0000009_03358_v"]
    assert len(v) == 20 and v.frame_keys[0] == 1 and v.fmt == "x1y1wh"
    assert check_label_format(v)["best"] == "x1y1wh"
    assert sorted(sp.seqmap) == ["uav0000009_03358_v+Cars", "uav0000009_03358_v+Red cars moving"]
    ex = {e.expression_id: e for e in sp.expressions["uav0000009_03358_v"]}["Cars"]
    rows = v.gt_rows(ex)
    assert rows and min(r[0] for r in rows) >= 1 and max(r[0] for r in rows) <= 20
    # boxes round-trip through the normalised label files
    perc, _, sv, _ = make_world(0, n=20)
    r0 = [r for r in rows if r[0] == 1]
    gt0 = {int(o[0]): o[1:] for o in sv.gt_boxes(0) if int(o[0]) <= 14}
    for _, oid, x, y, w, h in r0:
        assert np.allclose([x, y, w, h], gt0[oid], atol=0.05)


# ---------------------------------------------------------------------- parser
def test_parser():
    q = parse("Cars proceeding in contrary direction to ours")
    assert q.families == {"car"} and q.moving is True and q.rel_ego == "opposite" and q.measurable
    q = parse("Parked white vehicles")
    assert q.moving is False and "car" in q.families and "truck" in q.families and not q.measurable
    q = parse("Persons adorned in dark upper vestments")
    assert q.families == {"person"} and q.moving is None
    q = parse("Vehicles taking left turns in bottom section")
    assert q.turning == "left"
    assert parse("Cars driving from right to left").lateral == "right_to_left"
    assert parse("motor vehicles").families >= {"car", "truck"}
    assert parse("The red vehicles").class_ok("bus") and not parse("pedestrians").class_ok("car")


def test_field_yes_probability():
    toks = [('{"', {}), ("1", {}), ('":', {}), (' "', {}), ("yes", {"yes": math.log(0.6), "no": math.log(0.2)}),
            ('",', {}), (' "', {}), ("2", {}), ('":', {}), (' "no', {' "no': math.log(0.9), ' "yes': math.log(0.1)}),
            ('"}', {})]
    p = field_yes_probability(toks, ["1", "2"])
    assert abs(p["1"] - 0.75) < 1e-6 and abs(p["2"] - 0.1) < 1e-6
    p = field_yes_probability([('{"1": "yes", "2": "no"}', {})], ["1", "2"])    # no logprobs: text
    assert p == {"1": 1.0, "2": 0.0}


# ---------------------------------------------------------------------- verifier plumbing
class _FakeChat:
    name = "fake"

    def __init__(self):
        self.calls = []

    def chat(self, system, parts, max_new_tokens=128, schema=None, top_logprobs=5):
        self.calls.append((system, parts, schema))
        if schema is None:                       # prompt v2: one token
            return ChatResult("Yes", [("Yes", {"Yes": math.log(0.7), "No": math.log(0.3)})], 100, 1, 0.1)
        keys = schema["required"]
        txt = "{" + ", ".join(f'"{k}": "yes"' for k in keys) + "}"
        return ChatResult(txt, [(txt, {})], 100, 10, 0.2)


def test_set_prompt_prefix_is_expression_independent():
    perc, aff, sv, _ = make_world(0, n=30)
    g = TrackletGraph(GraphConfig(fps=10.0))
    frames = FrameSource()
    for r, fo in enumerate(perc):
        g.update(r, r / 10.0, fo.tracks, (1280, 720), aff[r])
        frames.put(r, np.full((720, 1280, 3), 90, np.uint8))
    grp = [s for s in g.segs.values() if s.family == "car" and s.obs[0].rank <= 20 <= s.obs[-1].rank][:4]
    assert len(grp) >= 2
    be = _FakeChat()
    ver = VLMVerifier(be)
    a1 = ver.refer(g, grp, 20, "red cars", frames)
    a2 = ver.refer(g, grp, 20, "parked trucks", frames)
    assert set(a1.probs) == {s.sid for s in grp} and all(p == 1.0 for p in a1.probs.values())
    p1, p2 = be.calls[0][1], be.calls[1][1]
    assert p1[-1][0] == "text" and "red cars" in p1[-1][1] and "parked trucks" in p2[-1][1]
    for (k1, v1), (k2, v2) in zip(p1[:-1], p2[:-1]):     # everything before the expression is identical
        assert k1 == k2 and (np.array_equal(v1, v2) if k1 == "image" else v1 == v2)
    link = ver.link(g, grp[0], grp[1], frames)
    assert list(link.probs.values()) == [1.0]


def test_prompt_v2_one_token_and_shared_prefix():
    perc, aff, sv, _ = make_world(0, n=30)
    g = TrackletGraph(GraphConfig(fps=10.0))
    frames = FrameSource(cache=100)
    for r, fo in enumerate(perc):
        g.update(r, r / 10.0, fo.tracks, (1280, 720), aff[r])
        frames.put(r, np.random.default_rng(r).integers(0, 255, (720, 1280, 3), dtype=np.uint8))
    cands = [s for s in g.segs.values() if s.obs[0].rank <= 20 <= s.obs[-1].rank][:3]
    be = _FakeChat()
    ver = VLMVerifier(be, prompt="v2")
    answers = [ver.refer(g, [s], 20, "red cars", frames) for s in cands]
    assert all(abs(a.probs[s.sid] - 0.7) < 1e-9 for a, s in zip(answers, cands))
    calls = [c[1] for c in be.calls]
    assert all(c[2] is None for c in be.calls)                       # no JSON schema: one token
    # frame image + description are the identical prefix of every candidate's prompt
    for parts in calls[1:]:
        assert parts[0] == calls[0][0] and parts[2] == calls[0][2]
        assert parts[1][1] is calls[0][1][1]                         # same (cached) frame array
        assert parts[3][1] is not calls[0][3][1]                     # candidate close-ups differ
    assert "red cars" in calls[0][2][1] and calls[0][-1][1].endswith("Answer yes or no.")
    ver1 = VLMVerifier(_FakeChat(), prompt="v1")                     # legacy JSON prompt still works
    assert set(ver1.refer(g, [cands[0]], 20, "red cars", frames).probs) == {cands[0].sid}


def test_cached_verifier_exact_replay(tmp_path):
    perc, aff, sv, expr = make_world(2, n=60)
    orc = GTOracle(sv, None)
    cfg = RMOTConfig.from_dict({"lag_s": 2.0, "sched": {"n_workers": 2}})
    path = str(tmp_path / "c.jsonl")
    c1 = CachedVerifier(OracleVerifier(orc.refer, orc.same, acc=0.8, seed=3), path)
    r1 = OnlineRMOT(cfg, c1).run(perc, expr, 10.0, (1280, 720), affines=aff)
    c1.close()
    c2 = CachedVerifier(OracleVerifier(orc.refer, orc.same, acc=0.5, seed=99), path)   # different inner: unused
    r2 = OnlineRMOT(cfg, c2).run(perc, expr, 10.0, (1280, 720), affines=aff)
    assert c2.misses == 0 and c2.hits > 0
    assert r1["rows"] == r2["rows"]


# ---------------------------------------------------------------------- scheduler & solver
def test_async_vlm_pool():
    from edgestream.rmot.verifier import Answer
    q = AsyncVLM(SchedulerConfig(n_workers=2, max_calls=3))
    mk = lambda: Answer("refer", {}, 1.0)    # noqa: E731
    assert q.free_worker(0.0) is not None
    q.submit(0.0, Request("refer", ("a",), []), mk)
    q.submit(0.0, Request("refer", ("b",), []), mk)
    assert q.free_worker(0.5) is None and q.collect(0.9) == []
    assert len(q.collect(1.0)) == 2 and q.free_worker(1.0) is not None
    q.submit(1.0, Request("refer", ("c",), []), mk)
    assert not q.budget_left()


def _chain_graph(rng, n=6):
    g = TrackletGraph(GraphConfig(fps=10.0))
    from perception.types import TrackOutput
    for r in range(n * 10):
        t = TrackOutput(1, 1, 3, "car", np.array([100 + r, 100, 30, 20], np.float32), np.zeros(2, np.float32),
                        0.9, 0.9, r, False)
        g.update(r, r / 10.0, [t], (1280, 720), None)
    for s in g.segs.values():
        s.vlm_logits = [float(rng.normal(0, 2))]
    return g


def test_solver_cpsat_equals_viterbi_on_chains():
    pytest.importorskip("ortools")
    rng = np.random.default_rng(0)
    for _ in range(10):
        g = _chain_graph(rng)
        free = list(g.segs)
        ya, _ = WindowSolver(SolverConfig(backend="cpsat")).solve(g, free)
        yb, _ = WindowSolver(SolverConfig(backend="greedy")).solve(g, free)
        assert ya == yb


def test_solver_smooths_single_wrong_answer():
    g = _chain_graph(np.random.default_rng(0), n=5)
    sids = sorted(g.segs)
    for s in g.segs.values():
        s.vlm_logits = [2.0]
    g.segs[sids[2]].vlm_logits = [-1.0]          # one wrong "no" in a confident "yes" chain
    for be in ("greedy", "cpsat") if _ortools() else ("greedy",):
        y, _ = WindowSolver(SolverConfig(backend=be)).solve(g, sids)
        assert all(y[s] == 1 for s in sids)


def _ortools():
    try:
        import ortools  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------- pipeline
def _run(world_seed, **kw):
    perc, aff, sv, expr = make_world(world_seed)
    orc = GTOracle(sv, None)
    cfg = RMOTConfig.from_dict({"sched": {"n_workers": 2}, **kw})
    out = OnlineRMOT(cfg, OracleVerifier(orc.refer, orc.same, acc=0.85, seed=world_seed)).run(
        perc, expr, 10.0, (1280, 720), affines=aff)
    gt = np.array(sv.gt_rows()).reshape(-1, 6)
    pr = np.array(out["rows"]).reshape(-1, 6)
    return H.hota_sequence(H.SeqData.from_rows(gt, pr, len(perc))), out


def test_search_beats_compute_matched_filter_offline():
    rs, rf = [], []
    for seed in range(3):
        r, out = _run(seed, lag_s=INF)
        rs.append(r)
        calls = out["stats"]["vlm"]["calls"]
        r2, out2 = _run(seed, lag_s=INF, method="filter", filter_propagate="nearest", sched={"n_workers": 2, "max_calls": calls})
        assert out2["stats"]["vlm"]["calls"] <= calls
        rf.append(r2)
    hs, hf = H.summarize(H.combine(rs))["HOTA"], H.summarize(H.combine(rf))["HOTA"]
    assert hs > hf + 10, (hs, hf)


def test_online_lags_and_outputs_are_valid():
    h = {}
    for L in (0.0, 2.0, INF):
        r, out = _run(1, lag_s=L)
        h[L] = H.summarize(r)["HOTA"]
        rows = out["rows"]
        keys = [(r_[0], r_[1]) for r_ in rows]
        assert len(keys) == len(set(keys)) and all(1 <= r_[0] <= 150 for r_ in rows)
    assert h[INF] >= h[0.0] and h[2.0] > h[0.0]


def test_stitching_repairs_identity():
    r_link, out = _run(0, lag_s=INF)
    r_nol, _ = _run(0, lag_s=INF, link_verify=False, graph={"stitch_gap_s": 0.0})
    assert out["stats"]["stitch_edges"] > 0
    assert H.summarize(r_link)["AssA"] > H.summarize(r_nol)["AssA"]


def test_benchmark_script_smoke(tmp_path):
    root = tmp_path / "am"
    tracks = write_fake_aerialmind(str(root), ("uav0000009_03358_v",), n=40)
    out = tmp_path / "run"
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "rmot_benchmark.py"), "--dataset", "aerialmind",
           "--root", str(root), "--split", "in_domain", "--vlm", "oracle", "--tracks-dir", tracks,
           "--methods", "search,filter_kf,prior", "--lags", "2,inf", "--match-compute", "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    assert (out / "summary.md").is_file() and (out / "search_Linf" / "uav0000009_03358_v" / "Cars" / "predict.txt").is_file()
    assert "search" in (out / "summary.md").read_text()


def test_graph_survives_real_tracker_patterns():
    """Same id re-found after the track was closed, duplicate ids in a frame, class flips,
    recovered runs, empty frames - patterns of real AERO-MOT output on VisDrone."""
    from perception.types import TrackOutput
    rng = np.random.default_rng(0)
    g = TrackletGraph(GraphConfig(fps=30.0))
    for r in range(300):
        tracks = []
        for gid in range(1, 12):
            if (r // 20 + gid) % 4 == 0:                  # id disappears > end_after_s, then resumes
                continue
            cls = "car" if (r // 50 + gid) % 7 else "van"
            box = np.array([50 * gid + r, 40 + gid, 30, 20], np.float32) + rng.normal(0, 1, 4).astype(np.float32)
            tracks.append(TrackOutput(gid, gid, 3, cls, box, np.zeros(2, np.float32), 0.8, 0.8, r,
                                      bool(r % 9 == 0)))
        if r % 37 == 0 and tracks:
            tracks.append(tracks[0])                       # duplicate id in one frame
        if r % 61 == 0:
            tracks = []                                    # empty frame
        g.update(r, r / 30.0, tracks, (1920, 1080), None)
    assert all(s.obs for s in g.segs.values())
    assert any(s.cut_reason == "resume" for s in g.segs.values())
    for (i, j), e in g.edges.items():
        assert i != j and g.segs[i].t1 <= g.segs[j].t0


def test_v2_frame_cache_never_mixes_videos(tmp_path):
    """Regression: the v2 frame cache was keyed by rank only and served the previous video's frame."""
    import cv2
    paths = {}
    for vid, val in (("a", 10), ("b", 240)):
        d = tmp_path / vid
        d.mkdir()
        paths[vid] = []
        for r in range(3):
            p = str(d / f"{r:06d}.jpg")
            cv2.imwrite(p, np.full((72, 128, 3), val, np.uint8))
            paths[vid].append(p)
    ver = VLMVerifier(_FakeChat(), prompt="v2", frame_width=64)
    fa = ver._frame_image(FrameSource(paths["a"]), 1)
    fb = ver._frame_image(FrameSource(paths["b"]), 1)
    assert fa.mean() < 50 and fb.mean() > 200
