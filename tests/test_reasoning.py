"""Unit tests for scene / candidates / gate / memory / VLM parsing."""
import numpy as np
import pytest

from edgestream.candidates import MotionCandidateGenerator, MotionRules
from edgestream.gate import DensityGate, GateConfig, TokenBucket
from edgestream.memory import MemoryConfig, StreamingMemory
from edgestream.scene import SceneState
from edgestream.types import Candidate
from edgestream.vlm import extract_json, normalize, verdict_confidence_from_logprobs

from .synthetic import FPS, FakePerceptor, scenario


def _run_scene(frames, grid=(16, 9)):
    scene = SceneState(fps=FPS, grid=grid)
    gen = MotionCandidateGenerator(MotionRules())
    p = FakePerceptor(frames)
    seen = {}
    for k in range(len(frames)):
        scene.update(p.process(None))
        if k % FPS == 0:
            for c in gen.step(scene):
                seen[c.key()] = c
    return scene, seen


def test_wrong_way_vehicle_becomes_counter_flow_candidate():
    _, seen = _run_scene(scenario(wrong_way_from=3.0))
    cf = [c for c in seen.values() if c.type_hint == "counter_flow"]
    assert [c.track_ids for c in cf] == [[999]]
    assert cf[0].duration >= 2.0


def test_no_counter_flow_without_wrong_way_vehicle():
    _, seen = _run_scene(scenario(wrong_way_from=None))
    assert not [c for c in seen.values() if c.type_hint == "counter_flow"]


def test_bidirectional_cells_do_not_flag_legal_opposite_traffic():
    _, seen = _run_scene(scenario(wrong_way_from=None, bidirectional=True), grid=(8, 4))
    assert not [c for c in seen.values() if c.type_hint == "counter_flow"]


def _cand(i, p_rule=0.9, dur=3.0, typ="speeding"):
    return Candidate(cand_id=i, type_hint=typ, track_ids=[i], t_start=0.0, t_last=dur,
                     s_rule=p_rule, s_anom=0.8, severity_prior=0.5, quality=0.9)


def test_gate_never_exceeds_budget():
    cfg = GateConfig(calls_per_min=6, bucket_capacity=3, reserve_frac=0.0, tau0=0.1, max_sends=99,
                     resend_after_s=0.0)
    gate = DensityGate(cfg)
    sent = 0
    for t in range(0, 120):                                    # 2 minutes, 1 window/s
        cands = [_cand(1000 * t + k) for k in range(20)]
        sel, _ = gate.step(float(t), cands, density=20, density_ref=20)
        sent += len(sel)
    assert sent <= 3 + 6 * 2 + 1                                # capacity + rate * minutes


def test_threshold_rises_with_density_and_low_budget():
    gate = DensityGate(GateConfig())
    lo = gate.threshold(density=5, density_ref=20)
    hi = gate.threshold(density=40, density_ref=20)
    assert hi > lo
    gate.bucket.tokens = 0.0
    assert gate.threshold(density=20, density_ref=20) > gate.threshold.__self__.cfg.tau0 * 0.99


def test_reserve_is_kept_for_critical_candidates():
    b = TokenBucket(rate_per_min=0.0, capacity=1.0, reserve_frac=0.5)
    assert not b.can_spend(1.0, critical=False)
    assert b.can_spend(1.0, critical=True)


def test_priority_grows_with_persistence():
    gate = DensityGate(GateConfig())
    assert gate.priority(_cand(1, dur=4.0)) > gate.priority(_cand(2, dur=0.0))


def test_memory_respects_token_budget_and_logs_evictions():
    frames = scenario(seconds=8.0)
    scene = SceneState(fps=FPS, grid=(16, 9))
    cfg = MemoryConfig(budget_tokens=256, crop_size=112, keyframe_width=224)
    mem = StreamingMemory(cfg)
    p = FakePerceptor(frames)
    img = np.zeros((360, 640, 3), np.uint8)
    for k in range(len(frames)):
        out = p.process(None)
        scene.update(out)
        if k % FPS == 0:
            cands = [Candidate(cand_id=g, type_hint="speeding", track_ids=[g], t_start=0, t_last=scene.t,
                               s_rule=0.5, s_anom=0.5, severity_prior=0.5, priority=0.6)
                     for g in list(scene.active_ids)]
            mem.observe(scene.t, img, scene, cands, tau=0.5)
            assert mem.used_tokens() <= cfg.budget_tokens
    assert mem.scene_log, "evicted slots must be summarised into the scene log"


@pytest.mark.parametrize("text", [
    '{"verdict": "yes", "event_type": "counter_flow", "severity": "high", "description": "x"}',
    '```json\n{"verdict": "yes", "event_type": "counter_flow", "severity": "high", "description": "x"}\n```',
    'Sure! {"verdict": "yes", "event_type": "counter_flow", "severity": "high", "description": "x"} done',
])
def test_extract_json_variants(text):
    obj, ok = normalize(extract_json(text), "counter_flow", "high")
    assert ok and obj["verdict"] == "yes"


def test_unparseable_answer_becomes_uncertain():
    obj, ok = normalize(extract_json("I think it is wrong-way"), "counter_flow", "high")
    assert not ok and obj["verdict"] == "uncertain"


def test_confidence_from_logprobs():
    toks = ['{"', 'verdict', '":', ' "', 'yes', '",']
    content = [{"token": t, "logprob": 0.0} for t in toks]
    content[4]["top_logprobs"] = [{"token": "yes", "logprob": np.log(0.8)},
                                  {"token": "no", "logprob": np.log(0.2)}]
    assert verdict_confidence_from_logprobs(content) == pytest.approx(0.8)
