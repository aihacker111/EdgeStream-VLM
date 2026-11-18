"""Video anomaly detection: metrics, loaders, normality models, gate VOI, end-to-end scoring."""
import json
from pathlib import Path

import numpy as np
import pytest

from edgestream.gate.gate import DensityGate, GateConfig, value_of_information
from edgestream.types import Candidate
from edgestream.vad import FrameScorer, ScorerConfig, average_precision, eer, roc_auc, vad_report
from edgestream.vad import datasets as D
from edgestream.vad.normality import ClassLocationMap, KinematicBank, StabilizedFrame
from edgestream.vad.pipeline import VADPipeline, load_vad_config
from edgestream.vlm import load_cards
from edgestream.vlm.prompt_vad import VADPromptBuilder, canonical_category, load_hypotheses

from .synthetic_vad import EVENTS, FakePerceptorVAD, scenario, write_video

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "event_cards" / "uav_surveillance"


# ---------------------------------------------------------------- metrics
def test_metrics_match_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = rng.integers(0, 2, 500)
        s = np.round(rng.random(500) + 0.6 * y, 2)          # rounding creates ties
        assert roc_auc(s, y) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-9)
        assert average_precision(s, y) == pytest.approx(sk.average_precision_score(y, s), abs=1e-9)


def test_vad_report_protocols():
    a = (np.array([0.1, 0.2, 0.9, 0.8]), np.array([0, 0, 1, 1]))
    b = (np.array([0.3, 0.4, 0.5]), np.array([0, 0, 0]))       # normal video: no per-video AUC
    r = vad_report({"a": a, "b": b})
    assert r["macro_auc"] == pytest.approx(1.0)
    assert np.isnan(r["per_video_auc"]["b"])
    assert r["auc_abnormal_videos"] == pytest.approx(1.0)
    assert r["micro_auc"] == pytest.approx(roc_auc(np.r_[a[0], b[0]], np.r_[a[1], b[1]]))
    assert eer(np.array([0.0, 1.0]), np.array([0, 1])) == pytest.approx(0.0)


# ---------------------------------------------------------------- loaders
def _touch_video(path: Path, n: int = 20):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(n):
        vw.write(np.zeros((48, 64, 3), np.uint8))
    vw.release()


def test_ucf_and_xd_loaders(tmp_path):
    vids = tmp_path / "v"
    vids.mkdir()
    _touch_video(vids / "Arrest001_x264.mp4")
    _touch_video(vids / "Normal_Videos_003_x264.mp4")
    (tmp_path / "ucf.txt").write_text("Arrest001_x264.mp4  Arrest  5  9  -1  -1\n"
                                      "Normal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1\n")
    ucf = {v.name: v for v in D.ucf_crime(str(vids), str(tmp_path / "ucf.txt"))}
    assert ucf["Arrest001_x264"].labels[5:10].all() and ucf["Arrest001_x264"].labels.sum() == 5
    assert ucf["Normal_Videos_003_x264"].labels.sum() == 0

    _touch_video(vids / "A.Beautiful.Mind__#0-17-50_0-18-50_label_B1-0-0.mp4")
    (tmp_path / "xd.txt").write_text("A.Beautiful.Mind__#0-17-50_0-18-50_label_B1-0-0 2 4 10 12\n")
    xd = {v.name: v for v in D.xd_violence(str(vids), str(tmp_path / "xd.txt"))}
    y = xd["A.Beautiful.Mind__#0-17-50_0-18-50_label_B1-0-0"].labels
    assert y.sum() == 6 and y[2] and y[12] and not y[13]


def test_generic_and_a2seek_loaders(tmp_path):
    vids, labs = tmp_path / "videos", tmp_path / "labels"
    vids.mkdir()
    labs.mkdir()
    _touch_video(vids / "clip_01.mp4", n=10)
    np.save(labs / "clip_01.npy", np.r_[np.zeros(6), np.ones(4)].astype(np.uint8))
    g = D.generic_frame_labels(str(vids), str(labs))
    assert len(g) == 1 and g[0].labels.tolist() == [0] * 6 + [1] * 4

    rows = [{"g_id": "video_group_001", "v_id": "video_1", "f_id": 12, "code": "E05S03",
             "anomaly_name": "Fighting", "objects": {"ref": ["Fighting"], "bbox": [[1, 2, 30, 40]]}},
            {"g_id": "video_group_001", "v_id": "video_1", "f_id": 3, "code": "E05S03",
             "anomaly_name": "Normal", "objects": {}}]
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    a2 = D.a2seek(str(p), str(tmp_path / "{g_id}" / "{v_id}.mp4"))
    assert len(a2) == 1 and [q["frame"] for q in a2[0].probes] == [3, 12]
    assert a2[0].probes[1]["sample_id"] == "video_group_001/video_1/00012"
    assert a2[0].probes[1]["gt_box"] == [1, 2, 30, 40] and a2[0].probes[1]["scene"] == "03"


# ---------------------------------------------------------------- normality models
def test_location_map_flags_people_where_nobody_goes():
    m = ClassLocationMap(cell_frac=0.1, alpha=25.0)
    st = StabilizedFrame()
    st.image_wh = (100, 100)
    for s in range(120):                                       # 2 min: people only on the path (y≈15)
        m.observe_view(float(s), st)
        for gid, x in enumerate(range(5, 95, 10)):
            m.observe(float(s), gid, "human", m.key(x, 15, 100))
    on_path, _ = m.surprise(99, "human", m.key(45, 15, 100))
    lawn, exposure = m.surprise(99, "human", m.key(45, 75, 100))
    assert on_path < 0 and lawn > 1.5 and exposure > 100


def test_stabilized_frame_integrates_drift():
    st = StabilizedFrame(reset_frac=10)
    A = np.array([[1, 0, -2.0], [0, 1, 0.0]])                 # camera pans right: scene moves left 2 px/frame
    for _ in range(10):
        st.update(A, (640, 360))
    x, y = st.to_ref(300, 100)                                 # a ground point now at 300 was at 320
    assert x == pytest.approx(320) and y == pytest.approx(100)


def test_knn_bank_is_scale_free_and_self_gated():
    bank = KinematicBank(k=3, min_items=20)
    rng = np.random.default_rng(1)
    walk = lambda: np.array([np.log1p(1.4), 0.1, 0.2, 0.9, 0.0, 0.05, 0.2]) + rng.normal(0, 0.02, 7)  # noqa: E731
    for _ in range(200):
        bank.insert("human", walk(), bank.score("human", walk()))
    normal = bank.score("human", walk())
    run = walk()
    run[0] = np.log1p(5.0)
    assert normal < 1.5 and bank.score("human", run) > 3.0
    n = len(bank)
    bank.insert("human", run, bank.score("human", run))       # outliers are not learned as normal
    assert len(bank) == n


# ---------------------------------------------------------------- gate / scorer / prompt
def _cand(i, p, vlm_p=None, sev=0.6):
    c = Candidate(i, "running", [i], 0.0, 5.0, 0.8, 0.8, sev)
    c.priority, c.vlm_p = p, vlm_p
    return c


def test_voi_prefers_unverified_and_skips_confident_resend():
    assert value_of_information(_cand(1, 0.8)) > value_of_information(_cand(2, 0.8, vlm_p=0.97))
    g = DensityGate(GateConfig(calls_per_min=60, bucket_capacity=10, voi=True, resend_after_s=1.0))
    g.priority = lambda c: c.priority
    sure, unsure = _cand(1, 0.9, vlm_p=0.98), _cand(2, 0.9, vlm_p=0.55)
    for c in (sure, unsure):
        c.times_sent, c.last_sent_t, c.last_sent_priority = 1, 0.0, 0.9
    sel, _ = g.step(10.0, [sure, unsure], 1, 1)
    assert [c.cand_id for c in sel] == [2]


def test_scorer_is_causal_about_vlm_latency():
    sc = FrameScorer(ScorerConfig(w_dense=0.0, tau_up_s=1e-6, tau_down_s=1e-6))
    c = _cand(1, 0.6)
    sc.update_candidates(0.0, [c])
    sc.add_verdict(1.0, 1, 0.95, "running")                   # answer arrives at t=1.0
    before, after = sc.frame(0.5), sc.frame(1.0)
    assert before == pytest.approx(sc.cfg.w_unverified * 0.6, abs=1e-3) and after > 0.9


def test_vad_prompt_is_open_set_with_stable_prefix():
    cards = load_cards(CARDS)
    b = VADPromptBuilder(cards, load_hypotheses(CARDS))
    assert "trespassing_on_lawn" in b.system and "normal" in b.card_ids
    c = Candidate(1, "close_interaction", [1, 2], 0.0, 3.0, 0.8, 0.7, 0.8)
    ctx = {"stats_text": "x", "tracks": [], "scene_log": []}
    req = b.build(c, ctx, np.zeros((360, 640, 3), np.uint8), [(10, 10, 20, 40), (30, 10, 20, 40)])
    assert req.meta["hypotheses"][0] == "fighting" and req.system == b.system
    assert canonical_category("Trespassing on Lawn", cards) == "trespassing_on_lawn"
    assert canonical_category("none", cards) == "normal"
    assert canonical_category("Fire", cards) == "other_anomaly"


# ---------------------------------------------------------------- end to end
def _run(tmp_path, backend, anomalies=True, budget=6.0):
    frames, labels = scenario(anomalies=anomalies)
    video = str(tmp_path / "syn.mp4")
    write_video(frames, video)
    cfg = load_vad_config(ROOT / "configs" / "vad_cctv.yaml")
    cfg["vlm"]["backend"] = backend
    cfg["gate"]["calls_per_min"] = budget
    if budget == 0:
        cfg["gate"]["bucket_capacity"] = 0
    r = VADPipeline(cfg, perceptor=FakePerceptorVAD(frames)).run(video, str(tmp_path / backend), labels=labels)
    return r, labels


@pytest.mark.parametrize("backend", ["simulated", "oracle"])
def test_synthetic_vad_scores_detect_all_three_anomalies(tmp_path, backend):
    r, labels = _run(tmp_path, backend)
    on, off = r["scores_online"], r["scores_offline"]
    assert len(on) == len(labels)
    assert roc_auc(off, labels) > 0.9 and roc_auc(on, labels) > 0.75
    t = np.arange(len(labels)) / 10.0
    for name, (a, b) in EVENTS.items():                         # every event is seen (offline)
        assert off[(t >= a) & (t < b)].max() > 0.6, name
    assert r["budget"]["vlm_calls"] <= 3 + 6 * r["duration_s"] / 60 + 1
    assert (tmp_path / backend / "scores_online.npy").exists()


def test_vlm_verification_beats_cheap_tier_alone(tmp_path):
    r0, y = _run(tmp_path, "oracle", budget=0)
    r1, _ = _run(tmp_path, "oracle", budget=6)
    assert r0["budget"]["vlm_calls"] == 0
    assert average_precision(r1["scores_offline"], y) > average_precision(r0["scores_offline"], y)


def test_normal_scene_stays_quiet(tmp_path):
    r, _ = _run(tmp_path, "simulated", anomalies=False)
    assert r["events"] == [] and float(np.quantile(r["scores_online"], 0.95)) < 0.5


def test_a2seek_probe_answers_use_benchmark_format(tmp_path):
    frames, labels = scenario()
    video = str(tmp_path / "syn.mp4")
    write_video(frames, video)
    cfg = load_vad_config(ROOT / "configs" / "vad_cctv.yaml")
    cfg["vlm"]["backend"] = "oracle"
    probes = [{"sample_id": "g/v/00050", "frame": 50, "gt_category": "Normal", "image_wh": (1280, 720)},
              {"sample_id": "g/v/00235", "frame": 235, "gt_category": "Unable to Stand", "image_wh": (1280, 720)}]
    r = VADPipeline(cfg, perceptor=FakePerceptorVAD(frames)).run(video, None, labels=labels, probes=probes)
    a = {x["sample_id"]: x for x in r["probe_answers"]}
    assert a["g/v/00050"]["prediction"] == "<answer>['Normal']; [[0, 0, 0, 0]]</answer>"
    pred = a["g/v/00235"]
    assert pred["category"] != "Normal"
    box = eval(pred["prediction"].split(";")[1].replace("</answer>", ""))[0]
    assert box[0] > 400 and box[2] <= 1280                      # scaled to the probe image size


# ---------------------------------------------------------------- dataset preparation (scripts/datasets/fetch.py)
def _fetch_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fetch", ROOT / "scripts" / "datasets" / "fetch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a2seek_clip_keys_for_both_layouts():
    f = _fetch_module()
    assert f._clip_key("videos/video_group_001/video_1.mp4") == "video_group_001/video_1"
    assert f._clip_key("video_group_012_@_video_10.MP4") == "video_group_012/video_10"
    assert f._clip_key("README.md") is None


def test_frame_labels_pairs_drone_anomaly_and_sibling_layouts(tmp_path):
    import cv2
    f = _fetch_module()
    seq = tmp_path / "src" / "Highway" / "sequence1"
    for k in (1, 2):                                   # Drone-Anomaly: test/01 + test/1.npy
        d = seq / "test" / f"{k:02d}"
        d.mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(str(d / f"{i}.jpg"), np.zeros((8, 8, 3), np.uint8))
        np.save(seq / "test" / f"{k}.npy", np.array([0, 1, k == 2], np.uint8))
    (seq / "train" / "01").mkdir(parents=True)
    cv2.imwrite(str(seq / "train" / "01" / "0.jpg"), np.zeros((8, 8, 3), np.uint8))
    uit = tmp_path / "src" / "UIT" / "test"                 # UIT-ADrone style: videos/ + labels/ siblings
    (uit / "videos").mkdir(parents=True)
    (uit / "labels").mkdir(parents=True)
    _touch_video(uit / "videos" / "cam_7.mp4", n=4)
    np.save(uit / "labels" / "cam_7.npy", np.array([0, 0, 1, 1], np.uint8))
    out = tmp_path / "out"
    f.main(["frame-labels", "--src", str(tmp_path / "src"), "--out", str(out)])
    vids = D.generic_frame_labels(str(out / "videos"), str(out / "labels"))
    names = sorted(v.name for v in vids)
    assert names == ["Highway__sequence1__test__01", "Highway__sequence1__test__02", "UIT__test__videos__cam_7"]
    by = {v.name: v for v in vids}
    assert by["Highway__sequence1__test__02"].labels.tolist() == [0, 1, 1]
    assert by["UIT__test__videos__cam_7"].labels.tolist() == [0, 0, 1, 1]
    assert len(list((out / "train").iterdir())) == 1


def test_frame_folders_play_in_natural_order(tmp_path):
    import cv2

    from edgestream.harness.stream import open_stream
    for i in (0, 1, 2, 10):
        cv2.imwrite(str(tmp_path / f"{i}.jpg"), np.full((8, 8, 3), i * 20, np.uint8))
    seen = [int(p.image[0, 0, 0]) // 20 for p in open_stream(str(tmp_path), fps=10)]
    assert seen == [0, 1, 2, 10]


def test_an_empty_bank_passed_in_is_the_one_that_learns(tmp_path):
    frames, labels = scenario(anomalies=False)
    video = str(tmp_path / "syn.mp4")
    write_video(frames, video)
    cfg = load_vad_config(ROOT / "configs" / "vad_cctv.yaml")
    bank = KinematicBank()
    VADPipeline(cfg, perceptor=FakePerceptorVAD(frames)).run(video, None, bank=bank)
    assert len(bank) > 50
