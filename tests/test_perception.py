from pathlib import Path

import numpy as np
import pytest

from perception.config import PerceptionConfig
from perception.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess
from perception.tracker.ops import association_ops

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "perception.yaml"
WEIGHTS = ROOT / "weights" / "falcon_detector_1.pth"


def test_preprocess_matches_training_pipeline():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    chw, tf = preprocess(frame, 320, 192)
    assert chw.shape == (3, 192, 320) and chw.dtype == np.float32
    import cv2
    ref = cv2.resize(frame, (320, 192), interpolation=cv2.INTER_AREA)[:, :, ::-1] / 255.0
    ref = ((ref - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    np.testing.assert_allclose(chw, ref, atol=1e-5)
    assert tf.ratio_x == pytest.approx(0.5) and tf.ratio_y == pytest.approx(192 / 360)


def test_numpy_iou_fallback_matches_cython_convention():
    a = np.array([[0, 0, 9, 9], [5, 5, 14, 14]], dtype=np.float64)
    b = np.array([[0, 0, 9, 9], [20, 20, 30, 30]], dtype=np.float64)
    iou = association_ops.bbox_ious(a, b)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == 0.0
    assert iou[1, 0] == pytest.approx(25 / 175)   # +1 pixel convention: 5*5 / (100+100-25)


def test_config_loads_and_resolves_weights():
    cfg = PerceptionConfig.from_yaml(CFG)
    assert cfg.detector.num_classes == 10
    assert Path(cfg.detector.weights).name == "falcon_yolo_lqcv2_baseline_jde_mot.pth"
    assert tuple(cfg.detector.input_wh) == (960, 544)
    opt = cfg.tracker_opt(appearance=False)
    assert opt.use_appearance_motion == 0 and opt.minority_large_classes == []


@pytest.mark.skipif(not WEIGHTS.exists(), reason="checkpoint not available")
def test_detection_only_checkpoint_runs_motion_only():
    import cv2
    from perception import Perceptor

    cfg = PerceptionConfig.from_yaml(CFG)
    cfg.detector.weights = str(WEIGHTS)          # detection-only checkpoint
    cfg.detector.device = "cpu"
    cfg.detector.input_wh = (480, 288)
    p = Perceptor(cfg)
    assert p.appearance is False            # detection-only checkpoint
    rng = np.random.default_rng(0)
    bg = rng.integers(90, 110, (540, 960, 3), dtype=np.uint8)
    for i in range(5):
        frame = bg.copy()
        cv2.rectangle(frame, (100 + 10 * i, 200), (140 + 10 * i, 290), (30, 80, 200), -1)
        out = p.process(frame, timestamp=i / 30)
        assert out.frame_idx == i and out.image_hw == (540, 960)
        assert set(out.latency_ms) == {"preprocess", "detector", "post+track", "total"}
        for t in out.tracks:
            assert t.tlwh.shape == (4,) and t.embedding is None and t.kappa is None
            assert 0.0 <= t.quality <= 1.0
    p.reset()
    assert p.process(bg).frame_idx == 0


JDE_WEIGHTS = ROOT / "weights" / "falcon_yolo_lqcv2_baseline_jde_mot.pth"


@pytest.mark.skipif(not JDE_WEIGHTS.exists(), reason="JDE checkpoint not available")
def test_perceptor_with_reid_enables_appearance():
    import cv2
    from perception import Perceptor

    cfg = PerceptionConfig.from_yaml(CFG)
    cfg.detector.weights = str(JDE_WEIGHTS)
    cfg.detector.device = "cpu"
    p = Perceptor(cfg)
    assert p.appearance is True and p.detector.use_reid
    rng = np.random.default_rng(0)
    bg = rng.integers(90, 110, (544, 960, 3), dtype=np.uint8)
    for i in range(4):
        frame = bg.copy()
        cv2.rectangle(frame, (300 + 8 * i, 200), (340 + 8 * i, 300), (30, 80, 200), -1)
        out = p.process(frame, timestamp=i / 30)
        for t in out.tracks:
            assert t.embedding is not None and t.embedding.shape == (128,)
            assert abs(float(np.linalg.norm(t.embedding)) - 1.0) < 1e-3
            assert t.kappa is not None and t.kappa >= 0.0
