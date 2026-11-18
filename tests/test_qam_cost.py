"""The vectorised QAM candidate cost must match AERO-MOT's original loop version."""
from types import SimpleNamespace

import numpy as np
import pytest

from perception.config import PAPER_TRACKER_DEFAULTS
from perception.tracker import FalconTrack, FalconTracker

from ._reference_qam_cost import build_qam_cost_reference


def _objects(rng, n, dim, with_feat=True, spread=400.0):
    objs = []
    for _ in range(n):
        xy = rng.uniform(0, spread, 2)
        wh = rng.uniform(4, 60, 2)
        feat = rng.normal(size=dim) if with_feat else np.zeros(1)
        t = FalconTrack(np.r_[xy, wh], float(rng.uniform(0.1, 0.9)), feat, 0)
        objs.append(t)
    return objs


def _tracker(appearance=True):
    opt = SimpleNamespace(**PAPER_TRACKER_DEFAULTS, num_classes=1, conf_thres=0.4,
                          use_appearance_motion=int(appearance), use_reid_gallery=0)
    return FalconTracker(opt, frame_rate=30)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("shift_w", [0.0, 0.3])
@pytest.mark.parametrize("global_search", [False, True])
def test_vectorised_cost_matches_reference(seed, shift_w, global_search):
    rng = np.random.default_rng(seed)
    dim = 16
    tracks = _objects(rng, int(rng.integers(1, 25)), dim, spread=150.0)
    dets = _objects(rng, int(rng.integers(1, 30)), dim, spread=150.0)
    # a few tracks without appearance memory, a few detections without features
    for t in tracks[::5]:
        t.smooth_feat = None
    for d in dets[::7]:
        d.curr_feat = np.zeros(1, dtype=np.float32)
    kw = dict(global_search=global_search, r_min_val=16, q_gate=0.25,
              shift_w=shift_w, margin_val=0.25)
    trk = _tracker()
    ours = trk._qam_candidate_cost(tracks, dets, **kw)
    ref = build_qam_cost_reference(tracks, dets, use_appearance_motion=True, **kw)
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_motion_only_returns_iou_distance():
    rng = np.random.default_rng(0)
    tracks, dets = _objects(rng, 5, 1, False), _objects(rng, 6, 1, False)
    trk = _tracker(appearance=False)
    ours = trk._qam_candidate_cost(tracks, dets, global_search=False, r_min_val=16,
                                   q_gate=0.25, shift_w=0.0, margin_val=0.25)
    ref = build_qam_cost_reference(tracks, dets, use_appearance_motion=False, global_search=False,
                                   r_min_val=16, q_gate=0.25, shift_w=0.0, margin_val=0.25)
    np.testing.assert_allclose(ours, ref)
