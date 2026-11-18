"""Track ids must be independent per tracker instance (one tracker per camera)."""
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from perception.config import PAPER_TRACKER_DEFAULTS
from perception.tracker import FalconTrack, FalconTracker


def _tracker():
    opt = SimpleNamespace(**PAPER_TRACKER_DEFAULTS, num_classes=1, conf_thres=0.4,
                          use_appearance_motion=0, use_reid_gallery=0)
    return FalconTracker(opt, frame_rate=30)


def _dets(boxes):
    out = defaultdict(list)
    for b in boxes:
        out[0].append(FalconTrack(np.asarray(b, np.float32), 0.9, np.zeros(1, np.float32), 0))
    return out


def _step(trk, boxes):
    targets = trk.update(_dets(boxes), 1080, 1920)
    return sorted(t.track_id for ts in targets.values() for t in ts)


def _boxes(n, frame, x0):
    return [[x0 + 120 * i + 2 * frame, 300, 40, 90] for i in range(n)]


def test_second_camera_does_not_reset_or_share_first_cameras_ids():
    cam_a, cam_b = _tracker(), _tracker()
    for f in range(5):
        ids_a = _step(cam_a, _boxes(3, f, 100))
    assert ids_a == [1, 2, 3]

    # camera B starts: with a class-level counter this reset camera A's counter
    for f in range(5):
        ids_b = _step(cam_b, _boxes(2, f, 900))
    assert ids_b == [1, 2]                      # B numbers its own objects from 1

    # a new object appears in camera A: it must get id 4, not a duplicate of 1..3
    for f in range(5, 10):
        ids_a = _step(cam_a, _boxes(3, f, 100) + [[1500, 700, 40, 90]])
    assert ids_a == [1, 2, 3, 4]
    assert len(set(ids_a)) == len(ids_a)


def test_reset_restarts_numbering():
    trk = _tracker()
    for f in range(5):
        _step(trk, _boxes(2, f, 100))
    trk.reset()
    for f in range(5):
        ids = _step(trk, _boxes(1, f, 500))
    assert ids == [1]
