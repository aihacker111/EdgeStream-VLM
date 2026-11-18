# encoding=utf-8
# AERO-MOT 🚁 AGPL-3.0 License
# Copyright (C) 2026 AERO-MOT Authors

"""Base classes for multi-class tracking.

Only the multi-class base (`MCBaseTrack`) is kept — it maintains a *per-class*
track-id counter so that, e.g., pedestrian #1 and car #1 are distinct objects.
The original single-object `BaseTrack` was unused and has been removed.
"""

from collections import defaultdict

import numpy as np


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class IdCounter:
    """Per-class id counter owned by ONE tracker instance (one camera stream).

    [port] AERO-MOT kept this counter on the class (``MCBaseTrack._count_dict``),
    so every tracker in the process shared it and a tracker starting its first
    frame reset the ids of all the others (duplicate ids in a running stream).
    """

    def __init__(self):
        self._last = defaultdict(int)

    def next(self, cls_id):
        self._last[cls_id] += 1
        return self._last[cls_id]

    def reset(self):
        self._last.clear()


_DEFAULT_COUNTER = IdCounter()   # only for tracks used outside a tracker (e.g. unit tests)


class MCBaseTrack(object):
    """Multi-class base track; ids are drawn from the owning tracker's IdCounter."""

    id_counter = None                # injected by FalconTracker before activation

    track_id = 0
    is_activated = False
    state = TrackState.New

    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera placeholder
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    def next_id(self, cls_id):
        counter = self.id_counter if self.id_counter is not None else _DEFAULT_COUNTER
        return counter.next(cls_id)

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
