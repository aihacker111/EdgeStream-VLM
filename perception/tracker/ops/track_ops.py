# AERO-MOT 🚁 AGPL-3.0 License
# Copyright (C) 2026 AERO-MOT Authors

"""Set algebra and spatial de-duplication over lists of tracks.

Every function here is a pure operation on lists/dicts of tracks — no tracker
state is touched — so the orchestrator in ``tracker.py`` stays focused on the
per-frame data flow instead of book-keeping.

Extracted verbatim from the original ``multitracker.py``:
    join_tracks / sub_tracks            — union / difference by track id
    remove_duplicate_tracks             — drop near-identical tracks across two lists
    suppress_same_class_overlaps        — keep the older of two overlapping same-class tracks
    remove_cross_class_duplicates       — drop the shorter of two overlapping tracks of different classes
"""

import numpy as np

from . import association_ops as matching
from .geometry import iou_1_to_n


def join_tracks(t_list_a, t_list_b):
    """Union of two track lists, keyed by ``track_id`` (entries from ``a`` win)."""
    if not t_list_b:
        return list(t_list_a)
    exists, res = set(), []
    for t in t_list_a:
        exists.add(t.track_id)
        res.append(t)
    for t in t_list_b:
        if t.track_id not in exists:
            exists.add(t.track_id)
            res.append(t)
    return res


def sub_tracks(t_list_a, t_list_b):
    """Difference ``a - b`` keyed by ``track_id``."""
    tracks = {t.track_id: t for t in t_list_a}
    for t in t_list_b:
        tracks.pop(t.track_id, None)
    return list(tracks.values())


def remove_duplicate_tracks(tracks_a, tracks_b):
    """Resolve near-identical (IoU-dist < 0.15) tracks across two lists.

    The younger of each colliding pair is dropped; ties keep both.
    """
    if not tracks_a or not tracks_b:
        return tracks_a, tracks_b
    p_dist = matching.iou_distance(tracks_a, tracks_b)
    ii, jj = np.nonzero(p_dist < 0.15)
    if ii.size == 0:
        return tracks_a, tracks_b
    ta = np.fromiter((t.frame_id - t.start_frame for t in tracks_a), np.int32, len(tracks_a))
    tb = np.fromiter((t.frame_id - t.start_frame for t in tracks_b), np.int32, len(tracks_b))
    a_older = ta[ii] > tb[jj]
    dup_b = set(jj[a_older].tolist())
    dup_a = set(ii[~a_older].tolist())
    return ([t for i, t in enumerate(tracks_a) if i not in dup_a],
            [t for i, t in enumerate(tracks_b) if i not in dup_b])


def suppress_same_class_overlaps(tracks, iou_thresh=0.8, min_age_gap=5):
    """Within one class, drop a young track heavily overlapping a much older one."""
    n = len(tracks)
    if n < 2:
        return tracks
    tlbrs = np.stack([t.tlbr for t in tracks])
    drop = set()
    order = sorted(range(n), key=lambda i: tracks[i].track_len, reverse=True)
    for oi in range(n):
        a = order[oi]
        if a in drop:
            continue
        ious = iou_1_to_n(tlbrs[a], tlbrs)
        for b in range(n):
            if b == a or b in drop:
                continue
            if (ious[b] > iou_thresh and tracks[a].track_len - tracks[b].track_len >= min_age_gap):
                drop.add(b)
    return [t for i, t in enumerate(tracks) if i not in drop]


def remove_cross_class_duplicates(tracked_dict, num_classes, iou_thresh=0.75):
    """Across classes, drop the shorter of two boxes that overlap by > ``iou_thresh``.

    Mutates ``tracked_dict`` in place (same behaviour as the original method).
    """
    entries = []
    for cls_id in range(num_classes):
        for idx, t in enumerate(tracked_dict[cls_id]):
            entries.append((cls_id, idx, t))
    if len(entries) < 2:
        return
    tlbrs = np.stack([e[2].tlbr for e in entries])
    drop = set()
    for a in range(len(entries)):
        if a in drop:
            continue
        ious = iou_1_to_n(tlbrs[a], tlbrs)
        for b in range(len(entries)):
            if b == a or b in drop or entries[a][0] == entries[b][0]:
                continue
            if ious[b] > iou_thresh:
                ta = entries[a][2].track_len
                tb = entries[b][2].track_len
                loser = b if ta >= tb else a
                drop.add(loser)
                if loser == a:
                    break
    if not drop:
        return
    drop_keys = set((entries[i][0], id(entries[i][2])) for i in drop)
    for cls_id in range(num_classes):
        tracked_dict[cls_id] = [t for t in tracked_dict[cls_id]
                                if (cls_id, id(t)) not in drop_keys]
