# AERO-MOT 🚁 AGPL-3.0 License
# Copyright (C) 2026 AERO-MOT Authors

"""Long-Term Re-Identification Gallery for Inactive Trajectory Revival."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional
import numpy as np

from .base import TrackState


class ReIDGallery:
    """Manages long-term appearance memory for reviving tracks after occlusion."""

    def __init__(self, frame_rate: int = 30, max_capacity: int = 256, enabled: bool = True, single_class: bool = False):
        self.frame_rate = max(1, int(frame_rate))
        self.max_capacity = max_capacity
        self.max_age = (2 * self.frame_rate) if single_class else (10 * self.frame_rate)
        self.enabled = enabled
        self.single_class = bool(single_class)
        self.gallery: dict[int, list] = defaultdict(list)

    def reset(self) -> None:
        """Clear all stored gallery features between sequences."""
        self.gallery = defaultdict(list)

    def retire(self, cls_id: int, track: Any, frame_id: int) -> None:
        """Retire an inactive mature track into the appearance gallery."""
        track.mark_removed()
        if not self.enabled:
            return
        if getattr(track, "feat_n", 0.0) < 4.0:
            return

        feature = getattr(track, "smooth_feat", None)
        if feature is None or feature.size <= 1:
            return

        track._gallery_frame = frame_id
        self.gallery[cls_id].append(track)

    @staticmethod
    def _compute_background_similarity(
        cls_id: int,
        det_feature: np.ndarray,
        tracked_tracks_dict: dict,
    ) -> float:
        """Estimate median background appearance similarity against active tracks."""
        active_features = []
        for track in tracked_tracks_dict.get(cls_id, []):
            feat = getattr(track, "smooth_feat", None)
            if feat is not None and feat.size == det_feature.size:
                active_features.append(np.asarray(feat, dtype=np.float64))

        if not active_features:
            return 0.0

        stacked_features = np.stack(active_features)
        return float(np.median(stacked_features @ det_feature))

    def revive(
        self,
        cls_id: int,
        det: Any,
        frame_id: int,
        tracked_tracks_dict: dict,
        kalman_filter: Any,
        lost_tracks_dict: Optional[dict] = None,
    ) -> Optional[Any]:
        """Revive an inactive or long-lost track if appearance and scale distinctly match."""
        if not self.enabled:
            return None

        det_feature = getattr(det, "curr_feat", None)
        if det_feature is None or det_feature.size <= 1:
            return None

        feat_vector = np.asarray(det_feature, dtype=np.float64)
        feat_norm = np.linalg.norm(feat_vector)
        if feat_norm <= 1e-9:
            return None
        feat_vector = feat_vector / feat_norm

        det_xy = np.asarray(det.tlwh[:2] + 0.5 * det.tlwh[2:], dtype=np.float64)
        det_wh = np.maximum(np.asarray(det.tlwh[2:], dtype=np.float64), 2.0)

        gallery_tracks = self.gallery.get(cls_id, [])
        lost_tracks = lost_tracks_dict.get(cls_id, []) if lost_tracks_dict is not None else []

        candidates = []
        cand_sources = []
        cand_feats = []

        for t in lost_tracks:
            dt = max(1, int(getattr(t, "time_since_update", 1)))
            if dt < 2 or getattr(t, "feat_n", 0.0) < 4.0:
                continue
            if self.single_class and dt > 12:
                continue
            sf = getattr(t, "smooth_feat", None)
            if sf is None or sf.size != feat_vector.size:
                continue
            t_xy = np.asarray(t.curr_tlwh[:2] + 0.5 * t.curr_tlwh[2:], dtype=np.float64)
            t_wh = np.maximum(np.asarray(t.curr_tlwh[2:], dtype=np.float64), 2.0)
            dist = float(np.linalg.norm(det_xy - t_xy))
            max_dist = min(25.0 + 6.0 * dt, 90.0) if self.single_class else min(35.0 + 10.0 * dt, 220.0)
            min_wh_r = 0.72 if self.single_class else 0.60
            w_r = min(det_wh[0], t_wh[0]) / max(det_wh[0], t_wh[0])
            h_r = min(det_wh[1], t_wh[1]) / max(det_wh[1], t_wh[1])
            if dist <= max_dist and w_r >= min_wh_r and h_r >= min_wh_r:
                candidates.append(t)
                cand_sources.append("lost")
                cand_feats.append(np.asarray(sf, dtype=np.float64))

        for t in ([] if self.single_class else gallery_tracks):
            sf = getattr(t, "smooth_feat", None)
            if sf is None or sf.size != feat_vector.size:
                continue
            t_xy = np.asarray(t.curr_tlwh[:2] + 0.5 * t.curr_tlwh[2:], dtype=np.float64)
            t_wh = np.maximum(np.asarray(t.curr_tlwh[2:], dtype=np.float64), 2.0)
            dist = float(np.linalg.norm(det_xy - t_xy))
            dt = max(1, int(frame_id - getattr(t, "frame_id", frame_id)))
            max_dist = min(60.0 + 6.0 * dt, 320.0)
            w_r = min(det_wh[0], t_wh[0]) / max(det_wh[0], t_wh[0])
            h_r = min(det_wh[1], t_wh[1]) / max(det_wh[1], t_wh[1])
            if dist <= max_dist and w_r >= 0.55 and h_r >= 0.55:
                candidates.append(t)
                cand_sources.append("gallery")
                cand_feats.append(np.asarray(sf, dtype=np.float64))

        if not cand_feats:
            return None

        cosine_scores = np.stack(cand_feats) @ feat_vector
        sorted_indices = np.argsort(cosine_scores)[::-1]

        best_idx = int(sorted_indices[0])
        best_cosine = float(cosine_scores[best_idx])
        second_cosine = float(cosine_scores[sorted_indices[1]]) if cosine_scores.size > 1 else 0.0

        bg_similarity = self._compute_background_similarity(cls_id, feat_vector, tracked_tracks_dict)

        # Distinctiveness gating against background clutter and nearest competitor
        min_cos = max(0.84, bg_similarity + 0.28) if self.single_class else max(0.70, bg_similarity + 0.25)
        if best_cosine < min_cos:
            return None
        min_margin = 0.12 if self.single_class else 0.08
        if cosine_scores.size > 1 and (best_cosine - second_cosine) < min_margin:
            return None

        revived_track = candidates[best_idx]
        src = cand_sources[best_idx]

        # Re-initialize track state with fresh detection kinematics
        from .track import FalconTrack
        revived_track.kalman_filter = kalman_filter
        revived_track.mean, revived_track.covariance = kalman_filter.initiate(
            FalconTrack.tlwh_to_xywh(np.asarray(det.tlwh, dtype=np.float32))
        )
        revived_track._ingest_feature(det.curr_feat)
        revived_track.update_template(getattr(det, "template", None))
        revived_track.curr_tlwh = np.asarray(det.tlwh, dtype=np.float32)
        revived_track.state = TrackState.Tracked
        revived_track.is_activated = True
        revived_track.score = det.score
        revived_track.frame_id = frame_id
        revived_track.track_len = 0
        revived_track.miss_streak = 0
        revived_track.recovered_streak = 0
        revived_track._is_recovered_this_frame = False
        revived_track.time_since_update = 0
        revived_track.hit_window.append(1)

        if hasattr(revived_track, "cast_vote"):
            revived_track.cast_vote(getattr(det, "cls_id", None), getattr(det, "score", 1.0))

        revived_track.invalidate()
        if src == "gallery" and revived_track in gallery_tracks:
            gallery_tracks.remove(revived_track)
        elif src == "lost" and lost_tracks_dict is not None and revived_track in lost_tracks_dict.get(cls_id, []):
            lost_tracks_dict[cls_id].remove(revived_track)

        return revived_track

    def prune(self, frame_id: int) -> None:
        """Prune outdated gallery entries exceeding the retention horizon or capacity budget."""
        if not self.enabled:
            return

        for cls_id in list(self.gallery.keys()):
            current_gallery = [
                track for track in self.gallery[cls_id]
                if (frame_id - getattr(track, "_gallery_frame", frame_id)) <= self.max_age
            ]
            if len(current_gallery) > self.max_capacity:
                current_gallery.sort(
                    key=lambda t: getattr(t, "_gallery_frame", 0),
                    reverse=True,
                )
                current_gallery = current_gallery[:self.max_capacity]

            self.gallery[cls_id] = current_gallery
