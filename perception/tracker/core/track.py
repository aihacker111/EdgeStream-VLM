# AERO-MOT 🚁 AGPL-3.0 License
# Copyright (C) 2026 AERO-MOT Authors

"""FalconTrack - the per-object track state for the Falcon-MOT tracker.

One instance represents a single tracked object (or a single detection before it
is associated). It owns:

  * the Kalman state (``mean`` / ``covariance``) in the linear x, y, w, h space;
  * an online **von Mises-Fisher** appearance memory (``smooth_feat`` plus the
    running resultant length ``app_Rbar`` used to estimate the vMF concentration
    ``kappa``); and
  * a short correlation ``template`` used by the query appearance-motion field.

The class carries a per-class id counter through :class:`MCBaseTrack`, so
``pedestrian #1`` and ``car #1`` are different objects.
"""

from collections import deque

import numpy as np

from ..methods import spear
from ..motion.kalman_filter import KalmanFilter
from .base import MCBaseTrack, TrackState


class FalconTrack(MCBaseTrack):
    shared_kalman = KalmanFilter()
    EMB_DIM = 128
    KAPPA_MAX = 300.0
    GLARE_SHIELD_INGEST = True

    def __init__(self, tlwh, score, temp_feat, cls_id, *, buff_size=30):
        self.cls_id = cls_id
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.track_len = 0
        self.smooth_feat = None
        self.curr_feat = None
        self.feat_sum = None
        self.feat_n = 0.0
        self.app_Rbar = 0.0
        self._ingest_feature(temp_feat)

        self.template = None
        self.pred_xy = None
        self.curr_tlwh = np.asarray(tlwh, dtype=np.float32)
        self.hit_window = deque([1], maxlen=30)
        self.birth_cls = cls_id
        self._emit_cls = cls_id
        self.cls_hist = deque([(cls_id, float(score))], maxlen=buff_size)
        self.q = float(score)
        self.conf = 0.0
        self.miss_streak = 0
        self.recovered_streak = 0
        self._is_recovered_this_frame = False
        self.time_since_update = 0
        self._tlwh_c = None
        self._tlbr_c = None

    # cached box accessors
    def invalidate(self):
        self._tlwh_c = None
        self._tlbr_c = None

    @property
    def tlwh(self):
        c = self._tlwh_c
        if c is not None:
            return c
        if self.mean is None:
            c = self._tlwh.copy()
        else:
            # tlwh straight from the x, y, w, h state
            c = self.mean[:4].copy()
            c[:2] -= c[2:] / 2
        self._tlwh_c = c
        return c

    @property
    def tlbr(self):
        c = self._tlbr_c
        if c is None:
            c = self.tlwh.copy()
            c[2:] += c[:2]
            self._tlbr_c = c
        return c

    @property
    def age(self):
        return self.frame_id - self.start_frame

    # appearance memory (online von Mises-Fisher)
    def _ingest_feature(self, feat):
        if feat is None:
            return
        feat = np.asarray(feat, dtype=np.float64)
        if feat.size <= 1:
            if self.curr_feat is None:
                self.curr_feat = feat.astype(np.float32)
            return
        n = np.linalg.norm(feat)
        if n <= 1e-12:
            return
        f = feat / n
        self.curr_feat = f.astype(np.float32)
        if self.feat_sum is None or self.smooth_feat is None:
            self.feat_sum = f.copy()
            self.feat_n = 1.0
            self.smooth_feat = f.astype(np.float32)
        else:
            rho = max(float(np.dot(self.smooth_feat, f)), 0.0)
            if getattr(FalconTrack, "GLARE_SHIELD_INGEST", True) and rho < 0.22:
                # Protect template direction from severe glare/blur outlier while contracting resultant length
                w_in = rho ** 3
                self.feat_sum = 0.92 * self.feat_sum + w_in * f
                self.feat_n = 0.92 * self.feat_n + 1.0
                nrm = np.linalg.norm(self.feat_sum)
                if nrm > 0:
                    self.smooth_feat = (self.feat_sum / nrm).astype(np.float32)
                self.app_Rbar = float(min(nrm / max(self.feat_n + 1.5, 1.0), rho ** 2))
                return
            self.feat_sum = rho * self.feat_sum + f
            self.feat_n = rho * self.feat_n + 1.0
            nrm = np.linalg.norm(self.feat_sum)
            if nrm > 0:
                self.smooth_feat = (self.feat_sum / nrm).astype(np.float32)
        self.app_Rbar = float(np.linalg.norm(self.feat_sum) / max(self.feat_n + 1.5, 1.0))

    def kappa(self, use_dynamic=True, kappa_max=None, fixed_kappa=None):
        if not use_dynamic:
            if fixed_kappa is not None:
                return float(fixed_kappa)
            return 1.0
        if kappa_max is None:
            kappa_max = getattr(FalconTrack, "KAPPA_MAX", 300.0)
        return float(spear.vmf_kappa(self.app_Rbar, FalconTrack.EMB_DIM, use_dynamic=True, kappa_max=kappa_max))

    def update_template(self, feat, alpha=0.9):
        if feat is None:
            return
        feat = np.asarray(feat, dtype=np.float32)
        n = np.linalg.norm(feat)
        if n > 0:
            feat = feat / n
        if self.template is None:
            self.template = feat
        else:
            self.template = alpha * self.template + (1.0 - alpha) * feat
            tn = np.linalg.norm(self.template)
            if tn > 0:
                self.template = self.template / tn

    def _update_conf(self, feat_conf):
        if feat_conf is not None:
            self.conf = float(feat_conf)

    # Kalman prediction (batched)
    @staticmethod
    def multi_predict(tracks, kalman_filter=None):
        if len(tracks) == 0:
            return
        kf = kalman_filter if kalman_filter is not None else FalconTrack.shared_kalman
        multi_mean = np.asarray([t.mean.copy() for t in tracks])
        multi_covariance = np.asarray([t.covariance for t in tracks])
        for i, t in enumerate(tracks):
            if t.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = kf.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            tracks[i].mean = mean
            tracks[i].covariance = cov
            tracks[i].invalidate()
            tracks[i].time_since_update = getattr(tracks[i], "time_since_update", 0) + 1

    def update_class_consensus(self, det_cls, score, log_margin=1.5):
        """Sequential Probability Ratio Test (SPRT) for online category consensus.

        Accumulates temporal evidence across a sliding observation window. The emitted
        category state transitions only when cumulative evidence for an alternative class
        exceeds the incumbent class by a decisive decision boundary (log_margin),
        preventing boundary observation noise and class flickering from inducing ID switches.
        """
        if det_cls is None or getattr(FalconTrack, "disable_tcc_global", False):
            return
        margin = getattr(FalconTrack, "tcc_margin", getattr(self, "tcc_margin", log_margin))
        self.cls_hist.append((int(det_cls), float(score)))
        cls_scores = {}
        for c, s in self.cls_hist:
            cls_scores[c] = cls_scores.get(c, 0.0) + s

        cur = self._emit_cls
        cur_score = cls_scores.get(cur, 0.0)

        best_cls = cur
        best_score = cur_score
        for c, s in cls_scores.items():
            if s > best_score:
                best_cls = c
                best_score = s

        if best_cls != cur and best_score >= cur_score + margin:
            self._emit_cls = best_cls

    # Backward-compatible alias
    cast_vote = update_class_consensus

    # lifecycle transitions
    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.birth_cls = self.cls_id
        self._emit_cls = self.cls_id
        self.cls_hist = deque([(self.cls_id, float(self.score))], maxlen=30)
        self.track_id = self.next_id(self.cls_id)
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))
        self.curr_tlwh = self._tlwh.copy()
        self.track_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0
        self.invalidate()

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update_nsa(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh), new_track.score)
        self._update_conf(getattr(new_track, 'conf', None))
        self._ingest_feature(new_track.curr_feat)
        self.update_template(new_track.template)
        self.curr_tlwh = np.array(new_track.curr_tlwh, dtype=np.float32)
        self.track_len = 0
        self.frame_id = frame_id
        self.score = new_track.score
        self.state = TrackState.Tracked
        self.is_activated = True
        self.miss_streak = 0
        self.recovered_streak = 0
        self._is_recovered_this_frame = False
        self.time_since_update = 0
        self.hit_window.append(1)
        self.cast_vote(getattr(new_track, 'cls_id', None), getattr(new_track, 'score', 1.0))
        if new_id:
            self.track_id = self.next_id(self.cls_id)
        self.invalidate()

    def update(self, new_track, frame_id, update_feature=True):
        self.frame_id = frame_id
        self.track_len += 1
        new_tlwh = np.array(new_track.tlwh, dtype=np.float32)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.curr_tlwh = new_tlwh
        self.miss_streak = 0
        self.recovered_streak = 0
        self._is_recovered_this_frame = False
        self.time_since_update = 0
        self.hit_window.append(1)
        if update_feature:
            self._update_conf(getattr(new_track, 'conf', None))
            self._ingest_feature(new_track.curr_feat)
            self.update_template(new_track.template)
        self.cast_vote(getattr(new_track, 'cls_id', None), getattr(new_track, 'score', 1.0))
        self.invalidate()

    def update_with_kf(self, new_track, frame_id, mean, covariance, update_feature=True):
        self.frame_id = frame_id
        self.track_len += 1
        self.mean, self.covariance = mean, covariance
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.curr_tlwh = np.array(new_track.tlwh, dtype=np.float32)
        self.miss_streak = 0
        self.recovered_streak = 0
        self._is_recovered_this_frame = False
        self.time_since_update = 0
        self.hit_window.append(1)
        if update_feature:
            self._update_conf(getattr(new_track, 'conf', None))
            self._ingest_feature(new_track.curr_feat)
            self.update_template(new_track.template)
        self.cast_vote(getattr(new_track, 'cls_id', None), getattr(new_track, 'score', 1.0))
        self.invalidate()

    def update_recovered(self, xywh, frame_id, R_scale, score=None):
        conf = float(1.0 / (1.0 + max(R_scale, 0.0)))
        self.mean, self.covariance = self.kalman_filter.update_nsa(
            self.mean, self.covariance, np.asarray(xywh, np.float32), conf)
        self.state = TrackState.Tracked
        self.is_activated = True
        c = self.mean[:4].copy()
        c[:2] -= c[2:] / 2
        self.curr_tlwh = c.astype(np.float32)
        self.frame_id = frame_id
        if score is not None:
            self.score = float(score)
        self.track_len += 1
        self.recovered_streak = getattr(self, 'recovered_streak', 0) + 1
        self._is_recovered_this_frame = True
        self.time_since_update = 0
        self.invalidate()

    # emitted class label
    @property
    def emit_cls(self):
        """The online temporal consensus class label."""
        return self._emit_cls

    # quality / confidence for output filtering
    def quality(self):
        hw = self.hit_window
        hr = float(np.clip(sum(hw) / max(len(hw), 1), 1e-3, 1.0))
        if (self.smooth_feat is not None and self.curr_feat is not None
                and self.smooth_feat.shape == self.curr_feat.shape
                and self.smooth_feat.size > 1):
            ac = 0.5 * (1.0 + float(np.dot(self.smooth_feat, self.curr_feat)))
            ac = float(np.clip(ac, 1e-3, 1.0))
        else:
            ac = 0.75
        return float(np.clip((hr ** 0.5) * (ac ** 0.5), 0.0, 1.0))

    # box format helpers
    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    def __repr__(self):
        return 'OT_({}-{})_({}-{})'.format(self.cls_id, self.track_id, self.start_frame, self.end_frame)


