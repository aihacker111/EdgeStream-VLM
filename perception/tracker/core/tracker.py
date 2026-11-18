# AERO-MOT 🚁 AGPL-3.0 License
# Copyright (C) 2026 AERO-MOT Authors

"""FalconTracker - the Falcon-MOT multi-class online tracker.

The per-frame data flow (see :meth:`update`) is:

    predict (Kalman) -> query appearance-motion field (QAM)
      -> associate high-score detections (SPEAR) + recover confident misses
      -> associate low-score detections against still-unmatched tracks
      -> match unconfirmed tracks / spawn or revive-from-gallery new ids
      -> survive or retire lost tracks -> de-duplicate -> emit outputs

The research logic is concentrated in two unified modules: **SPEAR** for
self-calibrated probabilistic evidence association, and **QAM** for
appearance-conditioned continuous latent field mode seeking and recovery.
"""

from collections import defaultdict
import cv2
import numpy as np

from ..motion.kalman_filter import KalmanFilter
from .base import IdCounter, TrackState
from .track import FalconTrack
from .gallery import ReIDGallery
from ..methods.qam import QueryAppearanceMotion
from ..methods.spear import SPEAR
from ..ops import track_ops as ops


class FalconTracker:
    def __init__(self, opt, frame_rate=30):
        cv2.setRNGSeed(42)
        self.opt = opt
        self.num_classes = opt.num_classes
        self.det_thresh = opt.conf_thres

        self.tracked_tracks_dict = defaultdict(list)
        self.lost_tracks_dict = defaultdict(list)
        self.frame_id = 0
        self.cost_mode = getattr(opt, 'cost_mode', 'evidential')
        self.kalman_filter = KalmanFilter(adaptive=(self.cost_mode != 'linear' and self.num_classes > 1))

        self.frame_rate = max(1, int(frame_rate))
        self.cand_iou_thres = float(getattr(opt, 'cand_iou_thres', 0.2))
        self.survival_mode = getattr(opt, 'survival_mode', 'kinetic')
        self.max_age = int(getattr(opt, 'max_age', 30))
        self.use_appearance_motion = bool(getattr(opt, 'use_appearance_motion', 1))
        self.qam_distinctiveness_margin = float(getattr(opt, 'qam_distinctiveness_margin', 0.25))
        self.qam_max_streak = int(getattr(opt, 'qam_max_streak', 1))
        self.qam_quality_gate = float(getattr(opt, 'qam_quality_gate', 0.25))
        self.qam = QueryAppearanceMotion(opt)
        self.spear = SPEAR(opt, self.kalman_filter, self.qam, frame_rate=self.frame_rate)
        self.use_reid_gallery = bool(getattr(opt, 'use_reid_gallery', True))
        self.gallery = ReIDGallery(frame_rate=self.frame_rate, enabled=self.use_reid_gallery, single_class=(self.num_classes == 1))

        self._curr_img = None
        self._image_area = None
        self.id_counter = IdCounter()   # [port] per-instance ids (one tracker per camera)

        # [port] dataset-specific switches lifted out of the code into config.
        self.minority_large_classes = set(int(c) for c in getattr(opt, 'minority_large_classes', ()) or ())
        self.class_penalty = float(getattr(opt, 'class_penalty', 18.0))
        self.qam_beta_modulation = getattr(opt, 'qam_beta_modulation', None)

    def set_dense(self, reid_dense, stride, ratio_x, ratio_y, pad_w=0.0, pad_h=0.0, dense_normed=None):
        """Forward dense ReID features to QAM; public API is unchanged."""
        emb_dim = self.qam.set_dense(reid_dense, stride, ratio_x, ratio_y, pad_w, pad_h, dense_normed)
        if emb_dim is not None:
            FalconTrack.EMB_DIM = emb_dim

    def set_image(self, img):
        """Store current frame reference (retained for API compatibility)."""
        self._curr_img = img

    def reset(self):
        cv2.setRNGSeed(42)
        self.tracked_tracks_dict = defaultdict(list)
        self.lost_tracks_dict = defaultdict(list)
        self.frame_id = 0
        self.kalman_filter = KalmanFilter(adaptive=(self.cost_mode != 'linear' and self.num_classes > 1))
        self.qam.reset()
        self.spear.kalman_filter = self.kalman_filter
        self.spear.reset()
        self.gallery.reset()
        self.id_counter.reset()
        self._curr_img = None

    # main per-frame update
    def update(self, dets_per_class, h_orig, w_orig, cands_per_class=None):
        """Unified per-frame tracking step structured in 5 principled stages:

        1. Kinematic Prediction (Kalman)
        2. Two-Stage Probabilistic Evidence Association (SPEAR)
        3. Measurement Update & Continuity-Preserving Latent Recovery (QAM)
        4. Lifecycle Management (Unconfirmed Tracks & Long-Term ReID Gallery)
        5. Temporal Category Consensus (TCC) Output Routing
        """
        self.frame_id += 1
        if self.frame_id == 1:
            self.id_counter.reset()
        image_area = float(h_orig) * float(w_orig)

        # Stage 1: Kinematic prediction
        pools, uncs = self._predict_pools()

        # Prepare detections
        all_dets = [d for c in range(self.num_classes) for d in dets_per_class.get(c, [])]
        for d in all_dets:
            d.id_counter = self.id_counter
        if cands_per_class:
            for c_list in cands_per_class.values():
                for d in c_list:
                    d.id_counter = self.id_counter
        n_dets_frame = len(all_dets)

        # Query Appearance-Motion (QAM) dense field sampling
        if self.use_appearance_motion and self.qam.dense_raw is not None:
            self.qam.sample_detection_templates(all_dets)

        all_tracks = [t for c in range(self.num_classes) for t in pools[c]]

        if self.use_appearance_motion:
            qam_handle, qam_idx = self.qam.launch(all_tracks, detections=all_dets)
            qam = self.qam.collect(qam_handle, qam_idx)
        else:
            qam = {}

        # Stage 2: Probabilistic evidence association (SPEAR)
        all_cands = [d for c in range(self.num_classes) for d in cands_per_class.get(c, [])] if cands_per_class else None
        tracked_pairs_dict, unmatched_tracks_dict, unmatched_dets, pseudo_dict = \
            self._associate(all_tracks, all_dets, qam, image_area, n_dets_frame, all_cands=all_cands)

        # Stage 2.5: Low-confidence candidate association (cascade rescue for active tracks)
        if cands_per_class:
            unmatched_tracks_dict = self._associate_candidates(
                unmatched_tracks_dict, cands_per_class, tracked_pairs_dict, pseudo_dict)

        # Stage 3: Measurement update & QAM latent field recovery
        activated_dict, refined_dict, lost_dict = self._update_and_recover(
            tracked_pairs_dict, unmatched_tracks_dict, pseudo_dict, qam)

        # Stage 4: Unconfirmed track matching & long-term gallery revival / spawn
        self._manage_unconfirmed_and_spawns(
            uncs, unmatched_dets, qam, image_area, n_dets_frame, activated_dict)

        # Stage 5: Information-theoretic survival & spatial deduplication
        self._survive_and_deduplicate(image_area, activated_dict, refined_dict, lost_dict)

        # Stage 6: Temporal Class Consensus (TCC) output emission
        return self._emit_targets()

    # Stage 1: Kinematic Prediction
    def _predict_pools(self):
        pools, uncs = {}, {}
        for cls_id in range(self.num_classes):
            tracked, unconfirmed = [], []
            for t in self.tracked_tracks_dict[cls_id]:
                (tracked if t.is_activated else unconfirmed).append(t)
            FalconTrack.multi_predict(self.lost_tracks_dict[cls_id], self.kalman_filter)
            FalconTrack.multi_predict(tracked, self.kalman_filter)
            FalconTrack.multi_predict(unconfirmed, self.kalman_filter)
            pool = ops.join_tracks(tracked, self.lost_tracks_dict[cls_id])
            pools[cls_id] = pool
            uncs[cls_id] = unconfirmed
        return pools, uncs

    # Stage 2: Probabilistic Evidence Association (SPEAR)
    def _associate(self, all_pools, all_dets, qam, image_area, n_dets_frame, all_cands=None):
        allow_cross = not bool(getattr(self.opt, "disable_tcc", False))
        class_penalty = self.class_penalty if (self.num_classes > 1 and allow_cross) else 0.0

        matches, u_track, u_det, pseudo = self.spear.associate(
            all_pools, all_dets, qam, image_area, n_dets_frame, self.frame_id,
            allow_recovery=True, class_penalty=class_penalty, allow_cross_class=allow_cross, cands=all_cands)

        tracked_pairs_dict = defaultdict(list)
        for it, idd in matches:
            track, det = all_pools[it], all_dets[idd]
            tracked_pairs_dict[track.cls_id].append((track, det))

        unmatched_pools = [all_pools[i] for i in u_track]
        unmatched_dets = [all_dets[i] for i in u_det]

        unmatched_tracks_dict = defaultdict(list)
        for t in unmatched_pools:
            unmatched_tracks_dict[t.cls_id].append(t)

        pseudo_dict = defaultdict(list)
        for p in pseudo:
            pseudo_dict[p[0].cls_id].append(p)

        return tracked_pairs_dict, unmatched_tracks_dict, unmatched_dets, pseudo_dict

    # Stage 2.5: Low-Confidence Candidate Association
    def _associate_candidates(self, unmatched_tracks_dict, cands_per_class, tracked_pairs_dict, pseudo_dict):
        """Rescue confirmed active and recently lost tracks using QAM Bayesian log-posterior over candidates."""
        from ..methods.spear import assign_linear

        margin_val = float(getattr(self.opt, "qam_distinctiveness_margin", 0.25))
        q_gate = float(getattr(self.opt, "qam_quality_gate", 0.25))
        r_min_val = int(getattr(self.opt, "qam_roi_radius", 16))
        beta_val = float(getattr(self.opt, "qam_beta", 5.0))
        global_search = bool(getattr(self.opt, "qam_global_search", False))
        cov_mode = str(getattr(self.opt, "qam_cov_mode", "margin_scaled"))
        eff_beta = 5.0 + 0.20 * (beta_val - 5.0)
        gate_relax = (0.25 - q_gate) + 1.8 * (0.25 - margin_val)
        sigma_eff_sq = float(
            0.095 * np.exp(2.2 * gate_relax + 0.02 * (r_min_val - 16))
        )
        ref_r0 = float(np.exp(-5.0 * 0.25))
        ref_w = 0.095 / (0.095 + ref_r0)
        base_r0 = float(QueryAppearanceMotion.visual_observation_variance(
            0.55, 0.30, 1.0, beta=eff_beta, cov_mode=cov_mode
        ))
        base_w_prec = sigma_eff_sq / (sigma_eff_sq + base_r0)
        alpha_excess = float(np.clip(base_w_prec - ref_w, -0.15, 0.65))
        shift_w = float(np.clip(0.95 * max(0.0, alpha_excess - 0.05), 0.0, 0.65))
        cost_limit = float(
            np.clip(self.cand_iou_thres + (0.35 * alpha_excess if (self.use_appearance_motion and not global_search) else 0.0), 0.18, 0.58)
        )

        new_unmatched_tracks = defaultdict(list)

        for c in range(self.num_classes):
            unm_tracks = unmatched_tracks_dict.get(c, [])
            cands_c = cands_per_class.get(c, [])
            if self.use_appearance_motion and not global_search and margin_val > 0.25 and cands_c:
                min_sc = 0.15 + 0.25 * (margin_val - 0.25)
                cands_c = [d for d in cands_c if getattr(d, "score", 0.0) >= min_sc]

            if not unm_tracks or not cands_c:
                new_unmatched_tracks[c] = unm_tracks
                continue

            active_unm = [t for t in unm_tracks if t.state == TrackState.Tracked]
            lost_unm = [t for t in unm_tracks if t.state == TrackState.Lost]
            other_unm = [t for t in unm_tracks if t.state not in (TrackState.Tracked, TrackState.Lost)]

            # [port] was hard-coded as ``num_classes == 5 and c in (2, 3, 4)`` (VisDrone 5-class eval).
            is_minority_large_cls = (c in self.minority_large_classes and not global_search)
            if is_minority_large_cls:
                cands_c = [d for d in cands_c if getattr(d, "score", 0.0) >= 0.28]
                if not cands_c:
                    new_unmatched_tracks[c] = unm_tracks
                    continue

            def _build_qam_cost(t_list, d_list):
                return self._qam_candidate_cost(
                    t_list, d_list, global_search=global_search, r_min_val=r_min_val,
                    q_gate=q_gate, shift_w=shift_w, margin_val=margin_val)

            rem_active = active_unm
            rem_cands = cands_c
            rescued_ids = set()
            eff_limit = 0.45 if global_search else (min(cost_limit, 0.20) if is_minority_large_cls else cost_limit)
            if active_unm and rem_cands:
                dists = _build_qam_cost(active_unm, rem_cands)
                m, u_t, u_d = assign_linear(dists, match_thresh=eff_limit)
                for r, col in m:
                    tracked_pairs_dict[c].append((active_unm[int(r)], rem_cands[int(col)]))
                    rescued_ids.add(id(active_unm[int(r)]))
                rem_active = [active_unm[i] for i in u_t]
                rem_cands = [rem_cands[j] for j in u_d]

            rem_lost = lost_unm
            if self.use_appearance_motion and not is_minority_large_cls and lost_unm and rem_cands:
                elig_lost = lost_unm
                if elig_lost:
                    dists_l = _build_qam_cost(elig_lost, rem_cands)
                    m_l, _, _ = assign_linear(dists_l, match_thresh=eff_limit)
                    matched_lost_ids = set()
                    for r, col in m_l:
                        trk_l = elig_lost[int(r)]
                        tracked_pairs_dict[c].append((trk_l, rem_cands[int(col)]))
                        rescued_ids.add(id(trk_l))
                        matched_lost_ids.add(id(trk_l))
                    rem_lost = [t for t in lost_unm if id(t) not in matched_lost_ids]

            new_unmatched_tracks[c] = rem_active + rem_lost + other_unm
            if rescued_ids and pseudo_dict.get(c):
                pseudo_dict[c] = [p for p in pseudo_dict[c] if id(p[0]) not in rescued_ids]

        return new_unmatched_tracks

    # [port] vectorised version of AERO-MOT's nested ``_build_qam_cost`` (was an
    # O(T*D) Python double loop). Numerically equivalent; see tests/test_qam_cost.py.
    def _qam_candidate_cost(self, t_list, d_list, *, global_search, r_min_val, q_gate, shift_w, margin_val):
        n_t, n_d = len(t_list), len(d_list)
        if n_t == 0 or n_d == 0:
            return np.ones((n_t, n_d), dtype=np.float64)
        iou_d = ops.matching.iou_distance(t_list, d_list)
        if not self.use_appearance_motion:
            return iou_d

        cos_sim = _pairwise_track_det_cosine(t_list, d_list)
        if global_search:
            return 1.0 - cos_sim

        t_tlwh = np.asarray([t.tlwh for t in t_list], dtype=np.float64)
        d_tlwh = np.asarray([d.tlwh for d in d_list], dtype=np.float64)
        tcx = t_tlwh[:, 0] + 0.5 * t_tlwh[:, 2]
        tcy = t_tlwh[:, 1] + 0.5 * t_tlwh[:, 3]
        dcx = d_tlwh[:, 0] + 0.5 * d_tlwh[:, 2]
        dcy = d_tlwh[:, 1] + 0.5 * d_tlwh[:, 3]
        t_scale = np.sqrt(np.maximum(t_tlwh[:, 2] * t_tlwh[:, 3], 1.0))
        rad = np.maximum(float(r_min_val) * 4.0, 1.8 * t_scale)[:, None]          # (T,1)
        dist = np.hypot(dcx[None, :] - tcx[:, None], dcy[None, :] - tcy[:, None])  # (T,D)

        in_win = ~(dist > rad)
        s_val = cos_sim - (dist ** 2) / (2.0 * (0.85 * rad) ** 2)

        gated = iou_d.copy()
        gated[~in_win] = 1.0
        if shift_w > 0.0:
            d_vis = 0.50 * (dist / np.maximum(rad, 1.0)) + 0.50 * (1.0 - np.maximum(cos_sim, 0.0))
            blended = (1.0 - shift_w) * iou_d + shift_w * d_vis
            if q_gate > 0.0:
                blended = np.where(s_val < q_gate, 1.0, blended)
            gated = np.where(in_win, blended, gated)
        else:
            hi = cos_sim > (0.30 + margin_val)
            lo = (~hi) & (q_gate > 0.0) & (cos_sim < (0.10 + q_gate))
            gated = np.where(in_win & hi, 0.70 * iou_d + 0.30 * (1.0 - cos_sim), gated)
            gated = np.where(in_win & lo, np.minimum(1.0, iou_d + 0.20), gated)

        if margin_val > 0.0:
            n_in = in_win.sum(axis=1)
            rows = np.where(n_in >= 2)[0]
            if rows.size:
                s_masked = np.where(in_win[rows], s_val[rows], -np.inf)
                top2 = -np.partition(-s_masked, 1, axis=1)[:, :2]       # [best, second]
                win_margin = top2[:, 0] - top2[:, 1]
                ambiguous = win_margin < 0.08 * (margin_val / 0.25) * (1.0 - 0.5 * shift_w)
                for r in rows[ambiguous]:
                    m = in_win[r]
                    gated[r, m] = np.minimum(1.0, gated[r, m] + 0.12 * (1.0 - 0.4 * shift_w))
        return gated

    # Stage 3: Measurement Update & Latent Field Recovery (QAM)
    def _update_and_recover(self, tracked_pairs_dict, unmatched_tracks_dict, pseudo_dict, qam):
        activated_dict = defaultdict(list)
        refined_dict = defaultdict(list)
        lost_dict = defaultdict(list)

        for cls_id in range(self.num_classes):
            tracked_pairs = tracked_pairs_dict[cls_id]
            tracked_active_pairs = []
            for track, det in tracked_pairs:
                if track.state == TrackState.Tracked:
                    tracked_active_pairs.append((track, det))
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_dict[cls_id].append(track)

            self._batch_kf_update(tracked_active_pairs, self.frame_id)
            activated_dict[cls_id].extend(t for t, _ in tracked_active_pairs)

            # QAM Latent Field active recovery under detection dropouts
            recovered = set()
            max_streak = int(getattr(self, 'qam_max_streak', 1))
            margin_thresh = float(getattr(self, "qam_distinctiveness_margin", 0.25))
            for track, xywh, R_scale in pseudo_dict[cls_id]:
                # Strict guard: only confirmed active tracks are eligible for single-frame dropout recovery
                if track.state != TrackState.Tracked:
                    continue
                st = getattr(track, 'recovered_streak', 0)
                if st >= max_streak:
                    continue
                info = qam.get(id(track))
                if info is not None:
                    _, pk, pk2 = info
                    if (float(pk) - float(pk2)) < margin_thresh:
                        continue
                track.update_recovered(xywh, self.frame_id, R_scale, score=getattr(track, 'score', 0.3))
                if track.state == TrackState.Lost:
                    refined_dict[cls_id].append(track)
                else:
                    activated_dict[cls_id].append(track)
                recovered.add(id(track))

            for track in unmatched_tracks_dict[cls_id]:
                if id(track) in recovered:
                    continue
                track.miss_streak += 1
                track.hit_window.append(0)
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_dict[cls_id].append(track)

        return activated_dict, refined_dict, lost_dict

    # Stage 4: Unconfirmed Track Matching & Gallery Revival / Spawns
    def _manage_unconfirmed_and_spawns(self, uncs, unmatched_dets_high, qam, image_area, n_dets_frame, activated_dict):
        all_uncs = [u for c in range(self.num_classes) for u in uncs[c]]
        pending_spawns = []

        if all_uncs and unmatched_dets_high:
            m_unc, u_unc, u_det2 = self.spear.associate(
                all_uncs, unmatched_dets_high, qam, image_area, n_dets_frame, self.frame_id,
                allow_recovery=False, allow_cross_class=False)[:3]
            for it, idd in m_unc:
                track = all_uncs[it]
                det = unmatched_dets_high[idd]
                track.update(det, self.frame_id)
                activated_dict[track.cls_id].append(track)
            for it in u_unc:
                all_uncs[it].mark_removed()
            for i_new in u_det2:
                det = unmatched_dets_high[i_new]
                if det.score >= self.det_thresh:
                    pending_spawns.append((det.cls_id, det))
        else:
            for u in all_uncs:
                u.mark_removed()
            for det in unmatched_dets_high:
                if det.score >= self.det_thresh:
                    pending_spawns.append((det.cls_id, det))

        # Spawn new tracks with Long-Term ReID Gallery revival
        all_tracked = [t for c in range(self.num_classes) for t in self.tracked_tracks_dict[c]]
        spawned_cls = set()
        for cls_id, det in pending_spawns:
            if all_tracked:
                dbox = det.tlbr
                ex_tlbr = np.stack([t.tlbr for t in all_tracked])
                if float(ops.iou_1_to_n(dbox, ex_tlbr).max()) > 0.7:
                    continue
            revived = self.gallery.revive(
                cls_id, det, self.frame_id, self.tracked_tracks_dict, self.kalman_filter, self.lost_tracks_dict
            ) if self.use_reid_gallery else None
            if revived is not None:
                self.tracked_tracks_dict[cls_id].append(revived)
                all_tracked.append(revived)
                spawned_cls.add(cls_id)
                continue
            det.activate(self.kalman_filter, self.frame_id)
            self.tracked_tracks_dict[cls_id].append(det)
            all_tracked.append(det)
            spawned_cls.add(cls_id)

        for cls_id in spawned_cls:
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = \
                ops.remove_duplicate_tracks(self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

    # Stage 5: Survival & Deduplication
    def _survive_and_deduplicate(self, image_area, activated_dict, refined_dict, lost_dict):
        for cls_id in range(self.num_classes):
            # Information-theoretic survival based on scene mean-free area
            self._survive_lost(cls_id, image_area)

            self.tracked_tracks_dict[cls_id] = ops.join_tracks(
                ops.join_tracks([t for t in self.tracked_tracks_dict[cls_id] if t.state == TrackState.Tracked],
                                activated_dict[cls_id]),
                refined_dict[cls_id])

            lost = ops.sub_tracks(self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
            lost.extend(lost_dict[cls_id])
            self.lost_tracks_dict[cls_id] = [t for t in lost if t.state == TrackState.Lost]

            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = \
                ops.remove_duplicate_tracks(self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            self.tracked_tracks_dict[cls_id] = ops.suppress_same_class_overlaps(
                self.tracked_tracks_dict[cls_id], iou_thresh=0.8, min_age_gap=5)

        ops.remove_cross_class_duplicates(self.tracked_tracks_dict, self.num_classes, iou_thresh=0.8)

    # Stage 6: Temporal Category Consensus (TCC) Output Emission
    def _emit_targets(self):
        output_dict = defaultdict(list)
        for cls_id in range(self.num_classes):
            new_tracked = []
            for t in self.tracked_tracks_dict[cls_id]:
                if t.state == TrackState.Tracked:
                    new_tracked.append(t)
                elif t.state == TrackState.Lost:
                    self.lost_tracks_dict[cls_id].append(t)
            self.tracked_tracks_dict[cls_id] = new_tracked

            max_streak = int(getattr(self, 'qam_max_streak', 1))
            quality_gate = float(getattr(self, 'qam_quality_gate', 0.25))
            for t in new_tracked:
                is_recovered = getattr(t, '_is_recovered_this_frame', False)
                if is_recovered:
                    # Suppress stale virtual extrapolations beyond initial trustworthy window
                    if getattr(t, 'recovered_streak', 99) > max_streak:
                        continue
                if t.is_activated:
                    t.q = t.quality()
                    if t.q:
                        if is_recovered and t.q < quality_gate:
                            continue
                        out_cls = getattr(t, 'birth_cls', t.cls_id) if getattr(self.opt, "disable_tcc", False) else t.emit_cls
                        output_dict[out_cls].append(t)

        self.gallery.prune(self.frame_id)
        return output_dict

    def _batch_kf_update(self, pairs: list, frame_id: int, update_feature: bool = True) -> None:
        """Vectorized multi-track Kalman filter update for confirmed active tracks."""
        if not pairs:
            return
        tracks, detections = zip(*pairs)
        means = np.stack([t.mean for t in tracks])
        covs = np.stack([t.covariance for t in tracks])
        measurements = np.stack([FalconTrack.tlwh_to_xywh(np.asarray(d.tlwh, dtype=np.float32)) for d in detections])
        up_means, up_covs = self.kalman_filter.multi_update(means, covs, measurements)
        beta_val = float(getattr(self.opt, "qam_beta", 5.0))
        # [port] original rule: modulate only when beta not in (2.0, 5.0); now overridable.
        rule = (beta_val not in (2.0, 5.0)) if self.qam_beta_modulation is None else bool(self.qam_beta_modulation)
        modulate_beta = self.use_appearance_motion and rule
        for idx, (t, d, m, c) in enumerate(zip(tracks, detections, up_means, up_covs)):
            sc = float(getattr(d, 'score', 0.0))
            up_feat = update_feature and (sc >= self.det_thresh)
            if modulate_beta and sc < self.det_thresh:
                # Precision-weighted blending between Kalman prior and candidate update under R_q = s^2 exp(-beta * Delta S)
                w_gain = float(np.clip(np.exp(-0.12 * (beta_val - 5.0)), 0.65, 1.25))
                m = means[idx] + w_gain * (m - means[idx])
                w_cov = float(np.clip(w_gain, 0.65, 1.0))
                c = (1.0 - w_cov) * covs[idx] + w_cov * c
            t.update_with_kf(d, frame_id, m, c, update_feature=up_feat)

    def _survive_lost(self, cls_id: int, image_area: float) -> None:
        """Information-theoretic survival for lost tracks based on scene mean-free area."""
        lost = self.lost_tracks_dict[cls_id]
        if not lost:
            return

        max_age = int(getattr(self, "max_age", 30))
        if getattr(self, "survival_mode", "kinetic") == "static":
            surviving = []
            for track in lost:
                if getattr(track, "time_since_update", 0) <= max_age:
                    surviving.append(track)
                else:
                    self.gallery.retire(cls_id, track, self.frame_id)
            self.lost_tracks_dict[cls_id] = surviving
            return

        refs = [t for t in self.tracked_tracks_dict[cls_id] if t.mean is not None] + [t for t in lost if t.mean is not None]
        positions = np.array([t.mean[:2] for t in refs], dtype=np.float64) if refs else np.zeros((0, 2))

        if positions.shape[0] >= 2:
            diff = positions[:, None, :] - positions[None, :, :]
            dist_sq = (diff * diff).sum(axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            mean_free_area = np.pi * (0.5 * float(np.median(np.sqrt(dist_sq.min(axis=1))))) ** 2
        else:
            mean_free_area = float(image_area)

        surviving = []
        for track in lost:
            tsu = getattr(track, "time_since_update", self.frame_id - track.frame_id)
            if tsu <= max_age:
                surviving.append(track)
                continue
            if tsu > max_age + 2:
                self.gallery.retire(cls_id, track, self.frame_id)
                continue
            _, cov = self.kalman_filter.project(track.mean, track.covariance)
            area = max(float(track.tlwh[2] * track.tlwh[3]), 1.0)
            cov_det = max(float(np.linalg.det(cov[:2, :2])), 1e-12)
            if 2.0 * np.pi * np.sqrt(cov_det) < max(mean_free_area, area):
                surviving.append(track)
            else:
                self.gallery.retire(cls_id, track, self.frame_id)
        self.lost_tracks_dict[cls_id] = surviving


def _pairwise_track_det_cosine(t_list, d_list):
    """cos(track.smooth_feat, det.curr_feat) with AERO-MOT's validity rules (0 when invalid)."""
    n_t, n_d = len(t_list), len(d_list)
    cos_sim = np.zeros((n_t, n_d), dtype=np.float64)
    t_feats = [getattr(t, "smooth_feat", None) for t in t_list]
    d_feats = [getattr(d, "curr_feat", None) for d in d_list]
    t_ok = [f is not None and np.linalg.norm(f) > 1e-6 for f in t_feats]
    if not any(t_ok):
        return cos_sim
    dims = {np.asarray(f).size for f, ok in zip(t_feats, t_ok) if ok}
    for dim in dims:
        ti = [i for i, (f, ok) in enumerate(zip(t_feats, t_ok)) if ok and np.asarray(f).size == dim]
        di = [j for j, f in enumerate(d_feats) if f is not None and np.asarray(f).size == dim]
        if not ti or not di:
            continue
        T = np.stack([np.asarray(t_feats[i], dtype=np.float64).ravel() for i in ti])
        D = np.stack([np.asarray(d_feats[j], dtype=np.float64).ravel() for j in di])
        cos_sim[np.ix_(ti, di)] = T @ D.T
    return cos_sim
