# AERO-MOT 🚁 AGPL-3.0 License
# Copyright (C) 2026 AERO-MOT Authors

"""SPEAR — Self-calibrating Probabilistic Evidence Association & Recovery.

This module implements the Bayesian hypothesis testing framework for aerial MOT:
multi-modal Log-Likelihood Ratios (Student-t marginal kinematic innovation and
Gate-Conditioned GLRT directional vMF appearance), principled joint log-evidence
fusion (L = L_mot + L_app), and single-pass bipartite assignment restricted to
positive evidence (L > 0).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import numpy as np

from ..core.base import TrackState

CHI2_2DOF_999 = 13.816  # Chi-Square critical value at p=0.999 (2 DoF) for spatial gating


# 1. Directional Appearance Concentration

def vmf_kappa(resultant_length: np.ndarray | float, embedding_dim: int, use_dynamic: bool = True, kappa_max: Optional[float] = None) -> np.ndarray:
    """Asymptotic MLE estimation of the von Mises-Fisher concentration kappa on S^{D-1}."""
    if not use_dynamic:
        r = np.asarray(resultant_length, dtype=np.float64)
        if r.ndim == 0:
            return 1.0
        return np.ones_like(r, dtype=np.float64)
    r_bar = np.clip(np.asarray(resultant_length, dtype=np.float64), 0.0, 1.0 - 1e-6)
    r_bar_sq = r_bar * r_bar
    kappa = r_bar * (embedding_dim - r_bar_sq) / (1.0 - r_bar_sq)
    if kappa_max is not None and float(kappa_max) > 0:
        kappa = np.minimum(kappa, float(kappa_max))
    return kappa.astype(np.float64)

# 2. Multi-Modal Log-Likelihood Ratios (LLR)

def motion_llr(
    track_projected_mean: np.ndarray,
    track_projected_covariance: np.ndarray,
    detection_measurements: np.ndarray,
    image_area: float,
    n_dets: Optional[int] = None,
    chi2_gate: float = CHI2_2DOF_999,
    nu: Optional[float] = None,
    stationary_indicator: Optional[np.ndarray] = None,
    use_two_regime_gate: Optional[bool] = None,
    disable_tau_mot: bool = False,
    adaptive_pan_inflation: float = 1.0,
) -> np.ndarray:
    """Kinematic Log-Likelihood Ratio against spatial Poisson clutter: L_mot = ln(p(z|H1)/p(z|H0))."""
    track_mean = np.asarray(track_projected_mean, dtype=np.float64)
    det_boxes = np.asarray(detection_measurements, dtype=np.float64)
    covariance = np.asarray(track_projected_covariance, dtype=np.float64)

    num_tracks, num_dets = track_mean.shape[0], det_boxes.shape[0]
    if num_tracks == 0 or num_dets == 0:
        return np.zeros((num_tracks, num_dets), dtype=np.float64)

    effective_dets = num_dets if n_dets is None else n_dets
    log_clutter_density = np.log(max(float(effective_dets), 1.0) / max(image_area, 1.0))

    inv_cov = np.linalg.inv(covariance)
    # 2-DoF spatial position gating & 2D log-determinant matching 2D image plane clutter density
    inv_pos_cov = np.linalg.inv(covariance[:, :2, :2])
    _, log_det_pos = np.linalg.slogdet(2.0 * np.pi * covariance[:, :2, :2])

    residual = det_boxes[None, :, :] - track_mean[:, None, :]
    mahalanobis_sq = np.einsum("tdi,tij,tdj->td", residual, inv_cov, residual)

    pos_residual = residual[:, :, :2]
    pos_mahalanobis_sq = np.einsum("tdi,tij,tdj->td", pos_residual, inv_pos_cov, pos_residual)

    use_two_regime = (
        bool(use_two_regime_gate)
        if use_two_regime_gate is not None
        else (bool(nu is not None and float(nu) > 0) or (stationary_indicator is not None))
    )
    eff_maha_sq = mahalanobis_sq
    eff_log_det_pos = log_det_pos
    if adaptive_pan_inflation != 1.0 and stationary_indicator is not None:
        pan_mask = ~np.asarray(stationary_indicator, dtype=bool)
        eff_maha_sq = mahalanobis_sq.copy()
        eff_maha_sq[pan_mask] /= float(adaptive_pan_inflation)
        eff_log_det_pos = log_det_pos.copy()
        eff_log_det_pos[pan_mask] += 2.0 * np.log(float(adaptive_pan_inflation))

    tau_mot = np.zeros((num_tracks, 1), dtype=np.float64) if disable_tau_mot else (-0.5 * eff_log_det_pos[:, None] - log_clutter_density)
    if nu is not None and float(nu) > 0:
        nu_f = float(nu)
        kinematic_llr = -0.5 * (nu_f + 4.0) * np.log1p(eff_maha_sq / nu_f) + tau_mot
    else:
        nu_f = 4.0
        kinematic_llr = -0.5 * eff_maha_sq + tau_mot

    if use_two_regime:
        scale_mahalanobis_sq = np.maximum(0.0, mahalanobis_sq - pos_mahalanobis_sq)
        pan_open = np.ones((num_tracks, 1), dtype=bool) if stationary_indicator is None else (~np.asarray(stationary_indicator, dtype=bool))[:, None]
        valid_gate = (pos_mahalanobis_sq <= 396.0) & (scale_mahalanobis_sq <= 62.0) & (
            (pos_mahalanobis_sq <= chi2_gate) | (pan_open & (np.log1p(pos_mahalanobis_sq / nu_f) + 0.10 * scale_mahalanobis_sq <= 6.2))
        )
        kinematic_llr[~valid_gate] = -np.inf
    else:
        kinematic_llr[pos_mahalanobis_sq > chi2_gate] = -np.inf
    return kinematic_llr


def appearance_llr(
    track_embeddings: np.ndarray,
    detection_embeddings: np.ndarray,
    kappa_track: np.ndarray,
    track_cls: Optional[np.ndarray] = None,
    det_cls: Optional[np.ndarray] = None,
    local_gate: Optional[np.ndarray] = None,
    is_active: Optional[np.ndarray] = None,
    lambda_glrt: float = 0.8,
) -> np.ndarray:
    """Directional appearance Log-Likelihood Ratio against ambient and competitive GLRT null on S^{D-1}."""
    tracks_mu = np.asarray(track_embeddings, dtype=np.float64)
    dets_feat = np.asarray(detection_embeddings, dtype=np.float64)

    num_tracks, num_dets = tracks_mu.shape[0], dets_feat.shape[0]
    if num_tracks == 0 or num_dets == 0:
        return np.zeros((num_tracks, num_dets), dtype=np.float64)

    valid_tracks = np.linalg.norm(tracks_mu, axis=1) > 1e-6
    valid_dets = np.linalg.norm(dets_feat, axis=1) > 1e-6
    cosine_sim = tracks_mu @ dets_feat.T

    if track_cls is not None and det_cls is not None:
        same_cls = (track_cls[:, None] == det_cls[None, :]) & valid_tracks[:, None] & valid_dets[None, :]
        same_cls_sims = cosine_sim[same_cls]
        median_bg = float(np.median(same_cls_sims)) if same_cls_sims.size > 0 else 0.0
    else:
        valid_sims = cosine_sim[np.ix_(valid_tracks, valid_dets)]
        median_bg = float(np.median(valid_sims)) if valid_sims.size > 0 else 0.0

    track_weights = np.asarray(kappa_track, dtype=np.float64)[:, None]
    if local_gate is not None:
        gate = np.asarray(local_gate, dtype=bool) & valid_tracks[:, None] & valid_dets[None, :]
        gated_sim = np.where(gate, cosine_sim, -1.0)
        if is_active is not None:
            act = np.asarray(is_active, dtype=bool)[:, None] & gate
            gated_sim_col = np.where(act, cosine_sim, -1.0)
        else:
            gated_sim_col = gated_sim

        row_second = np.full((num_tracks, num_dets), median_bg, dtype=np.float64)
        if num_dets > 1:
            idx_r = np.argpartition(gated_sim, -2, axis=1)
            top1_idx_r = idx_r[:, -1:]
            top2_idx_r = idx_r[:, -2:-1]
            top1_val_r = np.take_along_axis(gated_sim, top1_idx_r, axis=1)
            top2_val_r = np.take_along_axis(gated_sim, top2_idx_r, axis=1)
            is_top1_r = (np.arange(num_dets)[None, :] == top1_idx_r)
            second_r = np.where(is_top1_r, top2_val_r, top1_val_r)
            row_second = np.maximum(second_r, median_bg)

        col_second = np.full((num_tracks, num_dets), median_bg, dtype=np.float64)
        if num_tracks > 1:
            idx_c = np.argpartition(gated_sim_col, -2, axis=0)
            top1_idx_c = idx_c[-1:, :]
            top2_idx_c = idx_c[-2:-1, :]
            top1_val_c = np.take_along_axis(gated_sim_col, top1_idx_c, axis=0)
            top2_val_c = np.take_along_axis(gated_sim_col, top2_idx_c, axis=0)
            is_top1_c = (np.arange(num_tracks)[:, None] == top1_idx_c)
            second_c = np.where(is_top1_c, top2_val_c, top1_val_c)
            col_second = np.maximum(second_c, median_bg)

        comp_null = np.maximum(row_second, col_second)
        eff_diff = (cosine_sim - median_bg) + float(lambda_glrt) * (cosine_sim - comp_null)
        directional_llr = track_weights * eff_diff
    else:
        directional_llr = track_weights * (cosine_sim - median_bg)

    directional_llr[~valid_tracks, :] = 0.0
    directional_llr[:, ~valid_dets] = 0.0
    return directional_llr


# 3. Joint Bayesian Likelihood Fusion & Assignment

def fuse_evidence(
    L_mot: np.ndarray,
    L_app: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Joint Bayesian Log-Likelihood Ratio: ln p(z|H1)/p(z|H0) = L_mot + L_app."""
    mot_ev = np.asarray(L_mot, dtype=np.float64)
    if mot_ev.size == 0:
        return mot_ev

    spatial_gate = np.isfinite(mot_ev)
    fused = np.where(spatial_gate, mot_ev, 0.0)

    if L_app is not None and np.asarray(L_app).size > 0:
        app_ev = np.asarray(L_app, dtype=np.float64)
        fused = fused + np.where(spatial_gate, app_ev, 0.0)

    fused[~spatial_gate] = -np.inf
    return fused



def assign(log_evidence: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
    """Single-pass global bipartite matching restricted to strictly positive evidence (L > 0)."""
    ev = np.asarray(log_evidence, dtype=np.float64)
    num_tracks, num_dets = ev.shape
    if num_tracks == 0 or num_dets == 0:
        return np.empty((0, 2), dtype=int), list(range(num_tracks)), list(range(num_dets))

    is_positive = np.isfinite(ev) & (ev > 0.0)
    cost_matrix = np.where(is_positive, -ev, 1e9)

    try:
        import lap
        _, assignments, _ = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=0.0)
        matches = [
            [t_idx, int(d_idx)]
            for t_idx, d_idx in enumerate(assignments)
            if d_idx >= 0 and is_positive[t_idx, int(d_idx)]
        ]
    except Exception:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost_matrix)
        matches = [
            [int(r), int(c)]
            for r, c in zip(rows, cols)
            if is_positive[r, c]
        ]

    matched_t = {m[0] for m in matches}
    matched_d = {m[1] for m in matches}
    unmatched_tracks = [i for i in range(num_tracks) if i not in matched_t]
    unmatched_dets = [j for j in range(num_dets) if j not in matched_d]

    return np.asarray(matches, dtype=int).reshape(-1, 2), unmatched_tracks, unmatched_dets


def assign_linear(cost_matrix: np.ndarray, match_thresh: float = 0.70) -> Tuple[np.ndarray, List[int], List[int]]:
    """Linear cost assignment using Hungarian algorithm gated at match_thresh."""
    cost = np.asarray(cost_matrix, dtype=np.float64)
    num_tracks, num_dets = cost.shape
    if num_tracks == 0 or num_dets == 0:
        return np.empty((0, 2), dtype=int), list(range(num_tracks)), list(range(num_dets))

    cost_copy = cost.copy()
    cost_copy[cost_copy > match_thresh] = 1e9
    cost_copy[~np.isfinite(cost_copy)] = 1e9

    try:
        import lap
        _, assignments, _ = lap.lapjv(cost_copy, extend_cost=True, cost_limit=match_thresh)
        matches = [
            [t_idx, int(d_idx)]
            for t_idx, d_idx in enumerate(assignments)
            if d_idx >= 0 and cost[t_idx, int(d_idx)] <= match_thresh and np.isfinite(cost[t_idx, int(d_idx)])
        ]
    except Exception:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost_copy)
        matches = [
            [int(r), int(c)]
            for r, c in zip(rows, cols)
            if cost[r, c] <= match_thresh and np.isfinite(cost[r, c])
        ]

    matched_t = {m[0] for m in matches}
    matched_d = {m[1] for m in matches}
    unmatched_tracks = [i for i in range(num_tracks) if i not in matched_t]
    unmatched_dets = [j for j in range(num_dets) if j not in matched_d]

    return np.asarray(matches, dtype=int).reshape(-1, 2), unmatched_tracks, unmatched_dets


def linear_cost(
    tracks: list,
    detections: list,
    means: np.ndarray,
    covs: np.ndarray,
    alpha: float = 0.5,
    metric: str = "mahalanobis",
    chi2_gate: float = CHI2_2DOF_999,
    normalized: bool = False,
) -> np.ndarray:
    """Prior-art linear combination: C_ij = alpha * d_mot + (1 - alpha) * d_app with spatial gating."""
    num_tracks = len(tracks)
    num_dets = len(detections)
    if num_tracks == 0 or num_dets == 0:
        return np.empty((num_tracks, num_dets), dtype=np.float64)

    det_measurements = np.empty((num_dets, 4), dtype=np.float64)
    for j, det in enumerate(detections):
        x, y, w, h = det.tlwh
        det_measurements[j] = (x + w * 0.5, y + h * 0.5, w, h)

    t_feats = _extract_feature_matrix(tracks, "smooth_feat")
    for i, t in enumerate(tracks):
        if np.linalg.norm(t_feats[i]) <= 1e-6 and getattr(t, "curr_feat", None) is not None:
            f = np.asarray(t.curr_feat, dtype=np.float64)
            if f.size == t_feats.shape[1]:
                t_feats[i] = f

    d_feats = _extract_feature_matrix(detections, "curr_feat")
    dot = t_feats @ d_feats.T
    d_app = np.clip(1.0 - dot, 0.0, 2.0)

    if metric == "iou":
        from ..ops.association_ops import iou_distance
        d_mot = iou_distance(tracks, detections)
    elif metric == "biou":
        from ..ops.association_ops import biou_distance
        d_mot = biou_distance(tracks, detections)
    else:  # mahalanobis
        residual = det_measurements[None, :, :] - means[:, None, :]
        inv_cov = np.linalg.inv(covs)
        d_mot = np.einsum("tdi,tij,tdj->td", residual, inv_cov, residual)

    inv_pos_cov = np.linalg.inv(covs[:, :2, :2])
    pos_residual = det_measurements[None, :, :2] - means[:, None, :2]
    pos_maha = np.einsum("tdi,tij,tdj->td", pos_residual, inv_pos_cov, pos_residual)

    if normalized and metric == "mahalanobis":
        # Chi-square with 4 degrees of freedom at 0.95 quantile is 9.488
        CHI2_4DOF_95 = 9.4877
        norm_d_mot = np.clip(d_mot / CHI2_4DOF_95, 0.0, 1.0)
        norm_d_app = np.clip(d_app / 2.0, 0.0, 1.0)
        cost = alpha * norm_d_mot + (1.0 - alpha) * norm_d_app
    else:
        cost = alpha * d_mot + (1.0 - alpha) * d_app

    cost[pos_maha > chi2_gate] = np.inf
    return cost


# 4. SPEAR Association Engine

def _extract_feature_matrix(items: list, attr: str) -> np.ndarray:
    """Extract and stack 1D feature vectors from a list of objects."""
    dim = next((getattr(it, attr).size for it in items if getattr(it, attr, None) is not None and getattr(it, attr).size > 1), 1)
    arr = np.zeros((len(items), dim), dtype=np.float64)
    for i, it in enumerate(items):
        feat = getattr(it, attr, None)
        if feat is not None and feat.size == dim:
            arr[i] = feat
    return arr


class SPEAR:
    """Probabilistic Evidence Association & Recovery engine for FalconTracker."""

    def __init__(self, opt: Any, kalman_filter: Any, qam: Any, frame_rate: int = 30):
        self.opt = opt
        self.kalman_filter = kalman_filter
        self.qam = qam
        self.frame_rate = max(1, int(frame_rate))

    def reset(self) -> None:
        """Reset internal state."""
        pass

    def associate(
        self,
        tracks: list,
        detections: list,
        qam_result: dict,
        image_area: float,
        n_dets_frame: int,
        frame_id: int,
        allow_recovery: bool = True,
        class_penalty: float = 0.0,
        allow_cross_class: bool = True,
        cands: Optional[list] = None,
    ) -> Tuple[np.ndarray, List[int], List[int], list]:
        """Perform unified multi-modal evidence association and QAM recovery proposal generation."""
        num_tracks = len(tracks)
        num_dets = len(detections)
        if num_tracks == 0:
            return np.empty((0, 2), dtype=int), [], list(range(num_dets)), []

        # 1. Kinematic projection & QAM continuous precision fusion
        projected_means = np.empty((num_tracks, 4), dtype=np.float64)
        projected_covs = np.empty((num_tracks, 4, 4), dtype=np.float64)
        for i, track in enumerate(tracks):
            mean, cov = self.kalman_filter.project(track.mean, track.covariance)
            mean[2] = max(float(mean[2]), 2.0)
            mean[3] = max(float(mean[3]), 2.0)
            projected_means[i], projected_covs[i] = mean, cov

        fused_means, fused_covs = self.qam.fuse_with_kalman(projected_means, projected_covs, tracks, qam_result)

        # 2. Evidence evaluation & assignment
        if num_dets > 0:
            if getattr(self.opt, "cost_mode", "evidential") in ("linear", "normalized_linear"):
                cost = self._evaluate_evidence(
                    tracks, detections, fused_means, fused_covs, qam_result,
                    image_area, n_dets_frame, class_penalty, allow_cross_class, cands=cands
                )
                match_thresh = float(getattr(self.opt, "match_thresh", 0.70))
                matches, unmatched_tracks, unmatched_dets = assign_linear(cost, match_thresh=match_thresh)
            else:
                evidence = self._evaluate_evidence(
                    tracks, detections, fused_means, fused_covs, qam_result,
                    image_area, n_dets_frame, class_penalty, allow_cross_class, cands=cands
                )
                matches, unmatched_tracks, unmatched_dets = assign(evidence)
        else:
            matches = np.empty((0, 2), dtype=int)
            unmatched_tracks = list(range(num_tracks))
            unmatched_dets = []

        # 3. Continuous visual recovery proposals for dropped tracks
        proposals = []
        if allow_recovery:
            proposals = self._generate_recovery_proposals(
                unmatched_tracks, tracks, qam_result, fused_means, fused_covs, frame_id
            )

        return matches, unmatched_tracks, unmatched_dets, proposals

    def _evaluate_evidence(
        self,
        tracks: list,
        detections: list,
        means: np.ndarray,
        covs: np.ndarray,
        qam_result: dict,
        image_area: float,
        n_dets_frame: int,
        class_penalty: float,
        allow_cross_class: bool,
        cands: Optional[list] = None,
    ) -> np.ndarray:
        """Construct multi-modal LLR matrices or linear cost and apply cross-class constraints."""
        num_tracks = len(tracks)
        num_dets = len(detections)
        det_measurements = np.empty((num_dets, 4), dtype=np.float64)
        for j, det in enumerate(detections):
            x, y, w, h = det.tlwh
            det_measurements[j] = (x + w * 0.5, y + h * 0.5, max(w, 2.0), max(h, 2.0))

        track_cls = np.array([getattr(t, "emit_cls", getattr(t, "cls_id", 0)) for t in tracks], dtype=int)
        det_cls = np.array([getattr(d, "cls_id", 0) for d in detections], dtype=int)
        diff_cls = (track_cls[:, None] != det_cls[None, :])

        cost_mode = getattr(self.opt, "cost_mode", "evidential")
        if cost_mode in ("linear", "normalized_linear"):
            alpha = float(getattr(self.opt, "linear_alpha", 0.75 if cost_mode == "normalized_linear" else 0.5))
            metric = str(getattr(self.opt, "linear_metric", "mahalanobis"))
            is_norm = (cost_mode == "normalized_linear") or bool(getattr(self.opt, "normalized_linear", False))
            cost = linear_cost(tracks, detections, means, covs, alpha=alpha, metric=metric, normalized=is_norm)
            if not allow_cross_class:
                cost[diff_cls] = np.inf
            return cost

        cov_inflation = float(getattr(self.opt, "cov_inflation", 1.0))
        eff_covs = covs * cov_inflation if cov_inflation != 1.0 else covs
        pan_dx = float(getattr(self.opt, "_curr_pan_dx", 0.0))
        pan_dy = float(getattr(self.opt, "_curr_pan_dy", 0.0))
        if pan_dx != 0.0 or pan_dy != 0.0:
            eff_means = means.copy()
            eff_means[:, 0] += pan_dx
            eff_means[:, 1] += pan_dy
        else:
            eff_means = means
        chi2_gate = float(getattr(self.opt, "motion_chi2_gate", CHI2_2DOF_999))
        nu_val = getattr(self.opt, "motion_nu", 4.0)
        clutter_scale = float(getattr(self.opt, "clutter_scale", 1.0))
        eff_n_dets = max(float(n_dets_frame) * clutter_scale, 1.0) if clutter_scale != 1.0 else n_dets_frame

        inv_pos_cov = np.linalg.inv(eff_covs[:, :2, :2])
        pos_residual = det_measurements[None, :, :2] - eff_means[:, None, :2]
        pos_mahalanobis_sq = np.einsum("tdi,tij,tdj->td", pos_residual, inv_pos_cov, pos_residual)
        spatial_gate = (pos_mahalanobis_sq <= chi2_gate)

        use_mot = bool(getattr(self.opt, "use_motion_llr", 1))
        use_app = bool(getattr(self.opt, "use_appearance_llr", 1))

        if use_mot:
            gate_mode = getattr(self.opt, "gate_mode", "default")
            in_nominal_regime = None
            if gate_mode == "narrow_chi2":
                in_nominal_regime = np.ones(num_tracks, dtype=bool)
            elif gate_mode == "positional_conditional":
                is_active_trk = np.array([getattr(t, "state", None) == TrackState.Tracked for t in tracks], dtype=bool)
                det_maha_cls = pos_mahalanobis_sq.copy()
                det_maha_cls[diff_cls] = np.inf
                in_nominal_regime = is_active_trk & (det_maha_cls.min(axis=1) <= CHI2_2DOF_999)
            elif gate_mode == "appearance_verified_conditional":
                is_active_trk = np.array([getattr(t, "state", None) == TrackState.Tracked for t in tracks], dtype=bool)
                det_maha_cls = pos_mahalanobis_sq.copy()
                det_maha_cls[diff_cls] = np.inf
                if use_app:
                    t_f_nom = _extract_feature_matrix(tracks, "smooth_feat")
                    d_f_nom = _extract_feature_matrix(detections, "curr_feat")
                    if t_f_nom.shape[1] == d_f_nom.shape[1] and t_f_nom.shape[1] > 1:
                        tau_nom = float(getattr(self.opt, "tau_nom", 0.70))
                        det_maha_cls[(t_f_nom @ d_f_nom.T) < tau_nom] = np.inf
                in_nominal_regime = is_active_trk & (det_maha_cls.min(axis=1) <= CHI2_2DOF_999)
            elif cands:
                is_active_trk = np.array([getattr(t, "state", None) == TrackState.Tracked for t in tracks], dtype=bool)
                cand_xy = np.array([[c.tlwh[0] + 0.5 * c.tlwh[2], c.tlwh[1] + 0.5 * c.tlwh[3]] for c in cands], dtype=np.float64)
                cand_cls = np.array([getattr(c, "cls_id", 0) for c in cands], dtype=int)
                cand_res = cand_xy[None, :, :] - eff_means[:, None, :2]
                cand_maha_sq = np.einsum("tdi,tij,tdj->td", cand_res, inv_pos_cov, cand_res)
                cand_maha_sq[track_cls[:, None] != cand_cls[None, :]] = np.inf
                if use_app:
                    t_f_nom = _extract_feature_matrix(tracks, "smooth_feat")
                    c_f_nom = _extract_feature_matrix(cands, "curr_feat")
                    if t_f_nom.shape[1] == c_f_nom.shape[1] and t_f_nom.shape[1] > 1:
                        tau_nom = float(getattr(self.opt, "tau_nom", 0.70))
                        cand_maha_sq[(t_f_nom @ c_f_nom.T) < tau_nom] = np.inf
                in_nominal_regime = is_active_trk & (cand_maha_sq.min(axis=1) <= CHI2_2DOF_999)
            use_two_regime = getattr(self.opt, "use_two_regime_gate", None)
            adapt_pan_inf = float(getattr(self.opt, "adaptive_pan_inflation", 1.0))
            llr_mot = motion_llr(
                eff_means,
                eff_covs,
                det_measurements,
                image_area=image_area,
                n_dets=eff_n_dets,
                chi2_gate=chi2_gate,
                nu=nu_val,
                stationary_indicator=in_nominal_regime,
                use_two_regime_gate=use_two_regime,
                disable_tau_mot=bool(getattr(self.opt, "disable_tau_mot", False)),
                adaptive_pan_inflation=adapt_pan_inf,
            )
            if in_nominal_regime is not None:
                llr_mot[in_nominal_regime[:, None] & (~spatial_gate)] = -np.inf
            if int(getattr(self.opt, "num_classes", 5)) == 1:
                from ..ops.association_ops import iou_distance
                iou_mat = 1.0 - np.asarray(iou_distance(tracks, detections), dtype=np.float64)
                iou_ev = 8.0 * (iou_mat - 0.30)
                llr_mot = np.where(
                    iou_mat >= 0.30,
                    np.where(np.isfinite(llr_mot), 0.35 * llr_mot + 0.65 * iou_ev, iou_ev),
                    llr_mot,
                )
        else:
            # Gating-only: preserve spatial sanity, zero kinematic penalty
            llr_mot = np.where(spatial_gate, 0.0, -np.inf)

        if use_app:
            t_feats = _extract_feature_matrix(tracks, "smooth_feat")
            d_feats = _extract_feature_matrix(detections, "curr_feat")
            is_active = np.array([getattr(t, "state", None) == TrackState.Tracked for t in tracks], dtype=bool)
            local_gate = (~diff_cls) & np.isfinite(llr_mot)

            cue_mode = getattr(self.opt, "cue_weighting_mode", "vmf_dynamic")
            k_max = float(getattr(self.opt, "kappa_max", 300.0) or 300.0)
            det_scores_weight = None

            if cue_mode == "confidence_adaptive":
                # Detection-Confidence Adaptive Weighting: w_app,j = c_j, kappa = kappa_max
                kappas = np.full(num_tracks, k_max, dtype=np.float64)
                det_scores = np.array([float(getattr(d, "score", 1.0)) for d in detections], dtype=np.float64)
                det_scores_weight = np.clip(det_scores, 0.0, 1.0)[None, :]
            elif cue_mode == "cosine_ema":
                # Cosine-Consistency Linear EMA Weighting: kappa_i = r_bar_i * kappa_max
                kappas = np.array([float(getattr(t, "app_Rbar", 0.85)) * k_max for t in tracks], dtype=np.float64)
            elif cue_mode == "softmax_entropy":
                # Softmax-Entropy Adaptive Weighting: w_ent,i = 1 - H_i / ln(K)
                dot = t_feats @ d_feats.T
                tau = float(getattr(self.opt, "entropy_tau", 0.1))
                if num_dets > 1:
                    exp_dot = np.exp(np.clip((dot - np.max(dot, axis=1, keepdims=True)) / max(tau, 1e-6), -20.0, 20.0))
                    p = exp_dot / np.maximum(exp_dot.sum(axis=1, keepdims=True), 1e-12)
                    ent = -np.sum(p * np.log(p + 1e-12), axis=1)
                    w_ent = np.clip(1.0 - ent / np.log(max(num_dets, 2)), 0.0, 1.0)
                else:
                    w_ent = np.ones(num_tracks, dtype=np.float64)
                kappas = k_max * w_ent
            elif cue_mode == "fixed_kappa" or not bool(getattr(self.opt, "use_dynamic_kappa", 1)):
                fixed_k = float(getattr(self.opt, "fixed_kappa", 300.0) or 300.0)
                kappas = np.full(num_tracks, fixed_k, dtype=np.float64)
            else:  # "vmf_dynamic" (SPEAR online vMF concentration)
                use_dyn_kappa = bool(getattr(self.opt, "use_dynamic_kappa", 1))
                fixed_k = getattr(self.opt, "fixed_kappa", None)
                if use_dyn_kappa and bool(getattr(self.opt, "use_instantaneous_rbar", False)) and num_dets > 0:
                    raw_cos = t_feats @ d_feats.T
                    nom_gate = (~diff_cls) & spatial_gate
                    masked_maha = np.where(nom_gate, pos_mahalanobis_sq, np.inf)
                    min_nom_maha = np.min(masked_maha, axis=1)
                    has_close_kin = min_nom_maha <= CHI2_2DOF_999
                    j_kin = np.argmin(masked_maha, axis=1)
                    rho_kin = np.where(has_close_kin, np.clip(raw_cos[np.arange(num_tracks), j_kin], 0.0, 1.0), 0.0)
                    cos_gate = np.where(local_gate, raw_cos, -1.0)
                    rho_max = np.clip(np.max(cos_gate, axis=1), 0.0, 1.0)
                    r_bars = np.array([float(getattr(t, "app_Rbar", 0.4)) for t in tracks], dtype=np.float64)
                    r_clean = np.where(rho_max >= 0.58, np.maximum(r_bars, 0.85 * rho_max), r_bars)
                    r_eff = np.where((has_close_kin & (rho_kin < 0.75)) | (rho_max < 0.68), 0.0, r_clean)
                    kappas = vmf_kappa(r_eff, t_feats.shape[1], use_dynamic=True, kappa_max=k_max)
                else:
                    kappas = np.array([t.kappa(use_dynamic=use_dyn_kappa, kappa_max=k_max, fixed_kappa=fixed_k) for t in tracks], dtype=np.float64)

            lambda_glrt = float(getattr(self.opt, "lambda_glrt", 0.8))
            raw_app = appearance_llr(
                t_feats, d_feats, kappas,
                track_cls=track_cls, det_cls=det_cls,
                local_gate=local_gate, is_active=is_active,
                lambda_glrt=lambda_glrt,
            )
            w_app = float(getattr(self.opt, "w_app", 0.08))
            llr_app = w_app * raw_app
            if det_scores_weight is not None:
                llr_app = llr_app * det_scores_weight
        else:
            llr_app = None

        evidence = fuse_evidence(llr_mot, llr_app)

        if allow_cross_class and class_penalty > 0.0:
            confirmed = np.array([
                getattr(t, "state", None) == TrackState.Tracked and getattr(t, "is_activated", False) and getattr(t, "track_len", 0) >= 2
                for t in tracks
            ], dtype=bool)
            evidence = np.where(diff_cls & (~confirmed[:, None]), -np.inf, evidence)
            evidence = np.where(diff_cls & confirmed[:, None], evidence - class_penalty, evidence)
        elif not allow_cross_class:
            evidence = np.where(diff_cls, -np.inf, evidence)

        return evidence

    def _generate_recovery_proposals(
        self,
        unmatched_tracks: List[int],
        tracks: list,
        qam_result: dict,
        means: np.ndarray,
        covs: np.ndarray,
        frame_id: int,
    ) -> list:
        """Create pseudo-measurement proposals from continuous QAM queries for missed tracks."""
        proposals = []
        for idx in unmatched_tracks:
            track = tracks[idx]
            qam_info = qam_result.get(id(track))
            if qam_info is None or (frame_id - track.start_frame) < 2:
                continue

            pred_xy, peak_val, sidelobe_peak = qam_info
            target_size = float(np.sqrt(max(track.tlwh[2] * track.tlwh[3], 1.0)))
            beta = float(getattr(self.opt, "qam_beta", 5.0)) if hasattr(self, "opt") and self.opt else 5.0
            pseudo_var = float(self.qam.pseudo_variance(peak_val, sidelobe_peak, target_size, beta=beta))

            spatial_cov = covs[idx, :2, :2] + np.eye(2) * pseudo_var
            residual = np.asarray(pred_xy) - means[idx, :2]
            if float(residual @ np.linalg.inv(spatial_cov) @ residual) > CHI2_2DOF_999:
                continue

            pseudo_xywh = np.array([pred_xy[0], pred_xy[1], means[idx, 2], means[idx, 3]], dtype=np.float32)
            base_var = (self.kalman_filter._std_weight_position * max(means[idx, 3], 1.0)) ** 2
            proposals.append((track, pseudo_xywh, pseudo_var / base_var))

        return proposals
