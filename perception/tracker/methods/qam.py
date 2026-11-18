# AERO-MOT 🚁 AGPL-3.0 License
# Copyright (C) 2026 AERO-MOT Authors

"""Query Appearance-Motion (QAM): Bayesian Visual-Kinematic Observer for Aerial MOT."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..ops.geometry import centers_of


class SpatialProjection:
    """Projects 2D coordinates between image space and feature map grid."""

    def __init__(self, stride: float, ratio_x: float, ratio_y: float, pad_w: float = 0.0, pad_h: float = 0.0):
        self.stride = float(stride)
        self.ratio_x = float(ratio_x)
        self.ratio_y = float(ratio_y)
        self.pad_w = float(pad_w)
        self.pad_h = float(pad_h)

    def to_map(self, coords: np.ndarray | torch.Tensor) -> np.ndarray:
        points = np.asarray(coords, float)
        return np.stack(
            [(points[:, 0] * self.ratio_x + self.pad_w) / self.stride,
             (points[:, 1] * self.ratio_y + self.pad_h) / self.stride],
            axis=1,
        )

    def to_image(self, coords: np.ndarray | torch.Tensor) -> np.ndarray:
        points = np.asarray(coords, float)
        return np.stack(
            [(points[:, 0] * self.stride - self.pad_w) / self.ratio_x,
             (points[:, 1] * self.stride - self.pad_h) / self.ratio_y],
            axis=1,
        )


def orig_to_map(coords, stride, rx, ry, pw=0.0, ph=0.0):
    return SpatialProjection(stride, rx, ry, pw, ph).to_map(coords)


def map_to_orig(coords, stride, rx, ry, pw=0.0, ph=0.0):
    return SpatialProjection(stride, rx, ry, pw, ph).to_image(coords)


@torch.no_grad()
def normalize_dense(features: torch.Tensor) -> torch.Tensor:
    """L2-normalizes dense feature embeddings along the channel dimension."""
    return F.normalize(features, dim=0, eps=1e-6)


@torch.no_grad()
def sample_dense(dense_features: torch.Tensor, map_coords: torch.Tensor) -> torch.Tensor:
    """Bilinearly samples continuous feature vectors at sub-pixel coordinates."""
    if map_coords.numel() == 0:
        return dense_features.new_zeros((0, dense_features.shape[0]))

    channels, height, width = dense_features.shape
    normalized_grid = torch.stack(
        [map_coords[:, 0] / max(width - 1, 1) * 2 - 1,
         map_coords[:, 1] / max(height - 1, 1) * 2 - 1],
        dim=1,
    ).view(1, -1, 1, 2)

    sampled = F.grid_sample(dense_features[None], normalized_grid, mode="bilinear", align_corners=True)
    return sampled.view(channels, -1).t().contiguous()


def _compute_log_posterior_surface(
    templates: torch.Tensor,
    dense_features: torch.Tensor,
    prior_centers: torch.Tensor,
    search_radius: int,
    prior_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes spatial log-posterior density:
      log p(x | q, F, mu) = Visual Log-Likelihood + Kinematic Log-Prior
                          = q^T f(x) - ||x - mu||^2 / (2 * sigma^2)
    """
    num_tracks = templates.shape[0]
    channels, height, width = dense_features.shape
    window_size = 2 * search_radius + 1
    device = dense_features.device

    prior_x = prior_centers[:, 0].round().long().clamp(0, width - 1)
    prior_y = prior_centers[:, 1].round().long().clamp(0, height - 1)
    offsets = torch.arange(-search_radius, search_radius + 1, device=device)
    grid_x = (prior_x[:, None] + offsets).clamp(0, width - 1)
    grid_y = (prior_y[:, None] + offsets).clamp(0, height - 1)

    # 1. Visual Log-Likelihood via cuBLAS GEMM (q^T f(x))
    visual_log_likelihood = torch.mm(templates, dense_features.view(channels, -1)).view(num_tracks, height, width)
    visual_log_likelihood = visual_log_likelihood.gather(1, grid_y[:, :, None].expand(num_tracks, window_size, width))
    visual_log_likelihood = visual_log_likelihood.gather(2, grid_x[:, None, :].expand(num_tracks, window_size, window_size))

    # 2. Kinematic Gaussian Log-Prior (-||x - mu||^2 / (2 * sigma^2))
    distance_sq = (grid_x.float() - prior_x[:, None])[:, None, :] ** 2 + (grid_y.float() - prior_y[:, None])[:, :, None] ** 2
    kinematic_log_prior = -distance_sq / (2.0 * prior_sigma ** 2)

    # 3. Combined Bayesian Log-Posterior Surface S_i(x)
    log_posterior = visual_log_likelihood + kinematic_log_prior
    return log_posterior, grid_x, grid_y


def _estimate_map_mode(
    log_posterior: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimates Maximum A Posteriori (MAP) Mode on the discrete spatial grid:
      x* = argmax log p(x | q, F, mu)
    """
    num_tracks = log_posterior.shape[0]
    device = log_posterior.device

    flat_map_indices = log_posterior.flatten(start_dim=1).argmax(dim=1)
    row_mode = flat_map_indices // window_size
    col_mode = flat_map_indices % window_size
    track_indices = torch.arange(num_tracks, device=device)
    map_peak_scores = log_posterior[track_indices, row_mode, col_mode]

    discrete_x = grid_x.gather(1, col_mode[:, None]).squeeze(1).float()
    discrete_y = grid_y.gather(1, row_mode[:, None]).squeeze(1).float()
    map_centers = torch.stack([discrete_x, discrete_y], dim=1)

    return map_centers, map_peak_scores, row_mode, col_mode


def _measure_sidelobe_peak(
    log_posterior: torch.Tensor,
    row_mode: torch.Tensor,
    col_mode: torch.Tensor,
    map_peak_scores: torch.Tensor,
    window_size: int,
    sidelobe_radius: int,
) -> torch.Tensor:
    """
    Suppresses MAP mode neighborhood and extracts the highest secondary sidelobe score
    to compute the distinctiveness margin Delta S_i = S(x*) - max_{x notin N} S(x).
    """
    device = log_posterior.device
    window_coords = torch.arange(window_size, device=device)
    is_in_mode_neighborhood = (
        (window_coords[None, :, None] - row_mode[:, None, None]).abs() <= sidelobe_radius
    ) & (
        (window_coords[None, None, :] - col_mode[:, None, None]).abs() <= sidelobe_radius
    )

    suppressed_surface = log_posterior.masked_fill(is_in_mode_neighborhood, float("-inf"))
    sidelobe_candidates = suppressed_surface.flatten(start_dim=1).max(dim=1).values
    return torch.where(torch.isfinite(sidelobe_candidates), sidelobe_candidates, map_peak_scores)


@torch.no_grad()
def query_latent_field(
    templates: torch.Tensor,
    dense_features: torch.Tensor,
    prior_centers: torch.Tensor,
    search_radius: int = 16,
    prior_sigma: float = 15.0,
    return_peak2: bool = True,
    sidelobe_radius: int = 2,
):
    """
    Bayesian Visual-Kinematic Observer:
      Stage 1: Spatial log-posterior surface S_i(x) = Visual Log-Likelihood + Kinematic Prior
      Stage 2: Discrete Maximum A Posteriori (MAP) mode seeking on spatial feature grid
      Stage 3: Sidelobe suppression for distinctiveness margin Delta S_i
    """
    if templates.shape[0] == 0:
        empty = templates.new_zeros(0)
        return (templates.new_zeros(0, 2), empty, empty) if return_peak2 else (templates.new_zeros(0, 2), empty)

    window_size = 2 * search_radius + 1

    # Stage 1: Spatial Log-Posterior Density S_i(x)
    log_posterior, grid_x, grid_y = _compute_log_posterior_surface(
        templates, dense_features, prior_centers, search_radius, prior_sigma
    )

    # Stage 2: Discrete MAP Mode Seeking
    map_centers, map_peaks, row_mode, col_mode = _estimate_map_mode(
        log_posterior, grid_x, grid_y, window_size
    )

    if not return_peak2:
        return map_centers, map_peaks

    # Stage 3: Sidelobe Suppression & Distinctiveness Margin
    sidelobe_peaks = _measure_sidelobe_peak(
        log_posterior, row_mode, col_mode, map_peaks, window_size, sidelobe_radius
    )
    return map_centers, map_peaks, sidelobe_peaks


predict_centers = query_latent_field


class QamHandle:
    """Asynchronous synchronization handle for GPU visual queries."""
    __slots__ = ("gpu_data", "cpu_data", "event", "_keepalive")

    def __init__(self, gpu_data, cpu_data=None, event=None, keepalive=None):
        self.gpu_data = gpu_data
        self.cpu_data = cpu_data
        self.event = event
        self._keepalive = keepalive

    def collect(self) -> np.ndarray:
        if self.event is not None:
            self.event.synchronize()
            self._keepalive = None
            return self.cpu_data.numpy()
        if hasattr(self.gpu_data, "cpu"):
            return self.gpu_data.cpu().numpy()
        return np.asarray(self.gpu_data)


_ASYNC_STREAMS: dict[torch.device, torch.cuda.Stream] = {}
_PINNED_BUFFERS: dict[tuple, torch.Tensor] = {}


@torch.no_grad()
def query_latent_field_async(
    templates: torch.Tensor,
    dense_features: torch.Tensor,
    prior_centers: torch.Tensor,
    search_radius: int = 16,
    prior_sigma: float = 15.0,
    return_peak2: bool = True,
) -> QamHandle:
    """Executes non-blocking visual queries on a dedicated CUDA stream."""
    device = dense_features.device
    if device.type != "cuda":
        centers, peaks, sidelobes = query_latent_field(
            templates, dense_features, prior_centers,
            search_radius=search_radius, prior_sigma=prior_sigma, return_peak2=return_peak2
        )
        packed = torch.cat([centers, peaks[:, None], sidelobes[:, None]], dim=1)
        return QamHandle(packed)

    stream = _ASYNC_STREAMS.setdefault(device, torch.cuda.Stream(device=device))
    stream.wait_stream(torch.cuda.current_stream(device))
    event = torch.cuda.Event()

    with torch.cuda.stream(stream):
        centers, peaks, sidelobes = query_latent_field(
            templates, dense_features, prior_centers,
            search_radius=search_radius, prior_sigma=prior_sigma, return_peak2=return_peak2
        )
        gpu_packed = torch.cat([centers, peaks[:, None], sidelobes[:, None]], dim=1).contiguous()
        num_tracks, num_columns = gpu_packed.shape

        pinned_buffer = _PINNED_BUFFERS.setdefault(
            (device, num_columns),
            torch.empty((max(num_tracks, 256), num_columns), dtype=gpu_packed.dtype, pin_memory=True),
        )
        if pinned_buffer.shape[0] < num_tracks:
            pinned_buffer = torch.empty((num_tracks + 128, num_columns), dtype=gpu_packed.dtype, pin_memory=True)
            _PINNED_BUFFERS[(device, num_columns)] = pinned_buffer

        cpu_out = pinned_buffer[:num_tracks]
        cpu_out.copy_(gpu_packed, non_blocking=True)
        event.record(stream)

    return QamHandle(gpu_packed, cpu_out, event, keepalive=(templates, prior_centers))


predict_centers_launch = query_latent_field_async


class QueryAppearanceMotion:
    """Continuous Bayesian Visual-Kinematic Observer for FalconTracker."""

    def __init__(self, opt=None):
        self.opt = opt
        self.spatial_sigma = float(getattr(opt, "qam_spatial_sigma", 15.0)) if opt else 15.0
        self.roi_radius = int(getattr(opt, "qam_roi_radius", 16)) if opt else 16
        self.dense_raw: torch.Tensor | None = None
        self.dense_hat: torch.Tensor | None = None
        self.projection: SpatialProjection | None = None

    @property
    def transform(self):
        if self.projection is None:
            return None
        return (self.projection.stride, self.projection.ratio_x, self.projection.ratio_y, self.projection.pad_w, self.projection.pad_h)

    def reset(self) -> None:
        self.dense_raw = None
        self.dense_hat = None
        self.projection = None

    def set_dense(
        self,
        dense: torch.Tensor | None,
        stride: float,
        rx: float,
        ry: float,
        pw: float = 0.0,
        ph: float = 0.0,
        dense_normed: torch.Tensor | None = None,
    ) -> int | None:
        if dense is None:
            self.reset()
            return None
        if dense.ndim == 4 and dense.shape[0] == 1:
            dense = dense.squeeze(0)
        if dense_normed is not None and dense_normed.ndim == 4 and dense_normed.shape[0] == 1:
            dense_normed = dense_normed.squeeze(0)
        self.dense_raw = dense
        self.dense_hat = dense_normed if dense_normed is not None else normalize_dense(dense)
        self.projection = SpatialProjection(stride, rx, ry, pw, ph)
        return int(dense.shape[0])

    def sample_detection_templates(self, detections) -> None:
        dense = self.dense_hat if self.dense_hat is not None else self.dense_raw
        if dense is None or not detections or self.projection is None:
            return

        map_coords = torch.from_numpy(self.projection.to_map(centers_of(detections))).to(dense.device).float()
        features = sample_dense(dense, map_coords).cpu().numpy()
        feature_norms = np.linalg.norm(features, axis=1, keepdims=True)
        normalized_features = np.where(feature_norms > 0, features / np.maximum(feature_norms, 1e-12), features)

        for det, feat in zip(detections, normalized_features):
            det.template = feat

    def launch(self, tracks, detections=None) -> tuple[QamHandle | None, list]:
        if self.dense_hat is not None and self.projection is not None:
            active_tracks = [t for t in tracks if t.template is not None and t.mean is not None]
            if not active_tracks:
                return None, []

            templates = torch.from_numpy(np.stack([t.template for t in active_tracks])).to(self.dense_hat.device, non_blocking=True).float()
            prior_coords = torch.from_numpy(self.projection.to_map(np.stack([t.mean[:2] for t in active_tracks]))).to(self.dense_hat.device, non_blocking=True).float()

            global_search = bool(getattr(self.opt, "qam_global_search", False)) if self.opt else False
            eff_radius = max(int(self.dense_hat.shape[1]), int(self.dense_hat.shape[2])) if global_search else self.roi_radius
            eff_sigma = 1e6 if global_search else self.spatial_sigma

            handle = query_latent_field_async(
                templates,
                self.dense_hat,
                prior_coords,
                search_radius=eff_radius,
                prior_sigma=eff_sigma,
                return_peak2=True,
            )
            return handle, active_tracks

        return None, []

    def collect(self, handle: QamHandle | None, tracks: list) -> dict[int, tuple[np.ndarray, float, float]]:
        if handle is None or not tracks:
            return {}

        packed_results = handle.collect()
        if self.projection is not None:
            image_coords = self.projection.to_image(packed_results[:, :2])
        else:
            image_coords = packed_results[:, :2]

        results = {}
        for idx, track in enumerate(tracks):
            track.pred_xy = image_coords[idx]
            results[id(track)] = (
                image_coords[idx].astype(np.float64),
                float(packed_results[idx, 2]),
                float(packed_results[idx, 3]),
            )
        return results

    @staticmethod
    def visual_observation_variance(
        map_peak: float,
        sidelobe_peak: float,
        track_size: float | np.ndarray,
        beta: float = 5.0,
        cov_mode: str = "margin_scaled",
    ) -> float | np.ndarray:
        """
        Calibrates visual observation covariance from distinctiveness margin:
          R_{q, i} = s_i^2 * exp(-beta * Delta S_i)
        where Delta S_i = S(x*) - max_{x notin N} S(x).
        """
        s_sq = np.asarray(track_size, float) ** 2
        if cov_mode == "fixed":
            return s_sq * 0.08
        if cov_mode == "peak_scaled":
            peak_term = np.clip(float(map_peak) * float(beta), 0.0, 20.0)
            return s_sq * np.exp(-peak_term)
        distinctiveness_margin = np.clip((float(map_peak) - float(sidelobe_peak)) * float(beta), 0.0, 20.0)
        return s_sq * np.exp(-distinctiveness_margin)

    pseudo_variance = visual_observation_variance

    @staticmethod
    def corr_confidence(map_peak: float, sidelobe_peak: float) -> float:
        """Visual correlation distinctiveness margin: max(Delta S_i, 0)."""
        return np.clip(np.asarray(map_peak, float) - np.asarray(sidelobe_peak, float), 0.0, None)

    def fuse_with_kalman(
        self,
        prior_means: np.ndarray,
        prior_covs: np.ndarray,
        tracks: list,
        qam_measurements: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Closed-Form Gaussian Precision / Information Fusion:
          P_{k|k}^{-1} = P_{k|k-1}^{-1} + R_q^{-1}
          x_{k|k} = P_{k|k} * (P_{k|k-1}^{-1} x_{k|k-1} + R_q^{-1} x_tilde)
        """
        fused_means = np.asarray(prior_means, float).copy()
        fused_covs = np.asarray(prior_covs, float).copy()
        beta = float(getattr(self.opt, "qam_beta", 5.0)) if hasattr(self, "opt") and self.opt else 5.0
        margin_thresh = float(getattr(self.opt, "qam_distinctiveness_margin", 0.25)) if hasattr(self, "opt") and self.opt else 0.25
        cov_mode = str(getattr(self.opt, "qam_cov_mode", "margin_scaled")) if hasattr(self, "opt") and self.opt else "margin_scaled"

        for idx, track in enumerate(tracks):
            measurement = qam_measurements.get(id(track))
            if measurement is None:
                continue

            map_coords, map_peak, sidelobe_peak = measurement
            if cov_mode == "margin_scaled" and (float(map_peak) - float(sidelobe_peak)) < margin_thresh:
                continue

            track_scale = np.sqrt(max(track.tlwh[2] * track.tlwh[3], 1.0))
            visual_variance = float(self.visual_observation_variance(map_peak, sidelobe_peak, track_scale, beta=beta, cov_mode=cov_mode))

            kf_spatial_cov = fused_covs[idx, :2, :2]
            spatial_cov = kf_spatial_cov + np.eye(2) * visual_variance
            residual = np.asarray(map_coords) - fused_means[idx, :2]
            if float(residual @ np.linalg.inv(spatial_cov) @ residual) > 13.816:
                continue

            kf_precision = np.linalg.inv(kf_spatial_cov + 1e-6 * np.eye(2))
            visual_precision = np.eye(2) / visual_variance

            posterior_cov = np.linalg.inv(kf_precision + visual_precision)
            posterior_mean = posterior_cov @ (kf_precision @ fused_means[idx, :2] + visual_precision @ np.asarray(map_coords))

            fused_means[idx, :2] = posterior_mean
            fused_covs[idx, :2, :2] = posterior_cov

        return fused_means, fused_covs
