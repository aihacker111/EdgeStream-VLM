# AERO-MOT 🚁 AGPL-3.0 License
# Copyright (C) 2026 AERO-MOT Authors

import numpy as np
import scipy.linalg


def _isotropic_scale(w, h):
    """Isotropic root-mean-square bounding-box scale with a 4 px localization floor."""
    return 0.7071067811865475 * np.hypot(w, h) + 4.0


def _kinetic_scale(mean, gamma_kin: float = 6.0):
    """State-dependent kinetic diffusion scale: sqrt(s_box^2 + gamma_kin^2 * ||v||_2^2)."""
    if mean.ndim == 1:
        s_box = _isotropic_scale(mean[2], mean[3])
        v_norm = np.hypot(mean[4], mean[5])
        return np.hypot(s_box, gamma_kin * v_norm)
    s_box = _isotropic_scale(mean[:, 2], mean[:, 3])
    v_norm = np.hypot(mean[:, 4], mean[:, 5])
    return np.hypot(s_box, gamma_kin * v_norm)


class KalmanFilter(object):
    """
    8-dimensional state-space Kalman filter for 2D bounding-box trajectories:
        x_t = [x, y, w, h, vx, vy, vw, vh]^T

    Incorporates two first-principles state-space filtering formulations for aerial MOT:
      1. Isotropic & State-Dependent Kinetic Process Covariance:
         Replaces anisotropic width-tied horizontal noise with an isotropic spatial scale
         s_i = sqrt((w_i^2 + h_i^2)/2) + s_0 and state-dependent kinetic diffusion
         s_tilde_i(x_{t-1}) = sqrt(s_i^2 + gamma_kin^2 * (vx^2 + vy^2)), scaling transition
         uncertainty with trajectory speed while keeping stationary targets razor-sharp.
      2. Coupled Discrete-Time Acceleration Impulse Covariance:
         Models abrupt unmodeled planar accelerations a_t via the exact discrete-time
         impulse input matrix G = [I_2; I_2], injecting Delta P = hat{sigma}_a^2 (G G^T)
         with moment-matched impulse variance hat{sigma}_a^2 = max(0, 0.5*||nu_pos||^2 - 0.5*tr(S_pos)),
         simultaneously adapting position and velocity gains in a single frame.
    """

    _update_mat_static = np.eye(4, 8)

    def __init__(self, adaptive: bool = True):
        ndim, dt = 4, 1.0
        self.adaptive = bool(adaptive)

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        if self.adaptive:
            sx = sy = _isotropic_scale(measurement[2], measurement[3])
        else:
            sx, sy = measurement[2], measurement[3]
        std = [
            2 * self._std_weight_position * sx,
            2 * self._std_weight_position * sy,
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * sx,
            10 * self._std_weight_velocity * sy,
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        if self.adaptive:
            sx = sy = _kinetic_scale(mean)
            vel_mult = 2.0
        else:
            sx, sy = mean[2], mean[3]
            vel_mult = 1.0
        std_pos = [
            self._std_weight_position * sx,
            self._std_weight_position * sy,
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            vel_mult * self._std_weight_velocity * sx,
            vel_mult * self._std_weight_velocity * sy,
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        if self.adaptive:
            sx = sy = _isotropic_scale(float(mean[2]), float(mean[3]))
        else:
            sx, sy = float(mean[2]), float(mean[3])
        std = [
            self._std_weight_position * sx,
            self._std_weight_position * sy,
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        if self.adaptive:
            sx = sy = _kinetic_scale(mean)
            vel_mult = 2.0
        else:
            sx, sy = mean[:, 2], mean[:, 3]
            vel_mult = 1.0
        std_pos = [
            self._std_weight_position * sx,
            self._std_weight_position * sy,
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            vel_mult * self._std_weight_velocity * sx,
            vel_mult * self._std_weight_velocity * sy,
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        N = mean.shape[0]
        motion_cov = np.zeros((N, 8, 8), dtype=sqr.dtype)
        _di = np.arange(8)
        motion_cov[:, _di, _di] = sqr

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    @staticmethod
    def _adapt_cov_single(mean, covariance, measurement, R_cov, adaptive: bool = True):
        projected_mean = np.dot(KalmanFilter._update_mat_static, mean)
        innovation = measurement - projected_mean
        if not adaptive:
            return projected_mean, innovation, covariance
        nom_pos_var = 0.5 * (covariance[0, 0] + covariance[1, 1] + R_cov[0, 0] + R_cov[1, 1])
        innov_pos_var = 0.5 * (innovation[0] ** 2 + innovation[1] ** 2)
        sigma_a_sq = max(0.0, float(innov_pos_var - nom_pos_var))
        cov_adapted = covariance.copy()
        if sigma_a_sq > 0.0:
            for d_idx in (0, 1):
                cov_adapted[d_idx, d_idx] += sigma_a_sq
                cov_adapted[d_idx, d_idx + 4] += sigma_a_sq
                cov_adapted[d_idx + 4, d_idx] += sigma_a_sq
                cov_adapted[d_idx + 4, d_idx + 4] += sigma_a_sq
        return projected_mean, innovation, cov_adapted

    def multi_update(self, mean, covariance, measurements):
        N = mean.shape[0]
        if N == 0:
            return mean.copy(), covariance.copy()

        if self.adaptive:
            sx = sy = _isotropic_scale(mean[:, 2], mean[:, 3])
        else:
            sx, sy = mean[:, 2], mean[:, 3]
        std_x = self._std_weight_position * sx
        std_y = self._std_weight_position * sy
        std_w = self._std_weight_position * mean[:, 2]
        std_h = self._std_weight_position * mean[:, 3]
        diag_vals = np.stack([std_x, std_y, std_w, std_h], axis=1) ** 2

        innovation_cov = np.zeros((N, 4, 4), dtype=covariance.dtype)
        idx = np.arange(4)
        innovation_cov[:, idx, idx] = diag_vals

        projected_mean = mean @ self._update_mat.T
        innovation = measurements - projected_mean

        cov_adapted = covariance.copy()
        if self.adaptive:
            nom_pos_var = 0.5 * (covariance[:, 0, 0] + covariance[:, 1, 1] + diag_vals[:, 0] + diag_vals[:, 1])
            innov_pos_var = 0.5 * (innovation[:, 0] ** 2 + innovation[:, 1] ** 2)
            sigma_a_sq = np.maximum(0.0, innov_pos_var - nom_pos_var)
            for d_idx in (0, 1):
                cov_adapted[:, d_idx, d_idx] += sigma_a_sq
                cov_adapted[:, d_idx, d_idx + 4] += sigma_a_sq
                cov_adapted[:, d_idx + 4, d_idx] += sigma_a_sq
                cov_adapted[:, d_idx + 4, d_idx + 4] += sigma_a_sq

        tmp = cov_adapted @ self._update_mat.T
        projected_cov = (self._update_mat[None, :, :] @ tmp) + innovation_cov

        b = np.transpose(tmp, (0, 2, 1))
        x = np.linalg.solve(projected_cov, b)
        kalman_gain = np.transpose(x, (0, 2, 1))

        new_mean = mean + np.einsum('ni,nki->nk', innovation, kalman_gain)
        new_covariance = cov_adapted - np.einsum(
            'nik,nkl,njl->nij', kalman_gain, projected_cov, kalman_gain)
        return new_mean, new_covariance

    def update(self, mean, covariance, measurement):
        if self.adaptive:
            sx = sy = _isotropic_scale(float(mean[2]), float(mean[3]))
        else:
            sx, sy = float(mean[2]), float(mean[3])
        std = [
            self._std_weight_position * sx,
            self._std_weight_position * sy,
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        R_cov = np.diag(np.square(std))
        projected_mean, innovation, cov_adapted = self._adapt_cov_single(
            mean, covariance, measurement, R_cov, adaptive=self.adaptive)
        projected_cov = np.linalg.multi_dot((self._update_mat, cov_adapted, self._update_mat.T)) + R_cov
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(cov_adapted, self._update_mat.T).T,
            check_finite=False).T

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = cov_adapted - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def update_nsa(self, mean, covariance, measurement, confidence):
        if self.adaptive:
            sx = sy = _isotropic_scale(float(mean[2]), float(mean[3]))
        else:
            sx, sy = float(mean[2]), float(mean[3])
        std = [
            self._std_weight_position * sx,
            self._std_weight_position * sy,
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        nsa_scale = 1.0 + 2.0 * (1.0 - float(confidence)) ** 2
        R_cov = np.diag(np.square(std)) * nsa_scale
        projected_mean, innovation, cov_adapted = self._adapt_cov_single(
            mean, covariance, measurement, R_cov, adaptive=self.adaptive)
        projected_cov_nsa = np.linalg.multi_dot((self._update_mat, cov_adapted, self._update_mat.T)) + R_cov

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov_nsa, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(cov_adapted, self._update_mat.T).T,
            check_finite=False).T

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = cov_adapted - np.linalg.multi_dot(
            (kalman_gain, projected_cov_nsa, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')
