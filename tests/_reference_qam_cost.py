"""Reference (original, loop-based) QAM candidate cost from AERO-MOT, used only in tests."""
import numpy as np
from perception.tracker.ops import track_ops as ops


def build_qam_cost_reference(t_list, d_list, *, use_appearance_motion, global_search, r_min_val, q_gate, shift_w, margin_val):
    """Verbatim copy of AERO-MOT's nested ``_build_qam_cost`` (loop version)."""
    if not t_list or not d_list:
        return np.ones((len(t_list), len(d_list)), dtype=np.float64)
    iou_d = ops.matching.iou_distance(t_list, d_list)
    if not use_appearance_motion:
        return iou_d

    n_t, n_d = len(t_list), len(d_list)
    cos_sim = np.zeros((n_t, n_d), dtype=np.float64)
    for ti, trk in enumerate(t_list):
        tf = getattr(trk, "smooth_feat", None)
        if tf is None or np.linalg.norm(tf) <= 1e-6:
            continue
        for di, det in enumerate(d_list):
            df = getattr(det, "curr_feat", None)
            if df is not None and df.size == tf.size:
                cos_sim[ti, di] = float(np.dot(tf, df))

    if global_search:
        # Global WxH search: match purely by global visual correlation across entire frame
        return 1.0 - cos_sim

    # Local 3-sigma search window Omega_i + Bayesian log-posterior S_i(x) (Eqs. 5-7)
    gated = iou_d.copy()
    max_grid_px = float(r_min_val) * 4.0
    for ti, trk in enumerate(t_list):
        tcx = trk.tlwh[0] + 0.5 * trk.tlwh[2]
        tcy = trk.tlwh[1] + 0.5 * trk.tlwh[3]
        t_scale = float(np.sqrt(max(trk.tlwh[2] * trk.tlwh[3], 1.0)))
        rad = max(max_grid_px, 1.8 * t_scale)
        in_win = []
        s_vals = []
        dists_in_win = []
        for di, det in enumerate(d_list):
            dcx = det.tlwh[0] + 0.5 * det.tlwh[2]
            dcy = det.tlwh[1] + 0.5 * det.tlwh[3]
            dist = float(np.hypot(dcx - tcx, dcy - tcy))
            if dist > rad:
                gated[ti, di] = 1.0
            else:
                in_win.append(di)
                dists_in_win.append(dist)
                s_val = cos_sim[ti, di] - (dist ** 2) / (2.0 * (0.85 * rad) ** 2)
                s_vals.append(s_val)

        for k_idx, di in enumerate(in_win):
            s_val = s_vals[k_idx]
            dist = dists_in_win[k_idx]
            if q_gate > 0.0 and s_val < q_gate and shift_w > 0.0:
                gated[ti, di] = 1.0
            elif shift_w > 0.0:
                d_vis = 0.50 * (dist / max(rad, 1.0)) + 0.50 * (1.0 - max(cos_sim[ti, di], 0.0))
                gated[ti, di] = (1.0 - shift_w) * iou_d[ti, di] + shift_w * d_vis
            else:
                if cos_sim[ti, di] > (0.30 + margin_val):
                    gated[ti, di] = 0.70 * iou_d[ti, di] + 0.30 * (1.0 - cos_sim[ti, di])
                elif q_gate > 0.0 and cos_sim[ti, di] < (0.10 + q_gate):
                    gated[ti, di] = min(1.0, iou_d[ti, di] + 0.20)
        if len(s_vals) >= 2 and margin_val > 0.0:
            s_sorted = np.sort(s_vals)
            win_margin = float(s_sorted[-1] - s_sorted[-2])
            if win_margin < 0.08 * (margin_val / 0.25) * (1.0 - 0.5 * shift_w):
                for di in in_win:
                    gated[ti, di] = min(1.0, gated[ti, di] + 0.12 * (1.0 - 0.4 * shift_w))
    return gated

