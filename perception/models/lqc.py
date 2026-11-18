# AERO-MOT 🚁 AGPL-3.0 License
# Copyright (C) 2026 AERO-MOT Authors

# aeromot/nn/falcon_yolo/lqcv5.py
import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2


DEFAULT_LATENT_TOKENS = 96
DEFAULT_LATENT_DIM = 128
DEFAULT_HEAD_DIM = 32
DEFAULT_PYRAMID_STRIDES = (8, 16, 32)


def _resolve_num_heads(d: int, head_dim_target: int = DEFAULT_HEAD_DIM) -> int:
    """Choose a valid head count whose 2D-RoPE head width is closest to target."""
    if d <= 0 or head_dim_target <= 0:
        raise ValueError(f"d and head_dim_target must be positive, got d={d}, target={head_dim_target}.")

    candidates = [h for h in range(1, d + 1) if d % h == 0 and (d // h) % 4 == 0]
    if not candidates:
        raise ValueError(f"d={d} is incompatible with 2D RoPE; a head dimension divisible by 4 is required.")
    return min(candidates, key=lambda h: (abs((d // h) - head_dim_target), h))


def _bounded_relative_gate(saliency: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Bounded relative gate: identity at the mean and strictly below two."""
    eps = torch.finfo(saliency.dtype).eps
    mean = saliency.mean(dim=dim, keepdim=True)
    return 2.0 * (saliency + eps) / (saliency + mean + 2.0 * eps)


def _grid_shape(num_cells: int, aspect_ratio: float) -> tuple[int, int]:
    """Return (rows, cols) whose product is num_cells and ratio best matches W/H."""
    if num_cells <= 0 or aspect_ratio <= 0:
        raise ValueError(f"Invalid grid request: cells={num_cells}, aspect={aspect_ratio}.")
    candidates = []
    for rows in range(1, int(math.sqrt(num_cells)) + 1):
        if num_cells % rows:
            continue
        cols = num_cells // rows
        candidates.extend(((rows, cols), (cols, rows)))
    return min(candidates, key=lambda rc: abs(math.log((rc[1] / rc[0]) / aspect_ratio)))


def _build_window_index_table(H: int, W: int, rows: int, cols: int,
                              device: torch.device):
    """Create a padded table of flattened spatial indices for non-overlapping windows."""
    cells = []
    for row in range(rows):
        y0 = math.floor(row * H / rows)
        y1 = math.floor((row + 1) * H / rows)
        for col in range(cols):
            x0 = math.floor(col * W / cols)
            x1 = math.floor((col + 1) * W / cols)
            yy = torch.arange(y0, y1, device=device).view(-1, 1)
            xx = torch.arange(x0, x1, device=device).view(1, -1)
            cells.append((yy * W + xx).reshape(-1))

    lengths = torch.tensor([cell.numel() for cell in cells], device=device, dtype=torch.long)
    max_len = int(lengths.max().item()) if cells else 0
    table = torch.zeros((len(cells), max_len), device=device, dtype=torch.long)
    valid = torch.zeros((len(cells), max_len), device=device, dtype=torch.bool)
    for idx, cell in enumerate(cells):
        n = cell.numel()
        if n:
            table[idx, :n] = cell
            valid[idx, :n] = True
    return table, valid, lengths


def _spatially_diverse_topk(scores: torch.Tensor, H: int, W: int, M: int,
                            num_latents: int, cache: dict) -> torch.Tensor:
    """Select equal Top-K quotas from a latent-aligned spatial partition."""
    total = H * W
    M = min(total, int(M))
    if M <= 0:
        raise ValueError(f"Gather token count must be positive, got M={M}.")

    rows, cols = _grid_shape(num_latents, W / H)
    # Very small feature maps cannot provide one valid quota per latent cell.
    if M < num_latents or M % num_latents:
        return torch.topk(scores, M, dim=-1).indices

    quota = M // num_latents
    key = (H, W, rows, cols, scores.device)
    if key not in cache:
        cache[key] = _build_window_index_table(H, W, rows, cols, scores.device)
    table, valid, lengths = cache[key]
    if int(lengths.min().item()) < quota:
        return torch.topk(scores, M, dim=-1).indices

    cell_scores = scores[:, table]
    cell_scores = cell_scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
    local_idx = torch.topk(cell_scores, quota, dim=-1).indices
    spatial_idx = table.unsqueeze(0).expand(scores.shape[0], -1, -1)
    return spatial_idx.gather(2, local_idx).reshape(scores.shape[0], M)


# ============================== 2D RoPE Attention ==============================
class RoPEMultiheadAttention2D(nn.Module):
    def __init__(self, d: int, head_dim_target: int = DEFAULT_HEAD_DIM):
        super().__init__()
        self.d = d
        self.heads = _resolve_num_heads(d, head_dim_target)
        self.head_dim = d // self.heads

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

        d_half = self.head_dim // 2
        freq_dim = d_half // 2
        # Grid-centre coordinates live in (0, 1). Half-integer harmonics avoid
        # identifying opposite image boundaries as the same periodic position.
        freq_bands = math.pi * (torch.arange(freq_dim, dtype=torch.float32) + 0.5)
        self.register_buffer("freq_bands", freq_bands)

    def _apply_rope_2d(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, H, N, D = x.shape
        D_half = D // 2

        cx = coords[..., 0:1].unsqueeze(1)
        cy = coords[..., 1:2].unsqueeze(1)

        freq = self.freq_bands.view(1, 1, 1, -1)
        angles_x = cx * freq
        angles_y = cy * freq

        angles_x = torch.cat([angles_x, angles_x], dim=-1).to(x.dtype)
        angles_y = torch.cat([angles_y, angles_y], dim=-1).to(x.dtype)

        xx = x[..., :D_half]
        xy = x[..., D_half:]

        def rotate_half(t):
            mid = t.shape[-1] // 2
            return torch.cat([-t[..., mid:], t[..., :mid]], dim=-1)

        rx = xx * angles_x.cos() + rotate_half(xx) * angles_x.sin()
        ry = xy * angles_y.cos() + rotate_half(xy) * angles_y.sin()

        return torch.cat([rx, ry], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                q_coords: torch.Tensor, k_coords: torch.Tensor) -> torch.Tensor:
        B, N_q, _ = q.shape
        N_k = k.shape[1]

        q = self.q_proj(q).view(B, N_q, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, N_k, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, N_k, self.heads, self.head_dim).transpose(1, 2)

        q = self._apply_rope_2d(q, q_coords)
        k = self._apply_rope_2d(k, k_coords)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N_q, self.d)
        return self.out_proj(out)


# ============================== Latent Memory ==================================
class LatentMemory(nn.Module):
    def __init__(self, L=DEFAULT_LATENT_TOKENS, d=DEFAULT_LATENT_DIM):
        super().__init__()
        self.L, self.d = L, d
        self.z0 = nn.Parameter(torch.randn(L, d) * 0.02)
        # Learned offsets from a deterministic aspect-aware grid. tanh(ref) keeps
        # each latent inside its own cell, preventing spatial anchor collapse.
        self.ref_offset = nn.Parameter(torch.zeros(L, 2))
        self.state = None
        self.z_p3 = None
        self._cached_grid = {}
        self.deepest_stride = None

    def reset(self):
        self.state = None
        self.z_p3 = None

    def get_latent_coords(self, B: int, H: int, W: int,
                          device: torch.device) -> torch.Tensor:
        rows, cols = _grid_shape(self.L, W / H)
        gy = (torch.arange(rows, device=device, dtype=self.ref_offset.dtype) + 0.5) / rows
        gx = (torch.arange(cols, device=device, dtype=self.ref_offset.dtype) + 0.5) / cols
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        base = torch.stack([grid_x, grid_y], dim=-1).reshape(self.L, 2)
        cell = self.ref_offset.new_tensor([1.0 / cols, 1.0 / rows])
        coords = base + 0.5 * cell * torch.tanh(self.ref_offset)
        return coords.unsqueeze(0).expand(B, -1, -1).to(device)

    def get_spatial_coords(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, device)
        if key not in self._cached_grid:
            gy = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
            gx = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
            grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
            coords = torch.stack([grid_x, grid_y], dim=-1).view(1, H * W, 2)
            self._cached_grid[key] = coords
        return self._cached_grid[key]


# ============================== EAD Downsampling v6 (Fixed) ===================
class EAD(Conv):
    def __init__(self, c1, c2, k=3, s=2, tau_init=None,
                 d_latent=DEFAULT_LATENT_DIM, *a, **kw):
        super().__init__(c1, c2, k, s)
        ac = torch.tensor([[[0.5, 0.5], [-0.5, -0.5]],
                           [[0.5, -0.5], [0.5, -0.5]],
                           [[0.5, -0.5], [-0.5, 0.5]]], dtype=torch.float32)
        self.register_buffer("ac", ac.unsqueeze(1).repeat(c1, 1, 1, 1))
        self.c1 = c1
        # tau is an unconstrained parameter; softplus(tau) is the energy threshold.
        # tau_init <= 0 is treated as automatic calibration for old YAML compatibility.
        use_calibration = tau_init is None or float(tau_init) <= 0.0
        if use_calibration:
            tau_raw = torch.zeros(1, c1, 1, 1)
        else:
            tau_value = torch.full((1, c1, 1, 1), float(tau_init))
            tau_raw = torch.log(torch.expm1(tau_value))
        self.tau = nn.Parameter(tau_raw)
        self.register_buffer("tau_initialized", torch.tensor(not use_calibration, dtype=torch.bool))

        self.modulator = nn.Linear(d_latent, c1)
        nn.init.zeros_(self.modulator.weight)
        nn.init.zeros_(self.modulator.bias)

        self.refine = nn.Conv2d(3 * c1, 3 * c1, 3, padding=1, groups=3 * c1, bias=False)
        self.proj = nn.Conv2d(3 * c1, c2, 1, bias=False)
        nn.init.zeros_(self.proj.weight)  # Single-zero init: Transfer-safe

        self.alpha = nn.Parameter(torch.zeros(1, c2, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, c2, 1, 1))  # Giữ gamma = 1.0 để gradient chảy ngay lập tức

        self.latest_saliency = None
        self.bank = None

    @torch.no_grad()
    def _initialize_tau_from_energy(self, energy: torch.Tensor) -> None:
        """Set tau^2 to per-channel median energy, so the median initial gate is 0.5."""
        if bool(self.tau_initialized):
            return

        values = energy.detach().float().squeeze(2).permute(1, 0, 2, 3).reshape(self.c1, -1)
        median_energy = values.median(dim=1).values

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(median_energy, op=torch.distributed.ReduceOp.SUM)
            median_energy /= torch.distributed.get_world_size()

        target_tau = median_energy.clamp_min(torch.finfo(torch.float32).eps).sqrt()
        raw_tau = torch.where(
            target_tau > 20.0,
            target_tau,
            torch.log(torch.expm1(target_tau).clamp_min(torch.finfo(torch.float32).tiny)),
        )
        self.tau.copy_(raw_tau.view(1, self.c1, 1, 1).to(self.tau.dtype))
        self.tau_initialized.fill_(True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Old Falcon checkpoints contain a learned tau but not tau_initialized.
        # Preserve that learned value instead of recalibrating it on the next batch.
        if prefix + "tau" in state_dict and prefix + "tau_initialized" not in state_dict:
            self.tau_initialized.fill_(True)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, x):
        y = super().forward(x)
        b, c, h, w = x.shape
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2))

        # 1. Bóc tách Haar Wavelet
        ac = F.conv2d(x, self.ac, stride=2, groups=self.c1)
        hh, ww = ac.shape[-2:]
        ac5 = ac.view(b, self.c1, 3, hh, ww)
        E = (ac5 ** 2).sum(2, keepdim=True)

        # One-shot, data-driven initialization during training. A loaded checkpoint
        # keeps its learned tau because tau_initialized is persisted in state_dict.
        # DetectionModel runs a dummy zero forward while building strides, before
        # link_lqc() attaches the bank. Do not let that synthetic pass calibrate tau.
        if self.training and self.bank is not None and not bool(self.tau_initialized):
            self._initialize_tau_from_energy(E)

        # 2. Ngưỡng năng lượng tương đối (Loại bỏ Instance Norm)
        if self.bank is not None and self.bank.state is not None:
            z_global = self.bank.state.mean(dim=1)
            delta_tau = self.modulator(z_global).view(b, self.c1, 1, 1, 1)
            tau2 = F.softplus(self.tau.unsqueeze(-1) + delta_tau).square()
        else:
            tau2 = F.softplus(self.tau.unsqueeze(-1)).square()

        eps = torch.finfo(E.dtype).eps
        phi = E / (E + tau2 + eps)
        phi_spatial = phi.squeeze(2).mean(dim=1, keepdim=True)
        self.latest_saliency = phi_spatial  # Đạo hàm truyền thông suốt

        # 3. Residual Modulation
        if phi_spatial.shape[-2:] != y.shape[-2:]:
            phi_spatial_mod = F.interpolate(phi_spatial, size=y.shape[-2:], mode="nearest")
        else:
            phi_spatial_mod = phi_spatial
        y = y * (1.0 + self.alpha * torch.tanh(phi_spatial_mod))

        # 4. Nhánh vi phân High-Frequency
        ac_gated = (ac5 * phi).reshape(b, 3 * self.c1, hh, ww)
        d = self.proj(self.refine(ac_gated))
        if d.shape[-2:] != y.shape[-2:]:
            d = F.interpolate(d, size=y.shape[-2:], mode="nearest")

        return y + self.gamma * d


# ============================== LQC Module =====================================
class LQC(nn.Module):
    def __init__(self, c, d=DEFAULT_LATENT_DIM, head_dim_target=DEFAULT_HEAD_DIM):
        super().__init__()
        self.d = d
        self.down = nn.Conv2d(c, d, 1)

        self.gather = RoPEMultiheadAttention2D(d, head_dim_target=head_dim_target)
        self.ng = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.nf = nn.LayerNorm(d)
        self.distribute = RoPEMultiheadAttention2D(d, head_dim_target=head_dim_target)

        self.dw = nn.Conv2d(d, d, 3, 1, 1, groups=d)
        self.up = nn.Conv2d(d, c, 1)
        self.gamma = nn.Parameter(torch.full((1, c, 1, 1), 0.1))

        self.bank = None
        self.prev_ead = None
        self.is_first = False
        self.stage_idx = 0
        self.pyramid_stride = None
        self._window_index_cache = {}

    def forward(self, x):
        B, C, H, W = x.shape
        xd = self.down(x)
        t = xd.flatten(2).transpose(1, 2)

        if self.bank is None:
            return x

        if self.is_first or self.bank.state is None:
            z = self.bank.z0.unsqueeze(0).expand(B, -1, -1).contiguous()
            # Spatial Content-Seeded LQC: bootstrap the first latent state from
            # this image, in the same row-major grid as the RoPE latent anchors.
            # Functional LayerNorm adds no trainable parameters/state-dict keys.
            # Keep reduction/normalization in FP32 for AMP stability; gradients
            # still flow through pooling to every contributing spatial feature.
            rows, cols = _grid_shape(self.bank.L, W / H)
            content = F.adaptive_avg_pool2d(xd.float(), (rows, cols))
            content = content.flatten(2).transpose(1, 2)
            content = F.layer_norm(content, (self.d,))
            z = z + content.to(dtype=z.dtype)
        else:
            z = self.bank.state

        coords_latent = self.bank.get_latent_coords(B, H, W, x.device)
        coords_spatial = self.bank.get_spatial_coords(H, W, x.device).expand(B, -1, -1)

        # 1. Saliency Map
        if self.prev_ead is not None and self.prev_ead.latest_saliency is not None:
            sal = self.prev_ead.latest_saliency
            if sal.shape[-2:] != (H, W):
                sal = F.interpolate(sal, size=(H, W), mode="bilinear", align_corners=False)
            saliency_flat = sal.flatten(2).transpose(1, 2)
        else:
            hf = x - F.avg_pool2d(x, 3, 1, 1)
            saliency_flat = (hf * hf).mean(1, keepdim=True).flatten(2).transpose(1, 2)

        # 2. GATHER với RoPE 2D
        sal_scores = saliency_flat.squeeze(-1).detach()
        if self.pyramid_stride is None or self.bank.deepest_stride is None:
            raise RuntimeError("LQC pyramid strides were not configured by link_lqc().")
        stride_ratio = self.bank.deepest_stride / self.pyramid_stride
        M = min(H * W, math.ceil(self.bank.L * stride_ratio))
        top_idx = _spatially_diverse_topk(
            sal_scores, H, W, M, self.bank.L, self._window_index_cache
        )

        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, self.d)
        k_gather = torch.gather(t, 1, top_idx_exp)
        v_gather = k_gather
        coords_top = torch.gather(coords_spatial, 1, top_idx.unsqueeze(-1).expand(-1, -1, 2))

        z_read = self.gather(q=z, k=k_gather, v=v_gather, q_coords=coords_latent, k_coords=coords_top)
        z = self.ng(z + z_read)
        z = self.nf(z + self.ffn(z))

        if self.stage_idx == 0:
            self.bank.z_p3 = z.clone()

        self.bank.state = z

        # 3. DISTRIBUTE với RoPE 2D
        o = self.distribute(q=t, k=z, v=z, q_coords=coords_spatial, k_coords=coords_latent)
        o = o * _bounded_relative_gate(saliency_flat)
        o = o.transpose(1, 2).view(B, self.d, H, W)
        o = o + self.dw(o)

        return x + self.gamma * self.up(o)


class C3k2LQC(C3k2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lqc = LQC(self.cv2.conv.out_channels)

    def forward(self, x):
        return self.lqc(super().forward(x))


# ============================== P2 Full Dense Injection Module ================
class LQCInject(nn.Module):
    def __init__(self, c, *args, d=DEFAULT_LATENT_DIM,
                 head_dim_target=DEFAULT_HEAD_DIM, **kw):
        super().__init__()
        if len(args) > 0 and args[0] != c:
            d = args[0]

        self.d = d
        self.down = nn.Conv2d(c, self.d, 1)
        self.ca = RoPEMultiheadAttention2D(self.d, head_dim_target=head_dim_target)

        self.smooth_conv = nn.Conv2d(self.d, self.d, 3, padding=1, groups=self.d, bias=False)
        with torch.no_grad():
            kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], dtype=torch.float32) / 16.0
            self.smooth_conv.weight.copy_(kernel.view(1, 1, 3, 3).repeat(self.d, 1, 1, 1))

        self.up = nn.Conv2d(self.d, c, 1, bias=False)
        self.gamma = nn.Parameter(torch.full((1, c, 1, 1), 0.15))
        self.bank = None
        self.p2_ead = None

        self.lvl_p3 = nn.Parameter(torch.zeros(1, 1, d))
        self.lvl_p5 = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x):
        if self.bank is None or self.bank.state is None:
            return x

        B, C, H, W = x.shape
        xd = self.down(x)
        t = xd.flatten(2).transpose(1, 2)

        coords_spatial = self.bank.get_spatial_coords(H, W, x.device).expand(B, -1, -1)
        coords_latent = self.bank.get_latent_coords(B, H, W, x.device)
        z_p5 = self.bank.state

        if getattr(self.bank, 'z_p3', None) is not None:
            z_p3 = self.bank.z_p3
            k = torch.cat([z_p3 + self.lvl_p3, z_p5 + self.lvl_p5], dim=1)
            v = torch.cat([z_p3, z_p5], dim=1)
            k_coords = torch.cat([coords_latent, coords_latent], dim=1)
        else:
            k = z_p5
            v = z_p5
            k_coords = coords_latent

        o_dense = self.ca(q=t, k=k, v=v, q_coords=coords_spatial, k_coords=k_coords)

        if self.p2_ead is not None and self.p2_ead.latest_saliency is not None:
            sal = self.p2_ead.latest_saliency
            if sal.shape[-2:] != (H, W):
                sal = F.interpolate(sal, size=(H, W), mode="bilinear", align_corners=False)
            sal_gate = sal.flatten(2).transpose(1, 2)
            o_dense = o_dense * _bounded_relative_gate(sal_gate)

        o = o_dense.transpose(1, 2).view(B, self.d, H, W)
        o_smooth = self.smooth_conv(o)

        return x + self.gamma * self.up(o_smooth)


class LatentRefresh(nn.Module):
    """Gated semantic refresh from final P5_out; leaves the feature unchanged.

    The zero-initialized ReZero gate makes this branch an exact identity at
    initialization. The network must learn a non-zero ``eta`` before P5
    semantics are allowed to alter the latent state consumed by LQCInject.
    """

    def __init__(self, c, d=DEFAULT_LATENT_DIM, head_dim_target=DEFAULT_HEAD_DIM, *args, **kwargs):
        super().__init__()
        self.d = d
        self.down = nn.Conv2d(c, d, 1)
        self.gather = RoPEMultiheadAttention2D(d, head_dim_target=head_dim_target)
        self.norm_gather = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.norm_ffn = nn.LayerNorm(d)
        self.eta = nn.Parameter(torch.zeros(()))
        self.bank = None
        self._window_index_cache = {}

    def forward(self, x):
        if self.bank is None:
            return x

        B, _, H, W = x.shape
        t = self.down(x).flatten(2).transpose(1, 2)
        z_old = self.bank.state
        if z_old is None:
            z_old = self.bank.z0.unsqueeze(0).expand(B, -1, -1).contiguous()

        coords_latent = self.bank.get_latent_coords(B, H, W, x.device)
        coords_spatial = self.bank.get_spatial_coords(H, W, x.device).expand(B, -1, -1)

        # P5_out activation energy is semantic, unlike the early EAD edge map.
        saliency = t.detach().float().square().mean(dim=-1)
        M = min(H * W, self.bank.L)
        top_idx = _spatially_diverse_topk(
            saliency, H, W, M, self.bank.L, self._window_index_cache
        )
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, self.d)
        k_gather = torch.gather(t, 1, top_idx_exp)
        coords_top = torch.gather(
            coords_spatial, 1, top_idx.unsqueeze(-1).expand(-1, -1, 2)
        )

        z_read = self.gather(
            q=z_old, k=k_gather, v=k_gather,
            q_coords=coords_latent, k_coords=coords_top,
        )
        z_refresh = self.norm_gather(z_old + z_read)
        z_refresh = self.norm_ffn(z_refresh + self.ffn(z_refresh))

        # Learnable residual gate:
        #   z_out = z_old + tanh(eta) * (z_refresh - z_old)
        # eta=0 makes the refresh path exactly inactive at initialization.
        z_out = z_old + torch.tanh(self.eta) * (z_refresh - z_old)
        self.bank.state = z_out
        return x


# ============================== Link & Register Pipeline =======================
def _set_shared_bank(module: nn.Module, bank: LatentMemory) -> None:
    """Attach a runtime reference without registering the same bank many times."""
    object.__setattr__(module, "bank", bank)


def link_lqc(model, L=DEFAULT_LATENT_TOKENS, d=DEFAULT_LATENT_DIM,
             pyramid_strides=DEFAULT_PYRAMID_STRIDES):
    bank = LatentMemory(L, d)

    layers = list(model.model)
    ead_modules = [m for m in layers if isinstance(m, EAD)]
    lqc_blocks = [m for m in layers if isinstance(m, C3k2LQC)]
    lqc_modules = [m.lqc for m in lqc_blocks]
    inject_modules = [m for m in layers if isinstance(m, LQCInject)]
    refresh_modules = [m for m in layers if isinstance(m, LatentRefresh)]

    if len(lqc_modules) != len(pyramid_strides):
        raise ValueError(
            f"Expected {len(pyramid_strides)} C3k2LQC stages for strides {tuple(pyramid_strides)}, "
            f"but found {len(lqc_modules)}."
        )
    if (len(ead_modules) != len(lqc_modules) + 1
            or len(inject_modules) != 1 or len(refresh_modules) != 1):
        raise ValueError(
            "Falcon LQC topology must contain one P2 EAD, one EAD before each LQC stage, "
            "one LatentRefresh and one LQCInject; "
            f"found EAD={len(ead_modules)}, LQC={len(lqc_modules)}, "
            f"Refresh={len(refresh_modules)}, Inject={len(inject_modules)}."
        )

    pyramid_strides = tuple(int(s) for s in pyramid_strides)
    if any(a >= b for a, b in zip(pyramid_strides, pyramid_strides[1:])):
        raise ValueError(f"pyramid_strides must be strictly increasing, got {pyramid_strides}.")
    bank.deepest_stride = pyramid_strides[-1]

    for e in ead_modules:
        if e.modulator.in_features != d:
            raise ValueError(f"EAD latent dim={e.modulator.in_features} does not match bank dim={d}.")
        _set_shared_bank(e, bank)
    for i, (block, l, stride) in enumerate(zip(lqc_blocks, lqc_modules, pyramid_strides)):
        if l.d != d:
            raise ValueError(f"LQC stage {i} dim={l.d} does not match bank dim={d}.")
        block_idx = layers.index(block)
        if block_idx == 0 or not isinstance(layers[block_idx - 1], EAD):
            raise ValueError(f"C3k2LQC layer {block_idx} must immediately follow its EAD layer.")
        _set_shared_bank(l, bank)
        l.is_first = (i == 0)
        l.stage_idx = i
        l.pyramid_stride = stride
    for inj in inject_modules:
        if inj.d != d:
            raise ValueError(f"LQCInject dim={inj.d} does not match bank dim={d}.")
        _set_shared_bank(inj, bank)
    for refresh in refresh_modules:
        if refresh.d != d:
            raise ValueError(f"LatentRefresh dim={refresh.d} does not match bank dim={d}.")
        if layers.index(refresh) <= layers.index(lqc_blocks[-1]):
            raise ValueError("LatentRefresh must run after the deepest C3k2LQC stage.")
        if layers.index(refresh) >= layers.index(inject_modules[0]):
            raise ValueError("LatentRefresh must run before LQCInject.")
        _set_shared_bank(refresh, bank)

    for block, l in zip(lqc_blocks, lqc_modules):
        l.prev_ead = layers[layers.index(block) - 1]

    inject_modules[0].p2_ead = ead_modules[0]

    model.add_module("lqc_bank", bank)
    return bank


def register_lqc():
    import ultralytics.nn.tasks as _t
    _t.EAD = EAD
    _t.LQC = LQC
    _t.C3k2LQC = C3k2LQC
    _t.LQCInject = LQCInject
    _t.LatentRefresh = LatentRefresh

    if getattr(_t, "_lqc_patched_pan_fpn_v7", False):
        return

    src = inspect.getsource(_t.parse_model)

    def replace_required(source, old, new, label):
        if source.count(old) != 1:
            raise RuntimeError(
                f"Cannot register {label}: unsupported ultralytics parse_model source "
                f"(expected one exact insertion point, found {source.count(old)})."
            )
        return source.replace(old, new, 1)

    src = replace_required(
        src,
        "            Conv,\n            ConvTranspose,",
        "            Conv,\n            EAD,\n            ConvTranspose,",
        "EAD",
    )
    src = replace_required(
        src,
        "            C3k2,\n            RepNCSPELAN4,",
        "            C3k2,\n            C3k2LQC,\n            RepNCSPELAN4,",
        "C3k2LQC base module",
    )
    src = replace_required(
        src,
        "            C3k2,\n            C2fAttn,",
        "            C3k2,\n            C3k2LQC,\n            C2fAttn,",
        "C3k2LQC repeat module",
    )
    src = replace_required(
        src,
        "        elif m is Concat:\n            c2 = sum(ch[x] for x in f)",
        "        elif m is Concat:\n            c2 = sum(ch[x] for x in f)\n"
        "        elif m in {LQCInject, LatentRefresh}:\n"
        "            c2 = ch[f]\n            args = [ch[f], *args]",
        "LQCInject/LatentRefresh",
    )

    ns = dict(vars(_t))
    ns.update(
        EAD=EAD, LQC=LQC, C3k2LQC=C3k2LQC,
        LQCInject=LQCInject, LatentRefresh=LatentRefresh,
    )
    exec(compile(src, "<lqc_bifpn_patch>", "exec"), ns)
    _t.parse_model = ns["parse_model"]
    _t._lqc_patched_pan_fpn_v7 = True

# NOTE (port): registration is lazy. ``perception.models.detector`` calls
# ``register_lqc()`` only when the YAML actually uses EAD/LQC modules, so the
# baseline detector never depends on patching Ultralytics internals.
