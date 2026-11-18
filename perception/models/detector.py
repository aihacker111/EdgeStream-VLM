# Ported from AERO-MOT (AGPL-3.0). See perception/NOTICE.
"""YOLO11-P2 detector (optionally JDE with a dense ReID head) for the perception module.

Differences from ``aeromot/nn/models/model.py``:

* inference-only: COCO-pretrained remapping and training flags are dropped
  (training stays in the AERO-MOT repo for now, see TODO.md);
* EAD/LQC modules are registered with Ultralytics *only* when the YAML uses them,
  so the baseline detector never depends on patching ``parse_model``;
* checkpoint loading is strict and loud: missing/unexpected keys raise instead of
  being printed and silently ignored;
* the ReID head is enabled automatically when the checkpoint contains it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .jde_reid import StandardJDEReID

CFG_DIR = Path(__file__).resolve().parent / "cfg"
_LQC_MODULES = ("EAD", "C3k2LQC", "LQCInject", "LatentRefresh")


def resolve_cfg(cfg: str) -> Path:
    path = Path(cfg)
    if not path.is_file():
        path = CFG_DIR / cfg
    if not path.is_file():
        raise FileNotFoundError(f"Detector YAML not found: {cfg}")
    return path


def _yaml_uses_lqc(yolo_cfg: dict) -> bool:
    layers = list(yolo_cfg.get("backbone", [])) + list(yolo_cfg.get("head", []))
    return any(str(layer[2]) in _LQC_MODULES for layer in layers)


def _extract_state_dict(obj) -> dict:
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    if isinstance(obj, dict) and ("state_dict" in obj or "model" in obj):
        obj = obj.get("state_dict", obj.get("model"))
        if hasattr(obj, "state_dict"):
            obj = obj.state_dict()
    return {(k[7:] if k.startswith("module.") else k): v for k, v in obj.items()}


class FalconDetector(nn.Module):
    """Single-pass YOLO11-P2 detector, optionally emitting a stride-4 embedding map."""

    def __init__(self, cfg: str, num_classes: int, scale: str = "s",
                 use_reid: bool = False, reid_dim: int = 128):
        super().__init__()
        cfg_path = resolve_cfg(cfg)
        with open(cfg_path, "r") as f:
            yolo_cfg = yaml.safe_load(f)
        yolo_cfg["nc"] = int(num_classes)
        yolo_cfg["scale"] = scale

        self.uses_lqc = _yaml_uses_lqc(yolo_cfg)
        if self.uses_lqc:
            from .lqc import C3k2LQC, link_lqc, register_lqc
            register_lqc()

        from ultralytics.nn.tasks import DetectionModel
        self.yolo = DetectionModel(cfg=yolo_cfg, nc=int(num_classes), verbose=False)
        if self.uses_lqc and any(isinstance(m, C3k2LQC) for m in self.yolo.modules()):
            link_lqc(self.yolo, L=96, d=128)

        self.num_classes = int(num_classes)
        self.cfg_path = str(cfg_path)
        self.head_idx = len(self.yolo.model) - 1
        self.detect_head = self.yolo.model[self.head_idx]
        head_stride = getattr(self.detect_head, "stride", None)
        self.reid_dense_stride = int(head_stride[0].item()) if head_stride is not None else 4

        self.use_reid = bool(use_reid)
        self.reid_dim = int(reid_dim)
        if self.use_reid:
            ch_s4 = self.detect_head.cv2[0][0].conv.in_channels
            self.reid_head = StandardJDEReID(c=ch_s4, reid_dim=self.reid_dim)

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_checkpoint(cls, weights: str, cfg: Optional[str] = None, num_classes: int = 10,
                        scale: str = "s", reid_dim: int = 128,
                        map_location: str = "cpu") -> "FalconDetector":
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"Checkpoint not found: {weights}")
        raw = torch.load(weights, map_location=map_location, weights_only=False)
        sd = _extract_state_dict(raw)
        has_reid = any(k.startswith("reid_head.") for k in sd)
        if cfg is None:
            cfg = "yolo_falcon_lqcv2.yaml" if any(".tau" in k or "lqc" in k for k in sd) \
                else "yolo_falcon_lqcv2_baseline.yaml"
        model = cls(cfg, num_classes=num_classes, scale=scale, use_reid=has_reid, reid_dim=reid_dim)
        model.load_weights(sd)
        model.checkpoint_epoch = raw.get("epoch") if isinstance(raw, dict) else None
        return model

    def load_weights(self, state_dict: dict) -> None:
        own = self.state_dict()
        # training-only heads (ID classifiers, kappa head) are not part of inference
        # ``backbone_neck.*`` are aliases of ``yolo.model.*`` in the AERO-MOT wrapper.
        sd = {k: v for k, v in state_dict.items()
              if not k.startswith(("reid_classifier.", "reid_head.kappa_head", "backbone_neck."))}
        missing = [k for k in own if k not in sd]
        unexpected = [k for k in sd if k not in own]
        bad_shape = [k for k in sd if k in own and sd[k].shape != own[k].shape]
        if missing or unexpected or bad_shape:
            raise RuntimeError(
                "Checkpoint does not match detector "
                f"({os.path.basename(self.cfg_path)}): {len(missing)} missing, "
                f"{len(unexpected)} unexpected, {len(bad_shape)} shape mismatches. "
                f"e.g. missing={missing[:3]} unexpected={unexpected[:3]} shape={bad_shape[:3]}")
        self.load_state_dict(sd, strict=True)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> dict:
        bank = getattr(self.yolo, "lqc_bank", None)
        if bank is not None:
            bank.reset()
        y = []
        for m in self.yolo.model[:-1]:
            if m.f != -1:
                x_in = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            else:
                x_in = x
            x = m(x_in)
            y.append(x if m.i in self.yolo.save else None)
        feats = [x if j == -1 else y[j] for j in self.detect_head.f]

        out = {"reid_dense_stride": self.reid_dense_stride}
        if self.use_reid:
            emb_map, _ = self.reid_head(feats[0])
            out["emb_map"] = emb_map
            out["reid_dense_normed"] = F.normalize(emb_map, dim=1, eps=1e-6)
        out["yolo_out"] = self.detect_head(feats)
        if bank is not None:
            bank.reset()
        return out
