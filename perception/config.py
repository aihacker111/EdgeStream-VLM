"""Configuration for the perception module (one YAML file, no argparse sprawl)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import yaml

from .types import VISDRONE_CLASSES

# Tracker defaults = DEFAULT_FALCON_CFG in AERO-MOT tools/benchmark/common.py (paper setting).
PAPER_TRACKER_DEFAULTS = {
    "cost_mode": "evidential",
    "use_motion_llr": 1,
    "use_appearance_llr": 1,
    "use_dynamic_kappa": 1,
    "kappa_max": 300.0,
    "qam_cov_mode": "margin_scaled",
    "qam_distinctiveness_margin": 0.25,
    "qam_max_streak": 1,
    "qam_quality_gate": 0.25,
    "qam_roi_radius": 16,
    "qam_beta": 5.0,
    "disable_tcc": False,
    "cand_iou_thres": 0.35,
    "survival_mode": "kinetic",
    "max_age": 30,
    "w_app": 0.08,
    "lambda_glrt": 0.80,
    "clutter_scale": 1.0,
    # [port] switches that used to be hard-coded in the tracker
    "class_penalty": 18.0,
    "minority_large_classes": [],     # AERO-MOT VisDrone 5-class eval used [2, 3, 4]
    "qam_beta_modulation": None,      # None = original rule (modulate iff beta not in {2, 5})
}


@dataclass
class DetectorConfig:
    weights: str = "weights/falcon_detector_1.pth"
    cfg: Optional[str] = None          # None = infer from checkpoint keys
    num_classes: int = 10
    scale: str = "s"
    input_wh: tuple[int, int] = (960, 544)
    letterbox: bool = False
    conf_thres: float = 0.40           # high-score detections
    cand_floor: float = 0.15           # low-score band [cand_floor, conf_thres) for stage 2.5
    nms_iou: float = 0.60
    max_det: int = 500
    device: str = "auto"               # auto | cpu | cuda | cuda:0 | mps
    half: bool = False


@dataclass
class TrackerConfig:
    frame_rate: int = 30
    use_appearance: str = "auto"       # auto (=on iff checkpoint has a ReID head) | on | off
    params: dict = field(default_factory=lambda: dict(PAPER_TRACKER_DEFAULTS))


@dataclass
class OutputConfig:
    min_box_area: float = 100.0
    class_names: tuple[str, ...] = VISDRONE_CLASSES
    keep_recovered: bool = True        # emit QAM-recovered tracks (flagged is_recovered)


@dataclass
class PerceptionConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PerceptionConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        root = Path(path).resolve().parent
        det = DetectorConfig(**raw.get("detector", {}))
        det.input_wh = tuple(det.input_wh)
        if det.weights and not Path(det.weights).is_absolute():
            cand = (root / det.weights).resolve()
            if not cand.exists():
                cand = (root.parent / det.weights).resolve()
            det.weights = str(cand)
        trk_raw = raw.get("tracker", {})
        params = dict(PAPER_TRACKER_DEFAULTS)
        params.update(trk_raw.get("params", {}) or {})
        trk = TrackerConfig(frame_rate=trk_raw.get("frame_rate", 30),
                            use_appearance=str(trk_raw.get("use_appearance", "auto")),
                            params=params)
        out_raw = raw.get("output", {})
        out = OutputConfig(**{**out_raw, "class_names": tuple(out_raw.get("class_names", VISDRONE_CLASSES))})
        return cls(det, trk, out)

    def to_dict(self) -> dict:
        return asdict(self)

    def tracker_opt(self, appearance: bool) -> SimpleNamespace:
        """Build the ``opt`` namespace the ported AERO-MOT tracker expects."""
        p = dict(self.tracker.params)
        p.update(
            num_classes=self.detector.num_classes,
            conf_thres=self.detector.conf_thres,
            use_appearance_motion=int(appearance),
            use_reid_gallery=int(appearance),
        )
        return SimpleNamespace(**p)
