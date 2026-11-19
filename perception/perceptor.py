"""Perceptor: frame in -> tracked objects out. The only entry point downstream code uses."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from .config import PerceptionConfig
from .models.detector import FalconDetector
from .postprocess import YoloFalconPostProcessor
from .preprocess import preprocess, to_tensor
from .tracker import FalconTrack, FalconTracker
from .types import FrameOutput, TrackOutput


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Perceptor:
    """Detector + AERO-MOT tracker behind one stateful ``process(frame)`` call.

    One instance per camera stream. Call :meth:`reset` between unrelated streams.
    """

    def __init__(self, cfg: PerceptionConfig, detector: Optional[FalconDetector] = None):
        self.cfg = cfg
        d = cfg.detector
        self.device = _resolve_device(d.device)
        self.detector = detector or FalconDetector.from_checkpoint(
            d.weights, cfg=d.cfg, num_classes=d.num_classes, scale=d.scale)
        self.detector.to(self.device).eval()
        if d.half and self.device.type == "cuda":
            self.detector.half()

        mode = cfg.tracker.use_appearance
        if mode == "auto":
            self.appearance = bool(self.detector.use_reid)
        else:
            self.appearance = mode == "on"
            if self.appearance and not self.detector.use_reid:
                raise ValueError("use_appearance=on requires a checkpoint with a ReID head.")

        self.net_w, self.net_h = d.input_wh
        self.post = YoloFalconPostProcessor(
            num_classes=d.num_classes, num_top_queries=d.max_det, conf_thres=d.conf_thres,
            nms_iou=d.nms_iou, letterbox=d.letterbox)
        self.post.set_net_hw(self.net_h, self.net_w)
        self.post.cand_floor = float(d.cand_floor)

        self._opt = cfg.tracker_opt(self.appearance)
        self.tracker = FalconTracker(self._opt, frame_rate=cfg.tracker.frame_rate)
        self._frame_idx = 0
        self._next_global_id = 1

    # ------------------------------------------------------------------ API
    @classmethod
    def from_yaml(cls, path: str) -> "Perceptor":
        return cls(PerceptionConfig.from_yaml(path))

    def reset(self) -> None:
        self.tracker = FalconTracker(self._opt, frame_rate=self.cfg.tracker.frame_rate)
        self._frame_idx = 0
        self._next_global_id = 1

    @torch.no_grad()
    def process(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> FrameOutput:
        d = self.cfg.detector
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()

        chw, tf = preprocess(frame_bgr, self.net_w, self.net_h, letterbox=d.letterbox)
        x = to_tensor(chw, self.device, half=d.half and self.device.type == "cuda")
        t1 = time.perf_counter()

        out = self.detector(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        sizes = torch.tensor([[h, w]], device=self.device)
        res = self.post(out, sizes)[0]
        dets = self._to_tracks(res.get("boxes"), res.get("scores"), res.get("labels"), res.get("reid"))
        cands = self._to_tracks(res.get("cand_boxes"), res.get("cand_scores"),
                                res.get("cand_labels"), res.get("cand_reid"))

        if self.appearance and out.get("emb_map") is not None:
            self.tracker.set_dense(out["emb_map"][0].float(), stride=out["reid_dense_stride"],
                                   ratio_x=tf.ratio_x, ratio_y=tf.ratio_y,
                                   pad_w=tf.pad_w, pad_h=tf.pad_h,
                                   dense_normed=out["reid_dense_normed"][0].float())
        self.tracker.set_image(frame_bgr)
        targets = self.tracker.update(dets, h_orig=h, w_orig=w, cands_per_class=cands)
        t3 = time.perf_counter()

        tracks = self._emit(targets)
        self._frame_idx += 1
        n_det = sum(len(v) for v in dets.values())
        return FrameOutput(
            frame_idx=self._frame_idx - 1,
            timestamp=float(timestamp) if timestamp is not None else (self._frame_idx - 1) / self.cfg.tracker.frame_rate,
            image_hw=(h, w), tracks=tracks, num_detections=n_det,
            latency_ms={"preprocess": (t1 - t0) * 1e3, "detector": (t2 - t1) * 1e3,
                        "post+track": (t3 - t2) * 1e3, "total": (t3 - t0) * 1e3})

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray, conf_thres: Optional[float] = None) -> list:
        """Detector only (no tracker state change): [(tlwh, score, cls_name)] in input-image px.
        Used for zoom re-detection on crops."""
        d = self.cfg.detector
        h, w = frame_bgr.shape[:2]
        chw, _ = preprocess(frame_bgr, self.net_w, self.net_h, letterbox=d.letterbox)
        x = to_tensor(chw, self.device, half=d.half and self.device.type == "cuda")
        out = self.detector(x)
        res = self.post(out, torch.tensor([[h, w]], device=self.device))[0]
        names = self.cfg.output.class_names
        boxes, scores, labels = res.get("boxes"), res.get("scores"), res.get("labels")
        dets = []
        if boxes is None or len(boxes) == 0:
            return dets
        thr = d.conf_thres if conf_thres is None else conf_thres
        for b, sc, lb in zip(boxes.float().cpu().numpy(), scores.float().cpu().numpy(), labels.cpu().numpy()):
            if sc < thr or b[2] <= b[0] or b[3] <= b[1]:
                continue
            dets.append((np.array([b[0], b[1], b[2] - b[0], b[3] - b[1]], np.float32), float(sc),
                         names[int(lb)] if int(lb) < len(names) else str(int(lb))))
        return dets

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _to_tracks(boxes, scores, labels, reid) -> dict:
        per_cls = defaultdict(list)
        if boxes is None or len(boxes) == 0:
            return per_cls
        boxes = boxes.float().cpu().numpy()
        scores = scores.float().cpu().numpy()
        labels = labels.cpu().numpy()
        reid = reid.float().cpu().numpy() if reid is not None else None
        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        for i in np.where((ws > 0) & (hs > 0))[0]:
            tlwh = np.array([boxes[i, 0], boxes[i, 1], ws[i], hs[i]], dtype=np.float32)
            emb = reid[i] if reid is not None else np.zeros(1, dtype=np.float32)
            per_cls[int(labels[i])].append(FalconTrack(tlwh, float(scores[i]), emb, int(labels[i])))
        return per_cls

    def _emit(self, targets: dict) -> list[TrackOutput]:
        o = self.cfg.output
        names = o.class_names
        kappa_max = float(self._opt.kappa_max)
        out = []
        for cls_id, tracks in targets.items():
            for t in tracks:
                if t.track_id < 0:
                    continue
                tlwh = np.asarray(t.curr_tlwh, dtype=np.float32).copy()
                if tlwh[2] * tlwh[3] <= o.min_box_area:
                    continue
                recovered = bool(getattr(t, "_is_recovered_this_frame", False))
                if recovered and not o.keep_recovered:
                    continue
                gid = getattr(t, "_global_id", None)
                if gid is None:
                    gid = self._next_global_id
                    self._next_global_id += 1
                    t._global_id = gid
                vel = np.asarray(t.mean[4:6], dtype=np.float32) if t.mean is not None else np.zeros(2, np.float32)
                emb = t.smooth_feat if (self.appearance and t.smooth_feat is not None
                                        and t.smooth_feat.size > 1) else None
                out.append(TrackOutput(
                    global_id=int(gid), track_id=int(t.track_id), cls_id=int(cls_id),
                    cls_name=names[cls_id] if cls_id < len(names) else str(cls_id),
                    tlwh=tlwh, velocity=vel, quality=float(getattr(t, "q", 0.0) or 0.0),
                    det_score=float(t.score), age=int(t.frame_id - t.start_frame),
                    is_recovered=recovered,
                    kappa=t.kappa(kappa_max=kappa_max) if emb is not None else None,
                    embedding=None if emb is None else np.asarray(emb, dtype=np.float32).copy()))
        out.sort(key=lambda r: r.global_id)
        return out
