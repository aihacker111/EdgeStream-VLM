"""Referring-MOT datasets: AerialMind (UAV, primary) and Refer-KITTI / Refer-KITTI-V2 (ground).

Both follow the RMOT conventions (Wu et al., CVPR 2023), which this module reproduces exactly:

* ``expression/<video_id>/<expression>.json`` holds ``{"label": {frame_key: [object ids]}, "sentence": ...}``.
  The file name without ``.json`` is the query AND the ``expression_id`` of the seqmap row
  ``<video_id>+<expression_id>`` (the ``sentence`` field is not used, files are never de-duplicated).
* ``labels_with_ids/.../<video_id>/<frame>.txt`` rows ``class id x y w h`` normalised to [0, 1];
  the frame key of the expression equals ``int(<image file stem>)``.
* Evaluation: ``<root>/<video_id>/<expression_id>/{gt.txt,predict.txt}``, frames 1..N where N is
  the number of images of the video; frame ``r + 1`` for the image of rank ``r`` in sorted order.
  (For KITTI's 0-based names this is RMOT's ``int(k) + 1``; for VisDrone's 1-based names it is
  ``int(k)``. Both equal the rank, which is what TrackEval's sequence length assumes.)

Box convention of ``labels_with_ids``: top-left x,y,w,h for KITTI, VisDrone and UAVDT (checked by
drawing the real AerialMind labels; the AerialMind loader reads non-VisDrone paths as centre boxes,
which is wrong for UAVDT). ``label_format="auto"`` uses top-left unless :func:`check_label_format`
clearly measures centre boxes.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .hota import write_mot

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# AerialMind official test splits (Readme of shawnliang420/AerialMind)
AERIALMIND_TEST = {
    "in_domain": ["uav0000009_03358_v", "uav0000073_00600_v", "uav0000073_04464_v", "uav0000077_00720_v",
                  "uav0000088_00290_v", "uav0000119_02301_v", "uav0000120_04775_v", "uav0000161_00000_v",
                  "uav0000188_00000_v", "uav0000201_00000_v", "uav0000249_00001_v", "uav0000249_02688_v",
                  "uav0000297_00000_v", "uav0000297_02761_v", "uav0000306_00230_v", "uav0000355_00001_v",
                  "uav0000370_00001_v"],
    "cross_domain": ["M0203", "M0205", "M0209", "M0403", "M0701", "M0801", "M1001", "M1004", "M1007",
                     "M1101", "M1301", "M1302", "M1401"],
}
# Expected seqmap sizes (rows) of the official files, used as a completeness check
AERIALMIND_SEQMAP_ROWS = {"in_domain": 4986, "cross_domain": 3415}
# Refer-KITTI(-V2) test videos (seqmap of TempRMOT / RMOT)
REFER_KITTI_TEST = {"refer_kitti_v2": ["0005", "0011", "0013", "0019"],
                    "refer_kitti": ["0005", "0011", "0013"]}
FPS = {"aerialmind": 30.0, "refer_kitti_v2": 10.0, "refer_kitti": 10.0}


def _digits(stem: str) -> int:
    m = re.findall(r"\d+", stem)
    if not m:
        raise ValueError(f"no frame number in file name {stem!r}")
    return int(m[-1])


@dataclass
class RMOTExpression:
    video_id: str
    expression_id: str              # file name without .json == the query text
    label: dict                     # frame key (int) -> list of object ids
    path: str = ""
    sentence: str = ""

    @property
    def text(self) -> str:
        return self.expression_id

    @property
    def seq_name(self) -> str:
        return f"{self.video_id}+{self.expression_id}"


@dataclass
class RMOTVideo:
    dataset: str
    video_id: str
    image_dir: str
    label_dir: str
    frames: list = field(default_factory=list)          # sorted image paths
    frame_keys: list = field(default_factory=list)      # int(stem) per image
    fps: float = 30.0
    label_format: str = "auto"                          # auto | x1y1wh | cxcywh
    _wh: tuple | None = None
    _auto_fmt: str | None = None

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def fmt(self) -> str:
        """Box convention of labels_with_ids. ``auto``: top-left x,y,w,h unless the data clearly says
        centre (measured: the reading that keeps boxes inside the image).
        Checked on the real AerialMind release: VisDrone AND UAVDT are top-left. The AerialMind
        loader's centre branch for non-'uav' paths is wrong for UAVDT (reading it as centre puts
        7-9% of the boxes outside the image and shifts every box by half its size, e.g. M0209)."""
        if self.label_format != "auto":
            return self.label_format
        if self._auto_fmt is None:
            rule = "x1y1wh"
            chk = check_label_format(self, n_frames=40) if self.label_dir else {"n": 0}
            if chk["n"] and abs(chk["x1y1wh"] - chk["cxcywh"]) > 0.005:
                self._auto_fmt = chk["best"]
                if self._auto_fmt != rule:
                    warnings.warn(f"{self.video_id}: labels measured as {self._auto_fmt} "
                                  f"(out-of-image x1y1wh={chk['x1y1wh']:.3f}, cxcywh={chk['cxcywh']:.3f}); "
                                  f"the default is {rule}. Using {self._auto_fmt}.")
            else:
                self._auto_fmt = rule
        return self._auto_fmt

    @property
    def image_wh(self) -> tuple[int, int]:
        if self._wh is None:
            import cv2
            img = cv2.imread(self.frames[0])
            if img is None:
                raise FileNotFoundError(self.frames[0])
            self._wh = (img.shape[1], img.shape[0])
        return self._wh

    def rank_of_key(self) -> dict:
        return {k: r for r, k in enumerate(self.frame_keys)}

    def label_path(self, rank: int) -> str:
        stem = os.path.splitext(os.path.basename(self.frames[rank]))[0]
        return os.path.join(self.label_dir, stem + ".txt")

    def raw_labels(self, rank: int) -> np.ndarray:
        p = self.label_path(rank)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return np.zeros((0, 6), np.float64)
        return np.loadtxt(p, dtype=np.float64).reshape(-1, 6)

    def gt_boxes(self, rank: int, fmt: str | None = None) -> np.ndarray:
        """(K, 5) rows ``id, x, y, w, h`` in pixels (top-left)."""
        lab = self.raw_labels(rank)
        if not len(lab):
            return np.zeros((0, 5))
        W, H = self.image_wh
        fmt = fmt or self.fmt
        x, y, w, h = lab[:, 2] * W, lab[:, 3] * H, lab[:, 4] * W, lab[:, 5] * H
        if fmt == "cxcywh":
            x, y = x - w / 2, y - h / 2
        return np.stack([lab[:, 1], x, y, w, h], 1)

    def gt_rows(self, expr: RMOTExpression) -> list:
        """MOT rows (frame 1..N, id, x, y, w, h) of the referred objects."""
        r_of = self.rank_of_key()
        rows = []
        for k, ids in expr.label.items():
            r = r_of.get(int(k))
            if r is None:
                continue
            want = {int(i) for i in ids}
            for oid, x, y, w, h in self.gt_boxes(r):
                if int(oid) in want:
                    rows.append((r + 1, int(oid), x, y, w, h))
        rows.sort()
        return rows

    def write_gt(self, expr: RMOTExpression, root: str) -> str:
        p = os.path.join(root, self.video_id, expr.expression_id, "gt.txt")
        write_mot(p, self.gt_rows(expr))
        return p

    def referred_ids(self, expr: RMOTExpression, rank: int) -> set:
        k = self.frame_keys[rank]
        return {int(i) for i in expr.label.get(k, [])}


def check_label_format(video: RMOTVideo, n_frames: int = 30) -> dict:
    """Fraction of boxes that leave the image under each reading of ``labels_with_ids``."""
    out = {"x1y1wh": 0, "cxcywh": 0, "n": 0}
    idx = np.linspace(0, len(video) - 1, min(n_frames, len(video))).astype(int) if len(video) else []
    for r in idx:
        lab = video.raw_labels(int(r))
        if not len(lab):
            continue
        x, y, w, h = lab[:, 2], lab[:, 3], lab[:, 4], lab[:, 5]
        out["x1y1wh"] += int(((x + w > 1.02) | (y + h > 1.02)).sum())
        out["cxcywh"] += int(((x - w / 2 < -0.02) | (y - h / 2 < -0.02) | (x + w / 2 > 1.02) |
                              (y + h / 2 > 1.02)).sum())
        out["n"] += len(lab)
    n = max(out["n"], 1)
    out["x1y1wh"] /= n
    out["cxcywh"] /= n
    out["best"] = "x1y1wh" if out["x1y1wh"] <= out["cxcywh"] else "cxcywh"
    return out


# ---------------------------------------------------------------------- discovery
def _walk_dirs(root: str, max_depth: int = 5):
    root = os.path.abspath(root)
    base = root.count(os.sep)
    for d, subdirs, files in os.walk(root, followlinks=True):
        if d.count(os.sep) - base >= max_depth:
            subdirs[:] = []
        # macOS zip debris (__MACOSX/, ._* AppleDouble files) and hidden caches are never data
        subdirs[:] = sorted(x for x in subdirs if x != "__MACOSX" and not x.startswith("."))
        files[:] = [f for f in files if not f.startswith("._")]
        yield d, subdirs, files


@lru_cache(maxsize=8)
def _index(root: str) -> dict:
    """video_id -> {"images": dir, "labels": dir}; one walk per dataset root."""
    out: dict = {}
    for d, _, files in _walk_dirs(root):
        name = os.path.basename(d)
        low = d.lower()
        has_img = any(f.lower().endswith(IMG_EXT) for f in files[:50])
        has_txt = any(f.endswith(".txt") for f in files[:50])
        if "labels_with_ids" in low and has_txt:
            out.setdefault(name, {})["labels"] = d
        elif has_img and "labels" not in low:
            e = out.setdefault(name, {})
            # KITTI has training/image_02/0005 AND testing/image_02/0005: the RMOT labels belong to training
            if "images" not in e or ("testing" in e["images"].lower() and "testing" not in low):
                e["images"] = d
    return out


def _find_expression_root(root: str) -> str:
    for d, subdirs, _ in _walk_dirs(root, 3):
        if os.path.basename(d) == "expression":
            return d
    raise FileNotFoundError(f"no 'expression' folder under {root}")


def load_expressions(expr_root: str, video_id: str) -> list[RMOTExpression]:
    d = os.path.join(expr_root, video_id)
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json") or fn.startswith("._"):
            continue
        with open(os.path.join(d, fn)) as f:
            info = json.load(f)
        lab = {int(k): [int(i) for i in v] for k, v in (info.get("label") or {}).items()}
        out.append(RMOTExpression(video_id, fn[:-5], lab, os.path.join(d, fn), info.get("sentence", "")))
    return out


@dataclass
class RMOTSplit:
    dataset: str
    split: str
    root: str
    videos: dict                    # video_id -> RMOTVideo
    expressions: dict               # video_id -> [RMOTExpression]
    seqmap: list                    # ["video+expr", ...] in official order

    def seq_lengths(self) -> dict:
        return {v: len(x) for v, x in self.videos.items()}

    def write_seqmap(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write("".join(s + "\n" for s in self.seqmap))

    def items(self, max_expr_per_video: int | None = None):
        for vid in self.videos:
            ex = self.expressions[vid]
            if max_expr_per_video:
                ex = ex[:max_expr_per_video]
            for e in ex:
                yield self.videos[vid], e


def _build(dataset: str, split: str, root: str, video_ids: list, seqmap_file: str | None,
           label_format: str, fps: float | None) -> RMOTSplit:
    idx = _index(os.path.abspath(root))
    expr_root = _find_expression_root(root)
    videos, exprs, missing = {}, {}, []
    for vid in video_ids:
        e = idx.get(vid, {})
        if "images" not in e or not os.path.isdir(os.path.join(expr_root, vid)):
            missing.append(vid)
            continue
        imgs = sorted(f for f in os.listdir(e["images"]) if f.lower().endswith(IMG_EXT) and not f.startswith("._"))
        keys = [_digits(os.path.splitext(f)[0]) for f in imgs]
        v = RMOTVideo(dataset, vid, e["images"], e.get("labels", ""),
                      [os.path.join(e["images"], f) for f in imgs], keys,
                      fps or FPS.get(dataset, 30.0), label_format)
        videos[vid] = v
        exprs[vid] = load_expressions(expr_root, vid)
    if missing:
        warnings.warn(f"{dataset}/{split}: {len(missing)} test videos not found under {root}: {missing[:5]}")
    if seqmap_file and os.path.isfile(seqmap_file):
        from .hota import read_seqmap
        seqmap = [s for s in read_seqmap(seqmap_file) if s.split("+", 1)[0] in videos]
        by = {e.seq_name: e for v in exprs.values() for e in v}
        lost = [s for s in seqmap if s not in by]
        if lost:
            warnings.warn(f"{len(lost)} seqmap rows have no expression file, e.g. {lost[:3]}")
        keep = set(seqmap)
        for v in exprs:
            exprs[v] = [e for e in exprs[v] if e.seq_name in keep]
    else:
        seqmap = [e.seq_name for v in videos for e in exprs[v]]
    return RMOTSplit(dataset, split, root, videos, exprs, seqmap)


def aerialmind(root: str, split: str = "in_domain", seqmap_file: str | None = None,
               label_format: str = "auto", fps: float | None = None) -> RMOTSplit:
    """``split``: in_domain (VisDrone test sequences) | cross_domain (UAVDT) | <comma list of ids>."""
    vids = AERIALMIND_TEST.get(split) or [s for s in split.split(",") if s]
    sp = _build("aerialmind", split, root, vids, seqmap_file, label_format, fps)
    want = AERIALMIND_SEQMAP_ROWS.get(split)
    if want and len(sp.seqmap) != want and len(sp.videos) == len(vids):
        warnings.warn(f"AerialMind {split}: {len(sp.seqmap)} expressions, official seqmap has {want}. "
                      "Pass --seqmap with the official file to evaluate exactly the official rows.")
    return sp


def refer_kitti(root: str, version: str = "v2", split: str = "test", seqmap_file: str | None = None,
                fps: float | None = None) -> RMOTSplit:
    name = "refer_kitti_v2" if version == "v2" else "refer_kitti"
    vids = REFER_KITTI_TEST[name] if split == "test" else [s for s in split.split(",") if s]
    return _build(name, split, root, vids, seqmap_file, "x1y1wh", fps)


def load_split(dataset: str, root: str, split: str, seqmap_file: str | None = None, **kw) -> RMOTSplit:
    if dataset == "aerialmind":
        return aerialmind(root, split, seqmap_file, **kw)
    if dataset in ("refer_kitti_v2", "refer_kitti"):
        return refer_kitti(root, "v2" if dataset.endswith("v2") else "v1",
                           "test" if split in ("test", "in_domain") else split, seqmap_file, **kw)
    raise ValueError(f"unknown RMOT dataset {dataset!r}")
