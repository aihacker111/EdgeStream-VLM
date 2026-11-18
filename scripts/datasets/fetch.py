"""Download / convert the VAD benchmarks into the layouts ``scripts/vad_benchmark.py`` reads.

Called by ``scripts/run_benchmarks.sh``; every sub-command can also be run alone:

    python scripts/datasets/fetch.py a2seek        --root data/A2Seek
    python scripts/datasets/fetch.py a2seek-videos --root data/A2Seek --videos data/A2Seek-Videos --split test
    python scripts/datasets/fetch.py xd-violence   --out data/XD-Violence
    python scripts/datasets/fetch.py ucf-crime     --out data/UCF-Crime [--with-train-normal]
    python scripts/datasets/fetch.py uit-adrone    --out data/UIT-ADrone          (Google Drive, needs gdown)
    python scripts/datasets/fetch.py frame-labels  --src data/Drone-Anomaly --out data/Drone-Anomaly/edgestream
    python scripts/datasets/fetch.py frame-labels  --src data/UIT-ADrone    --out data/UIT-ADrone/edgestream

Sources (checked Nov 2026):
  A2Seek          HF datasets tiu2mo/A2Seek (annotations + image bank) and tiu2mo/A2Seek-Videos (videos)
  XD-Violence     HF mirror jherng/xd-violence  (data/test_list.txt, data/test_annotations.txt, data/video/*.mp4)
  UCF-Crime       HF mirror jinmang2/ucf_crime  (Anomaly-Videos-Part-{1..4}.zip, Testing_Normal_Videos.zip,
                  UCF_Crimes-Train-Test-Split/ with Temporal_Anomaly_Annotation_for_Testing_Videos.txt)
  UIT-ADrone      Google Drive folder from https://uit-together.github.io/datasets/UIT-ADrone
                  (frame labels: one .npy per test video, 0 normal / 1 abnormal)
  Drone-Anomaly   OneDrive link in https://github.com/Jin-Pu/Drone-Anomaly (manual download);
                  <scene>/<sequence>/{train,test}/, test frame folders 01, 02 ... + labels 1.npy, 2.npy ...
  MUVAD           announced at https://github.com/uavano/FTDMamba, not released yet
The mirrors are third-party copies of the official releases; respect each dataset's licence.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

VIDEO_EXT = (".mp4", ".avi", ".mkv", ".mov", ".webm")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
UIT_DRIVE = "https://drive.google.com/drive/folders/1ygWcuOZ7kEwS-rC35gWv6f6qeNx0BHyg"


def log(msg: str) -> None:
    print(f"[fetch] {msg}", flush=True)


def hf_download(repo: str, local_dir: Path, patterns=None) -> None:
    import os
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")      # many parallel streams (hf_xet)
    from huggingface_hub import snapshot_download
    local_dir.mkdir(parents=True, exist_ok=True)
    # resumable: files already in local_dir are skipped, interrupted ones continue
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(local_dir),
                      allow_patterns=patterns, max_workers=int(os.environ.get("HF_WORKERS", "16")))


def unzip_once(z: Path, dest: Path) -> None:
    marker = dest / f".unzipped_{z.name}"
    if marker.exists():
        return
    log(f"unzip {z.name} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(dest)
    marker.touch()


# ---------------------------------------------------------------------------------------- A2Seek
def cmd_a2seek(a) -> None:
    root = Path(a.root)
    hf_download("tiu2mo/A2Seek", root)
    n = materialize_a2seek_images(root)
    log(f"A2Seek annotations in {root / 'dataset'}; {n} images restored")


def materialize_a2seek_images(root: Path) -> int:
    """Restore images/ from the Parquet image bank (same layout as A2Seek's materialize_images.py)."""
    shards = sorted((root / "data" / "images").glob("*.parquet"))
    marker = root / ".local" / "edgestream-images.done"
    if not shards or marker.exists():
        return 0
    import hashlib

    import pyarrow.parquet as pq
    count = 0
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                rel = Path(row["path"])
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"unsafe image path in bank: {rel}")
                target = root / rel if rel.parts[0] == "images" else root / "images" / rel
                data = row["image"]["bytes"]
                if data is None or hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError(f"checksum mismatch: {rel}")
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                count += 1
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(count))
    return count


_GROUP = re.compile(r"video_group_\d+")
_VIDEO = re.compile(r"(video_\d+)\.(mp4|mov|avi|mkv)$", re.I)


def _clip_key(path: str):
    g, v = _GROUP.search(path), _VIDEO.search(path)
    return f"{g.group(0)}/{v.group(1)}" if g and v else None


def cmd_a2seek_videos(a) -> None:
    """Download only the videos of the clips used by one split, then write
    ``<root>/edgestream_videos_<split>.json`` = {"g_id/v_id": local video path, "_frame_scale": s}."""
    from huggingface_hub import HfApi
    root, vroot = Path(a.root), Path(a.videos)
    samples = root / "dataset" / a.variant / f"{a.split}.jsonl"
    rows = [json.loads(x) for x in samples.read_text().splitlines() if x.strip()]
    wanted = {f"{r['g_id']}/{r['v_id']}" for r in rows}
    if a.max_clips:                               # quick test: a few clips spread over the split
        ordered = sorted(wanted)
        wanted = set(ordered[:: max(1, len(ordered) // a.max_clips)][: a.max_clips])
    files = HfApi().list_repo_files("tiu2mo/A2Seek-Videos", repo_type="dataset")
    by_clip = {}
    for f in files:
        k = _clip_key(f)
        if k in wanted:
            by_clip.setdefault(k, f)
    missing = sorted(wanted - set(by_clip))
    log(f"{len(wanted)} clips in {a.split}; {len(by_clip)} found in A2Seek-Videos; {len(missing)} missing")
    if missing:
        log(f"  e.g. missing {missing[:3]} — check the repo layout if this number is large")
    hf_download("tiu2mo/A2Seek-Videos", vroot, patterns=sorted(by_clip.values()))
    mapping = {k: str((vroot / f).resolve()) for k, f in by_clip.items() if (vroot / f).exists()}
    scale = estimate_frame_scale(rows, mapping, root)
    out = root / f"edgestream_videos_{a.split}.json"
    out.write_text(json.dumps({**mapping, "_frame_scale": scale}, indent=1))
    log(f"wrote {out} ({len(mapping)} videos, frame scale {scale})")


def estimate_frame_scale(rows, mapping, root: Path, n: int = 12) -> float:
    """f_id -> video frame index. Compare sample images with video frames at f_id * s for candidate s
    and keep the s with the smallest median difference (images must be materialized)."""
    import cv2
    import numpy as np
    cands = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30]
    picked = [r for r in rows if r.get("images") and f"{r['g_id']}/{r['v_id']}" in mapping][:: max(1, len(rows) // 200)][:n]
    errs = {s: [] for s in cands}

    def small(img):
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)

    for r in picked:
        p = Path(r["images"][0])
        img = cv2.imread(str(p if p.is_absolute() else root / p))
        if img is None:
            continue
        ref = small(img)
        cap = cv2.VideoCapture(mapping[f"{r['g_id']}/{r['v_id']}"])
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for s in cands:
            k = int(round(int(r["f_id"]) * s))
            if k >= nf:
                errs[s].append(1e9)
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, fr = cap.read()
            errs[s].append(float(np.abs(small(fr) - ref).mean()) if ok else 1e9)
        cap.release()
    med = {s: float(np.median(v)) for s, v in errs.items() if v}
    if not med:
        log("frame scale: no materialized images to compare, using 1 (VERIFY by hand)")
        return 1.0
    best = min(med, key=med.get)
    log("frame scale check (median abs diff per scale): " +
        ", ".join(f"{s}:{med[s]:.1f}" for s in sorted(med)[:8]) + f" -> {best}")
    if med[best] > 25:
        log("WARNING: no scale matches well; verify --a2seek-frame-scale by hand")
    return float(best)


# ---------------------------------------------------------------------------------------- XD-Violence
def cmd_xd(a) -> None:
    out = Path(a.out)
    hf_download("jherng/xd-violence", out, patterns=["data/test_list.txt", "data/test_annotations.txt"])
    lst = (out / "data" / "test_list.txt").read_text().split()
    ids = [Path(x).name for x in lst if x.endswith(".mp4")]
    ids = ids[: a.max_videos] if a.max_videos else ids
    log(f"{len(ids)} test videos")
    hf_download("jherng/xd-violence", out, patterns=[f"data/video/{i}" for i in ids])
    log(f"videos in {out / 'data' / 'video'}")


# ---------------------------------------------------------------------------------------- UCF-Crime
def cmd_ucf(a) -> None:
    out = Path(a.out)
    zips = [f"Anomaly-Videos-Part-{i}.zip" for i in range(1, 5)] + ["Testing_Normal_Videos.zip"]
    if a.with_train_normal:
        zips.append("Training-Normal-Videos-Part-1.zip")      # 38 GB, only for the normal motion bank
    hf_download("jinmang2/ucf_crime", out / "zips", patterns=zips + ["UCF_Crimes-Train-Test-Split/*"])
    for z in zips:
        zp = out / "zips" / z
        if zp.exists():
            unzip_once(zp, out / ("train_normal" if "Training" in z else "videos"))
    ann = next((out / "zips").rglob("Temporal_Anomaly_Annotation_for_Testing_Videos.txt"), None)
    if ann is None:
        sys.exit("Temporal_Anomaly_Annotation_for_Testing_Videos.txt not found in the split folder")
    shutil.copy(ann, out / ann.name)
    if not a.keep_zips:
        for z in zips:
            (out / "zips" / z).unlink(missing_ok=True)
    log(f"videos in {out / 'videos'}, annotations {out / ann.name}")


# ---------------------------------------------------------------------------------------- UIT-ADrone
UIT_FOLDER_ID = "1ygWcuOZ7kEwS-rC35gWv6f6qeNx0BHyg"


def cmd_uit(a) -> None:
    """rclone (fast, parallel, resumable) when a Google Drive remote is configured, else gdown."""
    import os
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    remote = os.environ.get("RCLONE_REMOTE", "gdrive")
    if shutil.which("rclone"):
        remotes = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True).stdout.split()
        if f"{remote}:" in remotes:
            log(f"rclone {remote}: (folder {UIT_FOLDER_ID}) -> {out}")
            subprocess.run(["rclone", "copy", f"{remote}:", str(out), "--drive-root-folder-id", UIT_FOLDER_ID,
                            "--transfers", os.environ.get("RCLONE_TRANSFERS", "16"), "--checkers", "16",
                            "--drive-chunk-size", "64M", "--drive-acknowledge-abuse", "-P"], check=True)
            return
        log(f"rclone found but no '{remote}:' remote — run 'rclone config' once (type: drive) for fast downloads")
    try:
        subprocess.run([sys.executable, "-m", "gdown", "--folder", "--remaining-ok", UIT_DRIVE, "-O", str(out)],
                       check=True)
    except subprocess.CalledProcessError:
        sys.exit(f"gdown failed (Drive quota or >50 files per folder). Download {UIT_DRIVE} in a browser "
                 f"and unpack it into {out}, then re-run.")


# ---------------------------------------------------------------------------------------- frame labels
def _natural(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def _media(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXT or (p.is_dir() and any(q.suffix.lower() in IMG_EXT for q in p.iterdir()))


def cmd_frame_labels(a) -> None:
    """Find every directory that holds per-video ``.npy`` labels next to test videos / frame folders,
    pair them (same name if possible, else by sorted order as in the Drone-Anomaly code) and build
    ``out/videos/<name>`` (symlinks) + ``out/labels/<name>.npy`` for the generic loader.
    Also collects normal training media into ``out/train`` for scripts/fit_normal_bank.py."""
    src, out = Path(a.src).resolve(), Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "train").mkdir(parents=True, exist_ok=True)
    import numpy as np
    pairs, label_dirs = [], sorted({p.parent for p in src.rglob("*.npy") if out.resolve() not in p.parents})
    for d in label_dirs:
        labels = sorted(d.glob("*.npy"), key=_natural)
        media = sorted([p for p in d.iterdir() if _media(p)], key=_natural)
        if not media:                         # labels may sit in a sibling folder (e.g. test/labels)
            for sib in sorted(d.parent.iterdir(), key=_natural):
                if sib != d and sib.is_dir():
                    media = sorted([p for p in sib.iterdir() if _media(p)], key=_natural)
                    if media:
                        break
        by_name = {(p.stem if p.is_file() else p.name): p for p in media}
        matched = [(by_name[l.stem], l) for l in labels if l.stem in by_name]
        if len(matched) == len(labels) and labels:
            pairs += matched
        elif len(media) == len(labels) and labels:
            log(f"{d}: pairing {len(labels)} labels with media by sorted order")
            pairs += list(zip(media, labels))
        elif labels:
            log(f"{d}: {len(labels)} labels but {len(media)} videos/frame folders — skipped, check by hand")
    for m, l in pairs:
        rel = m.relative_to(src)
        name = "__".join(rel.parent.parts[-3:] + (m.stem if m.is_file() else m.name,)).replace(" ", "_")
        link = out / "videos" / (name + (m.suffix if m.is_file() else ""))
        if not link.exists():
            link.symlink_to(m)
        y = np.load(l).astype(np.uint8).ravel()
        np.save(out / "labels" / f"{name}.npy", y)
    train = [p for p in src.rglob("*") if _media(p) and "train" in [q.lower() for q in p.relative_to(src).parts[:-1]]
             and out.resolve() not in p.resolve().parents]
    for m in train:
        name = "__".join(m.relative_to(src).parts).replace(" ", "_")
        link = out / "train" / (Path(name).stem + (m.suffix if m.is_file() else ""))
        if not link.exists():
            link.symlink_to(m.resolve())
    log(f"{len(pairs)} labelled test videos -> {out}/videos + labels; {len(train)} training items -> {out}/train")
    if not pairs:
        sys.exit("no (video, label) pairs found")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("a2seek")
    p.add_argument("--root", default="data/A2Seek")
    p = sub.add_parser("a2seek-videos")
    p.add_argument("--root", default="data/A2Seek")
    p.add_argument("--videos", default="data/A2Seek-Videos")
    p.add_argument("--split", default="test")
    p.add_argument("--variant", default="cot_single")
    p.add_argument("--max-clips", type=int, default=None)
    p = sub.add_parser("xd-violence")
    p.add_argument("--out", default="data/XD-Violence")
    p.add_argument("--max-videos", type=int, default=None)
    p = sub.add_parser("ucf-crime")
    p.add_argument("--out", default="data/UCF-Crime")
    p.add_argument("--with-train-normal", action="store_true")
    p.add_argument("--keep-zips", action="store_true")
    p = sub.add_parser("uit-adrone")
    p.add_argument("--out", default="data/UIT-ADrone")
    p = sub.add_parser("frame-labels")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"a2seek": cmd_a2seek, "a2seek-videos": cmd_a2seek_videos, "xd-violence": cmd_xd, "ucf-crime": cmd_ucf,
     "uit-adrone": cmd_uit, "frame-labels": cmd_frame_labels}[a.cmd](a)


if __name__ == "__main__":
    main()
