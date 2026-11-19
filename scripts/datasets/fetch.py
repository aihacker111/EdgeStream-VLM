"""Download the referring-MOT benchmarks into the layouts ``scripts/rmot_benchmark.py`` reads.

Called by ``scripts/run_benchmarks.sh``; every sub-command can also be run alone:

    python scripts/datasets/fetch.py aerialmind      --out data/AerialMind          (UAV, primary)
    python scripts/datasets/fetch.py refer-kitti-v2  --out data/refer-kitti-v2      (ground, generality)

Sources (checked Nov 2026):
  AerialMind      HF dataset shawnliang0420/AerialMind (images, labels_with_ids, expression); official
                  seqmaps from github.com/shawnliang420/AerialMind (datasets/data_path/seqmap{,_cd}.txt)
  Refer-KITTI-V2  KITTI tracking left images (data_tracking_image_2.zip, cvlibs / AWS mirror) +
                  expression & labels_with_ids from the TempRMOT Google Drive folder; seqmap from
                  github.com/zyn213/TempRMOT (datasets/data_path/seqmap.txt)
Respect each dataset's licence.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


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


# ---------------------------------------------------------------------------------------- referring MOT
AERIALMIND_HF = "shawnliang0420/AerialMind"
AERIALMIND_GIT = "https://github.com/shawnliang420/AerialMind.git"
TEMPRMOT_GIT = "https://github.com/zyn213/TempRMOT.git"
RK2_DRIVE_ID = "1eaxuRK-ewl0cpGshOxylSFZ5PPu3_WUT"
KITTI_IMAGES = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip"


def _extract_archives(root: Path) -> None:
    import tarfile
    for z in sorted(root.rglob("*")):
        if not z.is_file():
            continue
        n = z.name.lower()
        if n.endswith(".zip"):
            unzip_once(z, z.parent)
        elif n.endswith((".tar", ".tar.gz", ".tgz")):
            marker = z.parent / f".untarred_{z.name}"
            if not marker.exists():
                log(f"untar {z.name}")
                with tarfile.open(z) as t:
                    t.extractall(z.parent)
                marker.touch()
        elif n.endswith((".7z", ".rar")) or re.search(r"\.(z\d\d|7z\.\d+|part\d+)$", n):
            log(f"{z} is a {n.rsplit('.', 1)[-1]} archive: extract it manually (7z x)")


def _git_file(repo: str, rel: list, dest: Path) -> None:
    """Shallow-clone a GitHub repo into third_party/ and copy files out of it."""
    tp = Path("third_party") / repo.rstrip("/").split("/")[-1].replace(".git", "")
    if not tp.exists():
        tp.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", repo, str(tp)], check=True)
    dest.mkdir(parents=True, exist_ok=True)
    for r in rel:
        src = tp / r
        if src.is_file():
            shutil.copy(src, dest / src.name)
            log(f"{src} -> {dest / src.name}")
        else:
            log(f"WARNING {src} not found in {repo}")


def cmd_aerialmind(a) -> None:
    out = Path(a.out)
    try:
        log(f"HF dataset {AERIALMIND_HF} -> {out}")
        hf_download(AERIALMIND_HF, out)
    except Exception as e:                                       # noqa: BLE001
        sys.exit(f"HF download failed ({e}). Alternative: Baidu Netdisk link in the AerialMind README; "
                 f"unpack into {out} so that it contains expression/ and labels_with_ids/.")
    _extract_archives(out)
    _git_file(AERIALMIND_GIT, ["datasets/data_path/seqmap.txt", "datasets/data_path/seqmap_cd.txt"], out)
    have = [p.name for p in out.iterdir()]
    log(f"{out}: {sorted(have)[:12]}")
    log("check: python -c \"from edgestream.rmot.datasets import aerialmind; "
        f"s=aerialmind('{out}','in_domain','{out}/seqmap.txt'); print(len(s.videos), len(s.seqmap))\"")


def cmd_refer_kitti_v2(a) -> None:
    import os
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kitti = out / "KITTI"
    if not (kitti / "training" / "image_02").is_dir():
        z = Path(a.kitti_zip) if a.kitti_zip else out / "data_tracking_image_2.zip"
        if not z.is_file():
            log(f"KITTI tracking images (~15 GB) {KITTI_IMAGES} -> {z}")
            subprocess.run(["curl", "-L", "-C", "-", "-o", str(z), KITTI_IMAGES], check=True)
        kitti.mkdir(parents=True, exist_ok=True)
        unzip_once(z, kitti)
    if not (out / "expression").is_dir():
        remote = os.environ.get("RCLONE_REMOTE", "gdrive")
        ok = False
        if shutil.which("rclone"):
            remotes = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True).stdout.split()
            if f"{remote}:" in remotes:
                subprocess.run(["rclone", "copy", f"{remote}:", str(out), "--drive-root-folder-id", RK2_DRIVE_ID,
                                "--transfers", "16", "-P"], check=True)
                ok = True
        if not ok:
            try:
                subprocess.run([sys.executable, "-m", "gdown", "--folder", "--remaining-ok",
                                f"https://drive.google.com/drive/folders/{RK2_DRIVE_ID}", "-O", str(out)], check=True)
            except subprocess.CalledProcessError:
                sys.exit(f"Download the Refer-KITTI-V2 Drive folder {RK2_DRIVE_ID} manually into {out}")
        _extract_archives(out)
        lw = list(out.rglob("labels_with_ids"))
        if lw and not (kitti / "labels_with_ids").exists():
            shutil.move(str(lw[0]), str(kitti / "labels_with_ids"))
    _git_file(TEMPRMOT_GIT, ["datasets/data_path/seqmap.txt"], out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("aerialmind")
    p.add_argument("--out", default="data/AerialMind")
    p = sub.add_parser("refer-kitti-v2")
    p.add_argument("--out", default="data/refer-kitti-v2")
    p.add_argument("--kitti-zip", default=None, help="already downloaded data_tracking_image_2.zip")
    a = ap.parse_args(argv)
    {"aerialmind": cmd_aerialmind, "refer-kitti-v2": cmd_refer_kitti_v2}[a.cmd](a)


if __name__ == "__main__":
    main()
