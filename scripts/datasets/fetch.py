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


def _describe(z: Path) -> str:
    size = z.stat().st_size
    with open(z, "rb") as f:
        head = f.read(64)
    kind = "zip" if head[:2] == b"PK" else "7z" if head[:6] == b"7z\xbc\xaf\x27\x1c" else \
        "rar" if head[:4] == b"Rar!" else "gzip" if head[:2] == b"\x1f\x8b" else \
        "git-lfs/xet pointer (text, not the data)" if head.startswith(b"version https://git-lfs") else \
        "html (an error page, not the data)" if head.lstrip().lower().startswith((b"<!doctype", b"<html")) else "unknown"
    return f"{size / 1e9:.2f} GB, header looks like: {kind}"


def _split_parts(z: Path) -> list:
    """Other volumes of a split archive (name.z01.., name.zip.001.., name.7z.001.., name.partN.rar)."""
    stem = re.sub(r"\.(zip|7z|rar)(\.\d+)?$|\.z\d\d$|\.part\d+$", "", z.name, flags=re.I)
    return sorted(p for p in z.parent.iterdir() if p != z and p.name.startswith(stem) and
                  re.search(r"\.(z\d\d|zip\.\d+|7z\.\d+|part\d+(\.rar)?|\d{3})$", p.name, re.I))


def _cli_extract(z: Path, dest: Path) -> bool:
    """7z handles split zip/7z/rar volumes; unzip handles what Python's zipfile refuses."""
    if shutil.which("7z") or shutil.which("7za"):
        exe = shutil.which("7z") or shutil.which("7za")
        return subprocess.run([exe, "x", "-y", f"-o{dest}", str(z)]).returncode == 0
    if z.name.lower().endswith(".zip") and shutil.which("unzip") and not _split_parts(z):
        return subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(dest)]).returncode == 0
    if z.name.lower().endswith(".zip") and shutil.which("zip") and _split_parts(z):
        full = z.with_name(z.stem + ".joined_tmp.zip")      # zip --out appends .zip otherwise
        try:
            if subprocess.run(["zip", "-q", "-s", "0", str(z), "--out", str(full)]).returncode != 0:
                return False
            if shutil.which("unzip"):          # python's zipfile rejects re-joined archives ("overlapped entries")
                return subprocess.run(["unzip", "-o", "-q", str(full), "-d", str(dest)]).returncode == 0
            with zipfile.ZipFile(full) as f:
                f.extractall(dest)
            return True
        except (zipfile.BadZipFile, OSError):
            return False
        finally:
            if full.exists():
                full.unlink()
    return False


# ---------------------------------------------------------------------------------------- referring MOT
AERIALMIND_HF = "shawnliang0420/AerialMind"
AERIALMIND_GIT = "https://github.com/shawnliang420/AerialMind.git"
TEMPRMOT_GIT = "https://github.com/zyn213/TempRMOT.git"
RK2_DRIVE_ID = "1eaxuRK-ewl0cpGshOxylSFZ5PPu3_WUT"
KITTI_IMAGES = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip"


def _extract_archives(root: Path) -> None:
    """Extract every archive under ``root`` once (marker files). Never aborts on one bad file:
    split volumes and formats Python cannot read go to 7z / unzip; what still fails is reported."""
    import tarfile
    failed, done_split = [], set()
    for z in sorted(root.rglob("*")):
        if (not z.is_file() or ".cache" in z.parts or "__MACOSX" in z.parts or z.name.startswith("._")
                or z.name.startswith(".unzipped_") or z.name.startswith(".untarred_")):
            continue
        n = z.name.lower()
        is_vol = re.search(r"\.(z\d\d|zip\.\d+|7z\.\d+|part\d+\.rar|\d{3})$", n)
        if is_vol:                                   # secondary volume: extracted through its first part
            continue
        if not n.endswith((".zip", ".7z", ".rar", ".tar", ".tar.gz", ".tgz")):
            continue
        marker = z.parent / f".unzipped_{z.name}"
        if marker.exists():
            continue
        try:
            if n.endswith(".zip") and not _split_parts(z):
                unzip_once(z, z.parent)
                continue
            if n.endswith((".tar", ".tar.gz", ".tgz")):
                log(f"untar {z.name}")
                with tarfile.open(z) as t:
                    t.extractall(z.parent)
                marker.touch()
                continue
            raise zipfile.BadZipFile("needs an external extractor")
        except (zipfile.BadZipFile, NotImplementedError, tarfile.TarError, OSError) as e:
            parts = _split_parts(z)
            log(f"{z.name}: python cannot extract it ({e}); {_describe(z)}"
                + (f"; split archive with {len(parts) + 1} volumes" if parts else ""))
            if _cli_extract(z, z.parent):
                marker.touch()
                done_split.add(z.name)
                log(f"extracted {z.name} with an external tool")
            else:
                failed.append(z)
    # first volumes named .zip.001 / .7z.001 (no plain .zip): extract via 7z
    for z in sorted(root.rglob("*.001")):
        marker = z.parent / f".unzipped_{z.name}"
        if marker.exists() or ".cache" in z.parts or "__MACOSX" in z.parts or z.name.startswith("._"):
            continue
        log(f"split archive {z.name}: extracting with 7z")
        if _cli_extract(z, z.parent):
            marker.touch()
        else:
            failed.append(z)
    if failed:
        log("COULD NOT EXTRACT:")
        for z in failed:
            log(f"   {z}  ({_describe(z)})")
        log("Fix: install 7-Zip (sudo apt install p7zip-full, or conda install -c conda-forge p7zip) and rerun; "
            "if a file is a pointer/html or much smaller than on the Hub, delete it and rerun to re-download.")


def _git_file(repo: str, rel: list, dest: Path) -> None:
    """Fetch single files of a GitHub repo (raw download; no clone).

    Cloning is avoided on purpose: some repos contain paths that NTFS/exFAT disks cannot hold
    (e.g. the AerialMind repo has a folder named "Annotation " with a trailing space), which makes
    ``git clone`` fail at checkout. Fallback: sparse checkout of exactly the requested files."""
    import urllib.request
    owner_repo = repo.rstrip("/").removesuffix(".git").split("github.com/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    missing = []
    for r in rel:
        out = dest / Path(r).name
        ok = False
        for branch in ("main", "master"):
            url = f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{r}"
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    out.write_bytes(resp.read())
                log(f"{url} -> {out}")
                ok = True
                break
            except Exception:                                   # noqa: BLE001
                continue
        if not ok:
            missing.append(r)
    if not missing:
        return
    import tempfile
    with tempfile.TemporaryDirectory() as td:                  # temp dir lives on the system disk
        tp = Path(td) / "repo"
        log(f"raw download failed for {missing}; sparse checkout of {repo}")
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", repo, str(tp)], check=True)
        subprocess.run(["git", "-C", str(tp), "sparse-checkout", "set", "--no-cone", *missing], check=True)
        subprocess.run(["git", "-C", str(tp), "checkout"], check=True)
        for r in missing:
            src = tp / r
            if src.is_file():
                shutil.copy(src, dest / src.name)
                log(f"{src.name} -> {dest / src.name}")
            else:
                log(f"WARNING {r} not found in {repo}")


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
