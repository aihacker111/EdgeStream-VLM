"""Benchmark the streaming VAD pipeline: frame-level AUC/AP vs VLM budget (and A2Seek answers).

Examples
--------
# UCF-Crime test set (290 videos), budgets 0 (cheap tier only), 2, 6, 12 VLM calls/min
python scripts/vad_benchmark.py --dataset ucf_crime --videos data/UCF_Crimes/Videos \
    --annotations data/UCF_Crimes/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
    --config configs/vad_cctv.yaml --vlm openai --budgets 0 2 6 12 --stride 2 --out runs/bench_ucf

# XD-Violence
python scripts/vad_benchmark.py --dataset xd_violence --videos data/XD/test_videos \
    --annotations data/XD/annotations.txt --config configs/vad_cctv.yaml --out runs/bench_xd

# Any frame-label dataset (UIT-ADrone, Drone-Anomaly, MUVAD after conversion)
python scripts/vad_benchmark.py --dataset generic --videos data/UIT-ADrone/test/videos \
    --labels data/UIT-ADrone/test/labels --fps 30 --config configs/vad_uav.yaml --out runs/bench_uit

# A2Seek (needs the optional A2Seek-Videos release); writes predictions for A2Seek's evaluate.py
python scripts/vad_benchmark.py --dataset a2seek --a2seek-samples data/A2Seek/dataset/cot_single/test.jsonl \
    --videos "data/A2Seek-Videos/{g_id}/{v_id}.mp4" --config configs/vad_uav.yaml --out runs/bench_a2seek

Perception runs once per video (tracks are cached under OUT/cache); every budget replays the cache,
so the budget sweep costs only the cheap tier + the VLM calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgestream.harness.perception_cache import CachedPerception  # noqa: E402
from edgestream.vad import category_report, vad_report  # noqa: E402
from edgestream.vad import datasets as D  # noqa: E402
from edgestream.vad.normality import KinematicBank  # noqa: E402
from edgestream.vad.pipeline import VADPipeline, load_vad_config  # noqa: E402


def load_dataset(a) -> list:
    if a.dataset == "ucf_crime":
        return D.ucf_crime(a.videos, a.annotations)
    if a.dataset == "xd_violence":
        return D.xd_violence(a.videos, a.annotations, a.test_list)
    if a.dataset == "generic":
        return D.generic_frame_labels(a.videos, a.labels, fps=a.fps or 30.0)
    if a.dataset == "a2seek":
        return D.a2seek(a.a2seek_samples, a.videos, frame_scale=a.a2seek_frame_scale)
    if a.dataset == "drone_anomaly":
        return D.generic_frame_labels(a.videos, a.labels, fps=a.fps or 30.0)
    raise ValueError(a.dataset)


def _safe(name: str) -> str:
    return name.replace("/", "__")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["ucf_crime", "xd_violence", "generic", "drone_anomaly", "a2seek"])
    ap.add_argument("--videos", required=True, help="video root (a2seek: path template with {g_id} {v_id})")
    ap.add_argument("--annotations")
    ap.add_argument("--labels", help="generic: folder of per-video label files")
    ap.add_argument("--test-list")
    ap.add_argument("--a2seek-samples")
    ap.add_argument("--a2seek-frame-scale", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--config", default="configs/vad_uav.yaml")
    ap.add_argument("--vlm", choices=["simulated", "oracle", "openai", "transformers"], default=None)
    ap.add_argument("--vlm-url", default=None, help="OpenAI-compatible base URL (Ollama/vLLM)")
    ap.add_argument("--vlm-model", default=None, help="model name on that server, e.g. qwen3-vl:4b")
    ap.add_argument("--budgets", type=float, nargs="+", default=[6.0], help="VLM calls per minute")
    ap.add_argument("--stride", type=int, default=1, help="process every k-th frame")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--bank", default=None, help="normal kinematic bank (.npz) from fit_normal_bank.py")
    ap.add_argument("--freeze-bank", action="store_true", help="do not adapt the bank online")
    ap.add_argument("--out", default="runs/vad_bench")
    a = ap.parse_args(argv)

    cfg = load_vad_config(a.config)
    if a.vlm:
        cfg["vlm"]["backend"] = a.vlm
    if a.vlm_url:
        cfg["vlm"]["openai"]["base_url"] = a.vlm_url
    if a.vlm_model:
        cfg["vlm"]["openai"]["model"] = a.vlm_model
        cfg["vlm"]["transformers"]["model_id"] = a.vlm_model if "/" in a.vlm_model else \
            cfg["vlm"]["transformers"]["model_id"]
    videos = load_dataset(a)[: a.max_videos]
    if not videos:
        sys.exit("No videos found: check --videos / --annotations / --labels")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(videos)} videos, budgets {a.budgets}, stride {a.stride}, VLM {cfg['vlm']['backend']}")

    rows = []
    shared_vlm = None                    # load a local model once for the whole budget sweep
    capacity = cfg["gate"].get("bucket_capacity", 3.0)
    for budget in a.budgets:
        cfg["gate"]["calls_per_min"] = budget
        cfg["gate"]["bucket_capacity"] = 0.0 if budget <= 0 else capacity   # budget 0 must not leak
        pipe = VADPipeline(cfg, vlm=shared_vlm)
        shared_vlm = pipe.vlm
        per_on, per_off, preds, cats, calls, dur = {}, {}, [], [], 0, 0.0
        for k, v in enumerate(videos):
            cache_dir = out / "cache" / f"stride{a.stride}" / _safe(v.name)
            cache = cache_dir / "tracks.jsonl"
            perceptor, record = None, False
            if cache.exists() and cache.stat().st_size > 0:
                perceptor = CachedPerception(str(cache))
            else:
                cache_dir.mkdir(parents=True, exist_ok=True)
                record = True
            bank = KinematicBank(**cfg.get("normality", {}).get("knn", {}))
            if a.bank:
                bank.load(a.bank)
            run_dir = out / f"budget_{budget:g}" / _safe(v.name)
            try:
                r = pipe.run(v.source, out_dir=str(run_dir), labels=v.labels, fps=v.fps or a.fps,
                             stride=a.stride, probes=v.probes, max_frames=a.max_frames,
                             perceptor=perceptor, record_tracks=False, bank=bank, freeze_bank=a.freeze_bank) \
                    if not record else _run_and_record(pipe, v, run_dir, cache, a, bank)
            except (FileNotFoundError, IndexError) as e:
                print(f"  skip {v.name}: {e}")
                continue
            calls += r["budget"]["vlm_calls"]
            dur += r["duration_s"]
            if v.labels is not None:
                n = min(len(v.labels), len(r["scores_online"]))
                per_on[v.name] = (r["scores_online"][:n], v.labels[:n])
                per_off[v.name] = (r["scores_offline"][:n], v.labels[:n])
            for ans, pr in zip(r["probe_answers"], sorted(v.probes, key=lambda p: p["frame"])):
                preds.append(ans)
                cats.append({"gt_category": _alias(pr["gt_category"]), "pred_category": _alias(ans["category"]),
                             "gt_box": pr.get("gt_box"), "pred_box": _box(ans["prediction"]),
                             "scene": pr.get("scene", "00")})
            print(f"  [{budget:g}/min] {k + 1}/{len(videos)} {v.name}: {r['processed_frames']} frames, "
                  f"{r['budget']['vlm_calls']} VLM calls, {r['wall_s']}s")
        row = {"budget_calls_per_min": budget, "videos": len(per_on) or len(videos),
               "vlm_calls": calls, "calls_per_min_used": round(60 * calls / max(dur, 1e-6), 2)}
        if per_on:
            ron, roff = vad_report(per_on), vad_report(per_off)
            for tag, rep in (("online", ron), ("offline", roff)):
                for m in ("micro_auc", "macro_auc", "ap", "auc_abnormal_videos", "eer"):
                    row[f"{tag}_{m}"] = round(rep[m], 4) if rep[m] == rep[m] else None
            with open(out / f"budget_{budget:g}" / "per_video_auc.json", "w") as f:
                json.dump({"online": ron["per_video_auc"], "offline": roff["per_video_auc"]}, f, indent=1)
        if preds:
            with open(out / f"budget_{budget:g}" / "a2seek_predictions.jsonl", "w") as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            row.update({f"a2seek_{k}": v for k, v in category_report(cats).items()})
        rows.append(row)
        print(json.dumps(row, indent=1))

    with open(out / "metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "budget_calls_per_min", k))
    with open(out / "budget_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out / 'metrics.json'}, {out / 'budget_curve.csv'}")


def _run_and_record(pipe, v, run_dir, cache, a, bank):
    r = pipe.run(v.source, out_dir=str(run_dir), labels=v.labels, fps=v.fps or a.fps, stride=a.stride,
                 probes=v.probes, max_frames=a.max_frames, record_tracks=True, bank=bank,
                 freeze_bank=a.freeze_bank)
    src = Path(run_dir) / "tracks.jsonl"
    if src.exists():
        src.replace(cache)
    return r


A2SEEK_ALIASES = {"Trespassing on Lawn": "Treading on Grass"}   # A2Seek's label_aliases.json


def _alias(name: str) -> str:
    return A2SEEK_ALIASES.get(name, name)


def _box(pred: str):
    import ast
    import re
    m = re.search(r"<answer>(.*)</answer>", pred, re.S)
    try:
        _, boxes = m.group(1).split(";", 1)
        b = ast.literal_eval(boxes.strip())
        return b[0] if b else None
    except (AttributeError, ValueError, SyntaxError, IndexError):
        return None


if __name__ == "__main__":
    main()
