#!/usr/bin/env python
"""Referring-MOT benchmark: AerialMind (official protocol) and Refer-KITTI-V2, HOTA@L.

    python scripts/rmot_benchmark.py --dataset aerialmind --root data/AerialMind --split in_domain \
        --seqmap data/AerialMind/seqmap.txt --methods search,filter_kf,filter_ef,prior \
        --lags 0,2,5,inf --vlm openai --vlm-url http://localhost:8000/v1 --out runs/rmot_am

Output (one folder per method and lag, official TrackEval layout):
    OUT/<method>_L<lag>/<video_id>/<expression_id>/{gt.txt,predict.txt}
    OUT/<method>_L<lag>/stats.jsonl          VLM calls, tokens, solver status per stream
    OUT/results.json, OUT/summary.md         HOTA / DetA / AssA / LocA per method x lag
    OUT/seqmap_<split>.txt                   seqmap of exactly the evaluated rows

Methods
    search        the method (set-level verifier + window solver + stitching + zoom)
    search_single same as search with the default group_size 1 (kept for old commands)
    search_set    ablation: set-level prompt, 8 candidates per call
    search_nolink ablation: no stitching / no link verification
    search_g4     ablation: 4 candidates per set prompt instead of 8
    search_nofacts, search_single_nofacts   ablations: without the measured-motion text
    filter_kf     YesTrack-style keyframe filter (per-candidate yes/no, track label propagation)
    filter_ef     every-frame filter (keyframe_every 1)
    filter_set    keyframe filter with set prompts (ablation: verifier without search)
    prior         no VLM (class family + measured kinematics only)
    ceiling       the method with a perfect GT verifier (perception ceiling; NOT a result)

Compute matching: online, all methods share the same VLM pool (n_workers, latency model), so
they are matched in VLM time. Offline (L = inf), ``--match-compute`` caps each baseline at the
number of calls the method used on the same stream.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

from edgestream.rmot import hota as H  # noqa: E402
from edgestream.rmot.datasets import check_label_format, load_split  # noqa: E402
from edgestream.rmot.perception import has_cache, load_mot_tracks, load_perception, run_perception  # noqa: E402
from edgestream.rmot.pipeline import INF, GTOracle, OnlineRMOT, RMOTConfig  # noqa: E402
from edgestream.rmot.verifier import CachedVerifier, FrameSource, OracleVerifier, VLMVerifier  # noqa: E402

METHODS = {
    "search": {"method": "search"},
    "search_single": {"method": "search", "group_size": 1},
    "search_set": {"method": "search", "group_size": 8},
    "search_nolink": {"method": "search", "link_verify": False, "graph": {"stitch_gap_s": 0.0}},
    "search_g4": {"method": "search", "group_size": 4},
    "search_nofacts": {"method": "search", "facts": False},
    "search_single_nofacts": {"method": "search", "group_size": 1, "facts": False},
    "filter_kf": {"method": "filter"},
    "filter_ef": {"method": "filter", "keyframe_every": 1},
    "filter_set": {"method": "filter", "filter_prompt": "set", "filter_facts": True},
    "prior": {"method": "prior"},
    "ceiling": {"method": "search", "_oracle": {"acc": 1.0, "conf": 0.99}},
}


def deep_update(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def lag_name(L: float) -> str:
    return "inf" if L == INF else (f"{L:g}")


def build_vlm(cfg: dict, args):
    kind = args.vlm or cfg["vlm"].get("backend", "openai")
    v = cfg["vlm"]
    if kind == "oracle":
        return "oracle", None
    from edgestream.vlm.backends import OpenAICompatBackend, TransformersBackend
    if kind == "openai":
        o = dict(v.get("openai") or {})
        if args.vlm_url:
            o["base_url"] = args.vlm_url
        if args.vlm_model:
            o["model"] = args.vlm_model
        be = OpenAICompatBackend(**o)
    elif kind == "transformers":
        t = dict(v.get("transformers") or {})
        if args.vlm_model:
            t["model_id"] = args.vlm_model
        if getattr(args, "vlm_adapter", None):
            t["adapter"] = args.vlm_adapter
        if getattr(args, "vlm_batch", None):
            t["max_batch"] = args.vlm_batch
        be = TransformersBackend(**t)
    else:
        raise SystemExit(f"unknown --vlm {kind}")
    return kind, VLMVerifier(be, overview_width=v.get("overview_width", 1280), facts=v.get("facts", True),
                             prompt=v.get("prompt", "v2"), frame_width=v.get("frame_width", 1024),
                             crop_h=v.get("crop_h", 160))


def vlm_tag(cfg: dict, args) -> str:
    """Names the answer cache after the backend AND the model (+ adapter): answers of a zero-shot
    model are never served to a fine-tuned one that shares the output folder."""
    kind = args.vlm or cfg["vlm"].get("backend", "openai")
    v = cfg["vlm"]
    if kind == "openai":
        model = args.vlm_model or (v.get("openai") or {}).get("model", "")
    elif kind == "transformers":
        model = args.vlm_model or (v.get("transformers") or {}).get("model_id", "")
    else:
        return kind
    ad = getattr(args, "vlm_adapter", None)
    tag = os.path.basename(os.path.normpath(model)) + (("+" + "_".join(os.path.normpath(ad).split(os.sep)[-2:])) if ad else "")
    return kind + "_" + re.sub(r"[^A-Za-z0-9._+-]+", "-", tag).strip("-")


_W: dict = {}


class _NoBar:
    """Stand-in when tqdm is missing: prints a line every ~30 s."""

    def __init__(self, total=None, desc="", unit="it", **_):
        self.total, self.desc, self.unit, self.n, self._t = total, desc, unit, 0, time.time()
        self.post = ""

    def update(self, k=1):
        self.n += k
        if time.time() - self._t > 30 or self.n == self.total:
            self._t = time.time()
            print(f"{self.desc}: {self.n}/{self.total} {self.unit} {self.post}", flush=True)

    def set_postfix_str(self, s, refresh=True):
        self.post = s

    def set_description(self, d, refresh=True):
        self.desc = d

    def reset(self, total=None):
        self.n, self.total = 0, total

    def write(self, s):
        print(s, flush=True)

    def close(self):
        pass


def bar(total, desc, unit, leave=True, position=0):
    """tqdm progress bar; refreshes rarely when the output is a log file (``| tee``) so logs stay small."""
    try:
        from tqdm import tqdm
    except ImportError:
        return _NoBar(total=total, desc=desc, unit=unit)
    tty = sys.stderr.isatty()
    # piped through `| tee` but a person is at the terminal (stdin is a tty): still refresh every second;
    # only unattended runs (nohup, cron, background) refresh every 30 s to keep logs small
    interval = 0.5 if tty else (1.0 if sys.stdin.isatty() else 30)
    # Some terminals (tmux/screen, IDE or web SSH consoles) report a size of 0 x 0. tqdm then hides
    # every bar (it thinks the screen has 0 rows) and only blank lines appear. Never trust a 0 size.
    import shutil
    size = shutil.get_terminal_size((140, 40))
    cols = size.columns if size.columns >= 60 else 140
    rows = size.lines if size.lines >= 10 else 40
    return tqdm(total=total, desc=desc, unit=unit, leave=leave, position=position, ncols=cols, nrows=rows,
                mininterval=interval, smoothing=0.05, file=sys.stderr)


def _init_worker(state: dict, split) -> None:
    """Per-process state: verifier (+ answer cache), zoom detector, perception of the current video."""
    import argparse as _ap
    a = _ap.Namespace(**state["args"])
    _W.clear()
    _W.update(state)
    _W["split"] = split
    kind, ver = build_vlm(state["cfg"], a) if state["vlm_kind"] != "oracle" else ("oracle", None)
    path = os.path.join(a.out, f"vlm_cache_{vlm_tag(state['cfg'], a)}.jsonl")
    _W["cached"] = CachedVerifier(ver, path) if ver is not None else None
    _W["zoom"] = None
    if not a.no_zoom and state["base"].get("zoom", True) and not a.tracks_dir:
        from perception import Perceptor
        _W["zoom"] = Perceptor.from_yaml(os.path.join(ROOT, state["cfg"]["perception_config"])).detect


def _run_unit(unit) -> list:
    vid, ei = unit
    a, split, fps = _W["args"], _W["split"], _W["fps"]
    video = split.videos[vid]
    W, Hh = video.image_wh
    vc = _W.setdefault("videos", {})                 # small per-process LRU of loaded videos
    if vid not in vc:
        if a["tracks_dir"]:
            pa = load_mot_tracks(os.path.join(a["tracks_dir"], f"{vid}.txt"), len(video), (Hh, W), fps)
        else:
            pa = load_perception(_W["cache_root"], vid)
        vc[vid] = (pa, FrameSource(video.frames, cache=96))
        while len(vc) > 2:
            vc.pop(next(iter(vc)))
    (perc, aff), frames = vc[vid]
    expr = split.expressions[vid][ei]
    if _W.get("cached") is not None:
        _W["cached"].scope = vid                 # answer-cache keys include the video
    gt_rows = video.gt_rows(expr)
    oracle = GTOracle(video, expr)
    calls_search_inf = None
    out_stats = []
    if _W.get("on_unit"):
        _W["on_unit"]()
    order = sorted(_W["methods"], key=lambda m: m != "search")      # the method first (compute matching)
    for m in order:
        spec = copy.deepcopy(METHODS[m])
        orc_spec = spec.pop("_oracle", None)
        for L in sorted(_W["lags"], key=lambda x: x != INF):
            cal = (_W["base"].get("calib_by_method") or {}).get(m)
            rc = RMOTConfig.from_dict(deep_update(_W["base"], {**spec, "lag_s": L, **({"calib": cal} if cal else {})}))
            if m.startswith("filter") and L == INF:
                rc.filter_propagate = "nearest"
            if a["match_compute"] and m != "search" and L == INF and calls_search_inf is not None:
                rc.sched.max_calls = calls_search_inf
            if orc_spec is not None or _W["vlm_kind"] == "oracle":
                o = deep_update((_W["cfg"]["vlm"].get("oracle") or {}), orc_spec or {})
                ver = OracleVerifier(oracle.refer, oracle.same, seed=ei, **o)
            else:
                ver = _W["cached"]
            t0 = time.time()
            if _W.get("on_start"):
                _W["on_start"](f"{vid} | {m}@L{lag_name(L)}")      # what is running NOW
            res = OnlineRMOT(rc, ver, zoom_detect=_W["zoom"]).run(perc, expr.text, fps, (W, Hh), frames, aff)
            if _W.get("on_run"):
                _W["on_run"]()
            if m == "search" and L == INF:
                calls_search_inf = res["stats"]["vlm"]["calls"]
            d = os.path.join(a["out"], f"{m}_L{lag_name(L)}")
            H.write_mot(os.path.join(d, vid, expr.expression_id, "gt.txt"), gt_rows)
            H.write_mot(os.path.join(d, vid, expr.expression_id, "predict.txt"), res["rows"])
            out_stats.append({"seq": expr.seq_name, "method": m, "lag": lag_name(L),
                              "wall_s": round(time.time() - t0, 2),
                              **{k: v for k, v in res["stats"].items() if k != "graph"}})
    return out_stats


def _done_units(out: str, methods, lags) -> set:
    """Streams with a stats line for every method x lag (written after their predict.txt)."""
    cnt: dict = {}
    for m in methods:
        for L in lags:
            p = os.path.join(out, f"{m}_L{lag_name(L)}", "stats.jsonl")
            if os.path.isfile(p):
                with open(p) as f:
                    for line in f:
                        try:
                            cnt[json.loads(line)["seq"]] = cnt.get(json.loads(line)["seq"], 0) + 1
                        except (json.JSONDecodeError, KeyError):
                            pass
    return {s for s, n in cnt.items() if n >= len(methods) * len(lags)}


def verify_with_trackeval(te_path: str, out: str, methods, lags, seqs, lengths) -> float:
    """Recompute HOTA with upstream TrackEval's metric class on the same parsed data."""
    import numpy as np
    if not hasattr(np, "float"):
        np.float = float                         # old TrackEval sources use np.float
    sys.path.insert(0, te_path)
    from trackeval.datasets._base_dataset import _BaseDataset
    from trackeval.metrics.hota import HOTA
    worst = 0.0
    for m in methods:
        for L in lags:
            d = os.path.join(out, f"{m}_L{lag_name(L)}")
            for s in seqs:
                vid, ex = s.split("+", 1)
                sd = H.SeqData.from_rows(H.read_mot(os.path.join(d, vid, ex, "gt.txt")),
                                         H.read_mot(os.path.join(d, vid, ex, "predict.txt")), lengths[vid])
                mine = H.hota_sequence(sd)
                data = {"gt_ids": sd.gt_ids, "tracker_ids": sd.tr_ids,
                        "num_gt_dets": sum(len(x) for x in sd.gt_ids), "num_tracker_dets": sum(len(x) for x in sd.tr_ids),
                        "num_gt_ids": max([int(x.max()) + 1 for x in sd.gt_ids if len(x)] or [0]),
                        "num_tracker_ids": max([int(x.max()) + 1 for x in sd.tr_ids if len(x)] or [0]),
                        "similarity_scores": [_BaseDataset._calculate_box_ious(g.copy(), t.copy())
                                              for g, t in zip(sd.gt_dets, sd.tr_dets)]}
                ref = HOTA().eval_sequence(data)
                worst = max(worst, float(np.max(np.abs(mine["HOTA"] - ref["HOTA"]))))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="aerialmind", choices=["aerialmind", "refer_kitti_v2", "refer_kitti"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="in_domain", help="in_domain | cross_domain | test | comma list of videos")
    ap.add_argument("--seqmap", default=None, help="official seqmap file (rows video+expression)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--label-format", default="auto", choices=["auto", "x1y1wh", "cxcywh"],
                    help="labels_with_ids box convention (auto: top-left unless the data clearly says centre)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="search,filter_kf,filter_ef,prior")
    ap.add_argument("--lags", default="0,2,5,inf")
    ap.add_argument("--vlm", default=None, choices=["openai", "transformers", "oracle"])
    ap.add_argument("--vlm-url", default=None)
    ap.add_argument("--vlm-model", default=None, help="served model name (openai) or HF id / local folder (transformers)")
    ap.add_argument("--vlm-adapter", default=None, help="transformers: LoRA folder from rmot_train_lora.py (e.g. runs/lora_vl2b/best)")
    ap.add_argument("--vlm-batch", type=int, default=None, help="transformers: max requests per forward pass (default 8)")
    ap.add_argument("--belief-params", default=None, help="YAML from rmot_fit_belief.py: enables the belief model")
    ap.add_argument("--no-belief", action="store_true", help="heuristic propagation / coverage instead of the belief model")
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--latency-scale", type=float, default=None)
    ap.add_argument("--fixed-latency", type=float, default=None)
    ap.add_argument("--match-compute", action="store_true", help="offline: cap baselines at the method's calls")
    ap.add_argument("--tracks-dir", default=None, help="external MOT tracks <dir>/<video_id>.txt instead of AERO-MOT")
    ap.add_argument("--perception-cache", default=None, help="default OUT/../perception_cache/<dataset>")
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--max-expr", type=int, default=None, help="expressions per video (smoke tests)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (openai / oracle VLM)")
    ap.add_argument("--resume", action="store_true", help="skip streams whose outputs already exist")
    ap.add_argument("--verify-trackeval", default=None,
                    help="path to a TrackEval checkout: recompute every sequence with its HOTA class and "
                         "report the largest difference to our evaluator")
    a = ap.parse_args()

    cfg_path = a.config or os.path.join(ROOT, "configs", "rmot_kitti.yaml" if "kitti" in a.dataset else "rmot_aerialmind.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    fps = float(cfg.get("fps", 30))
    base = dict(cfg.get("rmot") or {})
    if a.belief_params:                          # parameters fitted on train (scripts/rmot_fit_belief.py)
        with open(a.belief_params) as f:
            base = deep_update(base, (yaml.safe_load(f) or {}).get("rmot") or {})
        base.setdefault("belief", {})["enabled"] = True
    base["stride"] = int(cfg.get("stride", 1))
    if a.n_workers:
        base.setdefault("sched", {})["n_workers"] = a.n_workers
    if a.latency_scale is not None:
        base.setdefault("sched", {})["latency_scale"] = a.latency_scale
    if a.fixed_latency is not None:
        base.setdefault("sched", {})["fixed_latency_s"] = a.fixed_latency
    if a.no_zoom:
        base["zoom"] = False
    if a.no_belief:
        base.setdefault("belief", {})["enabled"] = False
    if (base.get("belief") or {}).get("enabled") and not a.belief_params:
        print("WARNING: belief model with UNFITTED fallback parameters; fit them on the train split with "
              "scripts/rmot_fit_belief.py and pass --belief-params (or --no-belief for the heuristic)", flush=True)
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m}; choose from {list(METHODS)}")
    lags = [INF if s.strip() in ("inf", "offline") else float(s) for s in a.lags.split(",")]

    kw = {"label_format": a.label_format} if a.dataset == "aerialmind" else {}
    split = load_split(a.dataset, a.root, a.split, a.seqmap, fps=fps, **kw)
    vids = list(split.videos)[: a.max_videos] if a.max_videos else list(split.videos)
    if not vids:
        raise SystemExit(f"no test videos of {a.dataset}/{a.split} under {a.root}")
    os.makedirs(a.out, exist_ok=True)
    for v in vids[:3]:
        chk = check_label_format(split.videos[v])
        if chk["n"] and chk["best"] != split.videos[v].fmt:
            print(f"WARNING {v}: labels look like {chk['best']} but {split.videos[v].fmt} is used "
                  f"(out-of-image fraction {chk}); set label_format explicitly if this is wrong")
    print(f"{a.dataset}/{a.split}: {len(vids)} videos, "
          f"{sum(min(len(split.expressions[v]), a.max_expr or 10**9) for v in vids)} expressions; "
          f"methods {methods}; lags {[lag_name(L) for L in lags]}")

    vlm_kind = a.vlm or cfg["vlm"].get("backend", "openai")
    if vlm_kind == "transformers" and a.jobs > 1:
        # every worker process would load its own copy of the model on the GPU; one process already
        # runs the verifier pool's concurrent requests as batched forward passes
        print(f"--vlm transformers: --jobs {a.jobs} -> 1 (the model is loaded once, requests are batched)", flush=True)
        a.jobs = 1
    cache_root = a.perception_cache or os.path.join(os.path.dirname(os.path.abspath(a.out)), "perception_cache", a.dataset)
    # ---- phase 1: perception once per video (GPU, sequential)
    if not a.tracks_dir:
        perceptor = None
        need = [v for v in vids if not has_cache(cache_root, v, len(split.videos[v]))]
        if need:
            pb = bar(sum(len(split.videos[v]) for v in need), "perception", "frame")
            for vi, vid in enumerate(need):
                video = split.videos[vid]
                if perceptor is None:
                    from perception import Perceptor
                    perceptor = Perceptor.from_yaml(os.path.join(ROOT, cfg["perception_config"]))
                pb.set_postfix_str(f"video {vi + 1}/{len(need)} {vid}")
                run_perception(video.frames, perceptor, cache_root, vid, fps, pbar=pb)
            pb.close()
        print(f"perception cache ready for {len(vids)} videos ({len(vids) - len(need)} were cached)", flush=True)
        del perceptor
    # ---- phase 2: every (video, expression) stream, all methods x lags
    units = []
    for vid in vids:
        exprs = split.expressions[vid][: a.max_expr] if a.max_expr else split.expressions[vid]
        units += [(vid, ei) for ei in range(len(exprs))]
    evaluated = [split.expressions[v][ei].seq_name for v, ei in units]
    if not a.resume:
        for m in methods:                        # fresh stats per run
            for L in lags:
                p = os.path.join(a.out, f"{m}_L{lag_name(L)}", "stats.jsonl")
                if os.path.isfile(p):
                    os.remove(p)
    done = _done_units(a.out, methods, lags) if a.resume else set()
    todo = [u for u in units if split.expressions[u[0]][u[1]].seq_name not in done]
    print(f"{len(units)} streams ({len(units) - len(todo)} already done), jobs={a.jobs}", flush=True)
    state = {"args": vars(a), "cfg": cfg, "base": base, "fps": fps, "methods": methods, "lags": lags,
             "cache_root": cache_root, "vlm_kind": vlm_kind}
    t_start = time.time()

    def record(rows):
        for st in rows:
            d = os.path.join(a.out, f"{st['method']}_L{st['lag']}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "stats.jsonl"), "a") as f:
                f.write(json.dumps(st) + "\n")

    pb = bar(len(todo), "streams", "stream")
    tot = {"calls": 0, "n": 0}

    def after(rows):
        record(rows)
        tot["calls"] += sum((r.get("vlm") or {}).get("calls", 0) for r in rows)
        tot["n"] += 1
        el = time.time() - t_start
        hits = f", cache hits {_W['cached'].hits}" if _W.get("cached") else ""
        pb.set_postfix_str(f"{rows[0]['seq'].split('+')[0] if rows else ''}, {el / tot['n']:.0f}s/stream, "
                           f"VLM calls {tot['calls']}{hits}")
        pb.update(1)

    if a.jobs <= 1:
        _init_worker(state, split)
        if sys.stderr.isatty():                  # a second line only in a real terminal (no blank lines in logs)
            inner = bar(len(methods) * len(lags), "  runs", "run", leave=False, position=1)
            _W["on_start"] = lambda label: inner.set_description(f"  {label}")
            _W["on_run"] = lambda: inner.update(1)
            _W["on_unit"] = lambda: inner.reset(total=len(methods) * len(lags))
        else:
            inner = _NoBar()
        for u in todo:
            after(_run_unit(u))
        inner.close()
        if _W.get("cached"):
            _W["cached"].close()
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                 initargs=(state, split)) as ex:
            futs = [ex.submit(_run_unit, u) for u in todo]
            for fu in as_completed(futs):
                after(fu.result())
    pb.close()
    print(f"{tot['n']} streams done in {(time.time() - t_start) / 60:.1f} min, {tot['calls']} VLM calls", flush=True)

    seqmap_path = os.path.join(a.out, f"seqmap_{a.split.replace(',', '_')}.txt")
    with open(seqmap_path, "w") as f:
        f.write("".join(s + "\n" for s in evaluated))
    lengths = {v: len(split.videos[v]) for v in vids}
    results = {}
    pe = bar(len(methods) * len(lags), "HOTA eval", "run")
    for m in methods:
        for L in lags:
            d = os.path.join(a.out, f"{m}_L{lag_name(L)}")
            pe.set_postfix_str(f"{m}@L{lag_name(L)}")
            r = H.evaluate_folder(d, evaluated, lengths)
            pe.update(1)
            calls = []
            with open(os.path.join(d, "stats.jsonl")) as f:
                for line in f:
                    calls.append(json.loads(line)["vlm"]["calls"])
            results[f"{m}@L{lag_name(L)}"] = {**r["combined"], "vlm_calls_mean": sum(calls) / max(len(calls), 1),
                                              "n_seqs": len(evaluated)}
            with open(os.path.join(d, "per_seq.json"), "w") as f:
                json.dump(r["per_seq"], f)
    pe.close()
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"dataset": a.dataset, "split": a.split, "vlm": vlm_kind, "results": results}, f, indent=1)
    lines = [f"# {a.dataset} / {a.split} - HOTA@L ({len(evaluated)} sequences, VLM {vlm_kind})", "",
             "| method | " + " | ".join(f"L={lag_name(L)}" for L in lags) + " |",
             "|---|" + "---|" * len(lags)]
    for m in methods:
        lines.append(f"| {m} | " + " | ".join(
            f"{results[f'{m}@L{lag_name(L)}']['HOTA']:.2f}" for L in lags) + " |")
    lines += ["", "| method@L | HOTA | DetA | AssA | LocA | DetRe | DetPr | VLM calls/stream |", "|---|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        lines.append(f"| {k} | {r['HOTA']:.2f} | {r['DetA']:.2f} | {r['AssA']:.2f} | {r['LocA']:.2f} | "
                     f"{r['DetRe']:.2f} | {r['DetPr']:.2f} | {r['vlm_calls_mean']:.1f} |")
    if vlm_kind == "oracle":
        lines += ["", "**VLM = oracle (ground truth with injected errors): plumbing check, not a result.**"]
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if a.verify_trackeval:
        diff = verify_with_trackeval(a.verify_trackeval, a.out, methods, lags, evaluated, lengths)
        print(f"TrackEval cross-check: max |HOTA difference| over all sequences = {diff:.2e}")


if __name__ == "__main__":
    main()
