#!/usr/bin/env python
"""Fine-tuning data for the RMOT verifier, generated from the AerialMind TRAIN split.

Every sample is exactly the request the method sends at inference (prompt v2, one candidate,
built by the same function `VLMVerifier.build_v2`), answered "yes" or "no" from the ground truth:

    candidates = pieces of the TRACKER's tracks (same distribution as at test time), restricted to
                 the class family of the expression; label = referred fraction of its frames
                 (box IoU >= 0.5 with a referred GT box) >= 0.6 -> yes, <= 0.1 -> no, else skipped
    sampling   = per (video, expression) up to --pos positives and --neg negatives; half of the
                 negatives are "hard" (referred by ANOTHER expression of the video: right class,
                 wrong attribute) when available
    anchor     = a random frame inside the piece; the graph is replayed up to that frame, so the
                 motion facts are what the online method would have seen then
    facts      = on / off / mix (each sample flips a coin), so one adapter serves both settings

Output (``--out``):
    images/<video>/frame_<rank>.jpg      resized frames, shared by all samples at that frame
    images/<video>/strip_<n>.jpg         candidate close-up strips
    train.jsonl, val.jsonl               {"messages": [...], "label": 0/1, "video", "expr", ...}
    stats.json
Videos are split into train / val by video (--val-videos), never by sample.

    python scripts/rmot_make_sft.py --root data/AerialMind --out data/rmot_sft --max-expr 100
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cv2  # noqa: E402
import yaml  # noqa: E402

import rmot_benchmark as RB  # noqa: E402
from edgestream.rmot.datasets import aerialmind  # noqa: E402
from edgestream.rmot.expression import parse  # noqa: E402
from edgestream.rmot.perception import has_cache, load_mot_tracks, load_perception, run_perception  # noqa: E402
from edgestream.rmot.pipeline import GTOracle  # noqa: E402
from edgestream.rmot.tracklets import GraphConfig, TrackletGraph  # noqa: E402
from edgestream.rmot.verifier import FrameSource, VLMVerifier  # noqa: E402


def to_messages(system: str, parts: list, answer: str | None, img_paths: list) -> list:
    """Chat messages in the transformers format; images replaced by file paths (in order)."""
    content, k = [], 0
    for kind, val in parts:
        if kind == "image":
            content.append({"type": "image", "image": img_paths[k]})
            k += 1
        else:
            content.append({"type": "text", "text": val})
    msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def replay(perc, aff, fps, image_wh, upto=None):
    g = TrackletGraph(GraphConfig(fps=fps))
    for r, fo in enumerate(perc):
        if upto is not None and r > upto:
            break
        if fo is not None:
            g.update(r, r / fps, fo.tracks, image_wh, aff[r] if aff else None)
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="AerialMind root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "rmot_aerialmind.yaml"))
    ap.add_argument("--videos", default="train", help="'train' (all non-test videos) or a comma list")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--max-expr", type=int, default=100, help="expressions per video (random subset)")
    ap.add_argument("--pos", type=int, default=4, help="positives per expression")
    ap.add_argument("--neg", type=int, default=4, help="negatives per expression")
    ap.add_argument("--val-videos", type=int, default=4, help="videos held out for validation")
    ap.add_argument("--facts", default="mix", choices=["on", "off", "mix"])
    ap.add_argument("--perception-cache", default=None, help="default: runs/perception_cache/aerialmind")
    ap.add_argument("--tracks-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cfg = yaml.safe_load(open(a.config))
    fps = float(cfg.get("fps", 30))
    v = cfg.get("vlm") or {}
    builder = VLMVerifier(None, prompt="v2", frame_width=v.get("frame_width", 1024), crop_h=v.get("crop_h", 160))
    split = aerialmind(a.root, a.videos)
    vids = list(split.videos)
    rng.shuffle(vids)
    vids = sorted(vids[: a.max_videos]) if a.max_videos else sorted(vids)
    if not vids:
        sys.exit(f"no video found under {a.root} for --videos {a.videos}")
    val_vids = set(sorted(vids, key=lambda x: rng.random())[: a.val_videos]) if len(vids) > a.val_videos else set()
    cache_root = a.perception_cache or os.path.join(ROOT, "runs", "perception_cache", "aerialmind")
    os.makedirs(a.out, exist_ok=True)

    if not a.tracks_dir:                              # tracker output for every video (cached)
        need = [x for x in vids if not has_cache(cache_root, x, len(split.videos[x]))]
        if need:
            from perception import Perceptor
            perceptor = Perceptor.from_yaml(os.path.join(ROOT, cfg["perception_config"]))
            pp = RB.bar(sum(len(split.videos[x]) for x in need), "perception", "frame")
            for i, x in enumerate(need):
                pp.set_postfix_str(f"video {i + 1}/{len(need)} {x}")
                run_perception(split.videos[x].frames, perceptor, cache_root, x, fps, pbar=pp)
            pp.close()
            del perceptor

    files = {"train": open(os.path.join(a.out, "train.jsonl"), "w"), "val": open(os.path.join(a.out, "val.jsonl"), "w")}
    stats = defaultdict(int)
    pb = RB.bar(len(vids), "videos", "video")
    n_strip = 0
    for vid in vids:
        video = split.videos[vid]
        W, H = video.image_wh
        if a.tracks_dir:
            perc, aff = load_mot_tracks(os.path.join(a.tracks_dir, f"{vid}.txt"), len(video), (H, W), fps)
        else:
            perc, aff = load_perception(cache_root, vid)
        g_full = replay(perc, aff, fps, (W, H))
        exprs = list(split.expressions[vid])
        rng.shuffle(exprs)
        exprs = exprs[: a.max_expr]
        # truth of every piece for every expression (hard negatives = referred by another expression)
        oracles = {e.expression_id: GTOracle(video, e) for e in exprs}
        truth = {e.expression_id: {} for e in exprs}
        for e in exprs:
            q = parse(e.text)
            for s in g_full.segs.values():
                if s.n >= 3 and q.class_ok(s.cls_name):
                    truth[e.expression_id][s.sid] = oracles[e.expression_id].refer(s)
        picks = []                                    # (anchor_rank, sid, expr, label)
        for e in exprs:
            tr = truth[e.expression_id]
            pos = [sid for sid, f in tr.items() if f >= 0.6]
            neg = [sid for sid, f in tr.items() if f <= 0.1]
            other = {sid for e2 in exprs if e2 is not e for sid, f in truth[e2.expression_id].items() if f >= 0.6}
            hard = [sid for sid in neg if sid in other]
            easy = [sid for sid in neg if sid not in other]
            rng.shuffle(pos)
            rng.shuffle(hard)
            rng.shuffle(easy)
            nh = min(len(hard), a.neg // 2)
            chosen_neg = hard[:nh] + easy[: a.neg - nh]
            if len(chosen_neg) < a.neg:
                chosen_neg += hard[nh: nh + a.neg - len(chosen_neg)]
            for sid, lab in [(x, 1) for x in pos[: a.pos]] + [(x, 0) for x in chosen_neg]:
                s = g_full.segs[sid]
                anchor = rng.choice(s.obs).rank
                picks.append((anchor, sid, e.text, lab, sid in hard))
            stats["skipped_ambiguous"] += sum(1 for f in tr.values() if 0.1 < f < 0.6)
        # second pass: replay the graph and build each sample at its anchor frame (online state)
        picks.sort()
        frames = FrameSource(video.frames, cache=64)
        img_dir = os.path.join(a.out, "images", vid)
        os.makedirs(img_dir, exist_ok=True)
        part = "val" if vid in val_vids else "train"
        g = TrackletGraph(GraphConfig(fps=fps))
        k = 0
        for r, fo in enumerate(perc):
            if fo is not None:
                g.update(r, r / fps, fo.tracks, (W, H), aff[r] if aff else None)
            while k < len(picks) and picks[k][0] == r:
                anchor, sid, expr, lab, is_hard = picks[k]
                k += 1
                if sid not in g.segs:
                    continue
                builder.facts = (a.facts == "on") or (a.facts == "mix" and rng.random() < 0.5)
                system, parts = builder.build_v2(g, g.segs[sid], anchor, expr, frames)
                paths, first = [], True
                for kind, val in parts:
                    if kind != "image":
                        continue
                    if first:                         # the frame: one file per (video, rank)
                        p = os.path.join(img_dir, f"frame_{anchor:06d}.jpg")
                        if not os.path.isfile(p):
                            cv2.imwrite(p, val, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        first = False
                    else:
                        p = os.path.join(img_dir, f"strip_{n_strip:07d}.jpg")
                        n_strip += 1
                        cv2.imwrite(p, val, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    paths.append(os.path.relpath(p, a.out))
                rec = {"messages": to_messages(system, parts, "yes" if lab else "no", paths), "label": lab,
                       "video": vid, "expr": expr, "anchor": anchor, "cls": g.segs[sid].cls_name,
                       "facts": builder.facts, "hard_negative": bool(is_hard and not lab)}
                files[part].write(json.dumps(rec) + "\n")
                stats[f"{part}_{'pos' if lab else 'neg'}"] += 1
                stats["hard_negatives"] += int(is_hard and not lab)
        stats["videos_" + part] += 1
        pb.set_postfix_str(f"{vid} | " + ", ".join(f"{x} {stats[x]}" for x in ("train_pos", "train_neg", "val_pos", "val_neg")))
        pb.update(1)
    pb.close()
    for f in files.values():
        f.close()
    stats = dict(stats)
    stats["val_videos"] = sorted(val_vids)
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
