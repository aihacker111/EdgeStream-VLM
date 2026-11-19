"""Set-level comparative VLM verifier, pairwise link verifier and the single-candidate baseline prompt.

Prompt layout (fixed order so the static part is shared by every expression and can be
prefix-cached by vLLM / SGLang):

    system | overview frame with [k] marks | close-up gallery | measured facts | EXPRESSION (last)

Group composition depends only on perception (anchor frame, class family, partition round),
never on the expression, so the same prefix is reused across all queries of a video.

The answer is ``{"1": "yes|no", "2": ...}``; P(yes) per mark is read from the token logprobs at
each value (``field_yes_probability``), so one call scores the whole group.
"""
from __future__ import annotations

import math
import random
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..memory.memory import visual_tokens
from ..vlm.backends import field_yes_probability
from .tracklets import Segment, TrackletGraph

SYSTEM_SET = (
    "You are a careful judge for referring multi-object tracking in drone (UAV) or vehicle video. "
    "An expression describes a set of target objects. Candidate objects are marked [k] in an overview "
    "frame; a close-up gallery shows each candidate at three moments (left = earliest); measured facts "
    "give each candidate's class, image position and ground motion with the camera motion removed. "
    "Decide for EVERY mark whether that object matches the expression during its shown time span. "
    "Any number of marks may match, including none. Use the other candidates for comparison when the "
    "expression is relative (largest, leftmost, same direction as the others). 'us', 'we', 'our' and "
    "'the observer' mean the drone/camera. Trust the measured motion facts over your impression of a "
    "single frame. Reply only with a JSON object mapping every mark number to \"yes\" or \"no\"."
)
SYSTEM_SINGLE = (
    "You judge whether ONE marked object in drone (UAV) or vehicle video matches a referring "
    "expression. The object is marked [1] in the frame and shown in close-ups (left = earliest). "
    "'us', 'we', 'our' mean the drone/camera. Reply only with {\"1\": \"yes\"} or {\"1\": \"no\"}."
)
SYSTEM_LINK = (
    "You decide whether two tracked boxes are the SAME physical object. Row A shows the object at "
    "the end of its first track, row B the start of a second track (left = earliest). Consider "
    "colour, shape, size and the measured motion. Reply only with {\"same\": \"yes\"} or {\"same\": \"no\"}."
)

PALETTE = [(0, 0, 255), (0, 200, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (255, 255, 0),
           (128, 0, 255), (0, 128, 255), (255, 0, 128), (128, 255, 0), (0, 255, 255), (200, 200, 200)]


def yes_no_schema(keys: list) -> dict:
    keys = [str(k) for k in keys]
    return {"type": "object", "properties": {k: {"type": "string", "enum": ["yes", "no"]} for k in keys},
            "required": keys, "additionalProperties": False}


# ---------------------------------------------------------------------- frames & drawing
class FrameSource:
    """Random access to frames by rank: image files on disk or an in-memory ring buffer."""

    def __init__(self, paths: list | None = None, cache: int = 64):
        self.paths = paths or []
        self.buf: OrderedDict = OrderedDict()
        self.cache = cache

    def put(self, rank: int, frame: np.ndarray) -> None:
        self.buf[rank] = frame
        self.buf.move_to_end(rank)
        while len(self.buf) > self.cache:
            self.buf.popitem(last=False)

    def get(self, rank: int) -> np.ndarray | None:
        if rank in self.buf:
            self.buf.move_to_end(rank)
            return self.buf[rank]
        if 0 <= rank < len(self.paths):
            img = cv2.imread(self.paths[rank])
            if img is not None:
                self.put(rank, img)
            return img
        return None


def crop(frame: np.ndarray, box, ctx: float = 0.6, min_side: int = 40) -> np.ndarray:
    x, y, w, h = [float(v) for v in box]
    side_w, side_h = max(w * (1 + 2 * ctx), min_side), max(h * (1 + 2 * ctx), min_side)
    cx, cy = x + w / 2, y + h / 2
    H, W = frame.shape[:2]
    x0, y0 = int(max(cx - side_w / 2, 0)), int(max(cy - side_h / 2, 0))
    x1, y1 = int(min(cx + side_w / 2, W)), int(min(cy + side_h / 2, H))
    c = frame[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)].copy()
    # thin outline of the object inside its context crop
    bx0, by0 = int(x - x0), int(y - y0)
    cv2.rectangle(c, (bx0, by0), (int(bx0 + w), int(by0 + h)), (0, 255, 255), 1)
    return c


def overview(frame: np.ndarray, marks: list, width: int = 1280) -> np.ndarray:
    """``marks`` = [(label, tlwh)]. Resize first, then draw so labels stay legible."""
    H, W = frame.shape[:2]
    s = min(1.0, width / float(W))
    img = cv2.resize(frame, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else frame.copy()
    for n, (lab, (x, y, w, h)) in enumerate(marks):
        c = PALETTE[n % len(PALETTE)]
        p0 = (int(x * s), int(y * s))
        p1 = (int((x + w) * s), int((y + h) * s))
        cv2.rectangle(img, p0, p1, c, 2)
        txt = f"[{lab}]"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = p0[1] - 3 if p0[1] - th - 4 > 0 else p1[1] + th + 3
        cv2.rectangle(img, (p0[0], ty - th - 2), (p0[0] + tw + 2, ty + 2), c, -1)
        cv2.putText(img, txt, (p0[0] + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def gallery(rows: list, tile_h: int = 112, max_w: int = 168, label_w: int = 44) -> np.ndarray:
    """``rows`` = [(label, [crops])] -> one image, one row per candidate."""
    rendered = []
    for lab, crops in rows:
        tiles = [np.full((tile_h, label_w, 3), 255, np.uint8)]
        cv2.putText(tiles[0], f"[{lab}]", (2, tile_h // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
                    cv2.LINE_AA)
        for c in crops:
            if c is None or c.size == 0:
                continue
            h, w = c.shape[:2]
            s = tile_h / float(h)
            nw = int(min(max(w * s, 8), max_w))
            t = cv2.resize(c, (nw, tile_h), interpolation=cv2.INTER_LINEAR)
            tiles.append(t)
            tiles.append(np.full((tile_h, 4, 3), 255, np.uint8))
        rendered.append(np.concatenate(tiles, 1))
    if not rendered:
        return np.full((tile_h, label_w, 3), 255, np.uint8)
    W = max(r.shape[1] for r in rendered)
    rows_img = []
    for r in rendered:
        pad = np.full((tile_h, W - r.shape[1], 3), 255, np.uint8)
        rows_img.append(np.concatenate([r, pad], 1))
        rows_img.append(np.full((4, W, 3), 200, np.uint8))
    return np.concatenate(rows_img, 0)


# ---------------------------------------------------------------------- facts
_DIRS = ["right", "top-right", "top", "top-left", "left", "bottom-left", "bottom", "bottom-right"]


def direction_word(deg: float) -> str:
    return _DIRS[int(((deg + 22.5) % 360) // 45)]


def region_word(pos: tuple) -> str:
    x, y = pos
    v = "top" if y < 1 / 3 else ("bottom" if y > 2 / 3 else "middle")
    h = "left" if x < 1 / 3 else ("right" if x > 2 / 3 else "center")
    return "center" if (v, h) == ("middle", "center") else f"{v}-{h}"


def drone_fact(g: TrackletGraph) -> str:
    v = g.ego_vel
    sp = float(np.linalg.norm(v))
    if sp < g.cfg.ego_min_px_s:
        return "Drone/camera: hovering (no significant camera motion)."
    d = direction_word(math.degrees(math.atan2(-v[1], v[0])))
    return f"Drone/camera is moving toward the image {d} ({sp:.0f} px/s apparent)."


def candidate_fact(label, s: Segment, g: TrackletGraph) -> str:
    k = g.kinematics(s)
    last = s.obs[-1]
    parts = [f"[{label}] {s.cls_name}", f"{region_word(k['pos'])} of image",
             f"{int(last.box[2])}x{int(last.box[3])}px"]
    if not k["moving"]:
        parts.append("stationary on the ground")
    else:
        mv = f"moving toward image {direction_word(k['heading'])} at {k['speed_bl_s']:.1f} body-lengths/s"
        if k["turn_deg"] is not None and abs(k["turn_deg"]) > 20:
            mv += f", turning {'left' if k['turn_deg'] > 0 else 'right'} ({abs(k['turn_deg']):.0f} deg)"
        parts.append(mv)
        c = k["rel_ego_cos"]
        if c is not None:
            parts.append("same direction as the drone" if c > 0.5 else (
                "opposite to the drone's direction" if c < -0.5 else "crossing the drone's direction"))
    return ", ".join(parts) + "."


# ---------------------------------------------------------------------- answers
@dataclass
class Answer:
    kind: str                       # refer | link
    probs: dict                     # sid -> p (refer) or {(i, j): p} (link)
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    visual_tokens: int = 0
    n_cands: int = 0
    text: str = ""
    meta: dict = field(default_factory=dict)


def history_crops(g: TrackletGraph, s: Segment, frames: FrameSource, n: int = 3, span_s: float = 1.5,
                  at: str = "around") -> list:
    """Close-ups of the track of ``s`` at ``n`` moments: around the segment (refer), at its end or
    at its start (link)."""
    track = [o for sid in g.track_of[s.gid] for o in g.segs[sid].obs]
    if at == "end":
        lo, hi = s.t1 - span_s, s.t1
    elif at == "start":
        lo, hi = s.t0, s.t0 + span_s
    else:
        mid = s.mid().t
        lo, hi = min(s.t0, mid - span_s / 2), max(s.t1, mid + span_s / 2)
    pool = [o for o in track if lo <= o.t <= hi and o.t <= g.t] or s.obs
    picks = [pool[int(round(i * (len(pool) - 1) / max(n - 1, 1)))] for i in range(n)]
    out = []
    for o in picks:
        f = frames.get(o.rank)
        if f is not None:
            out.append(crop(f, o.box))
    return out


class VLMVerifier:
    """Wraps any backend with a generic ``chat()`` (OpenAI-compatible server or transformers)."""

    def __init__(self, backend, overview_width: int = 1280, facts: bool = True, max_new_tokens: int = 160):
        self.backend = backend
        self.overview_width = overview_width
        self.facts = facts
        self.max_new_tokens = max_new_tokens

    def _cost(self, images: list) -> int:
        return sum(visual_tokens(im.shape[1], im.shape[0]) for im in images)

    def refer(self, g: TrackletGraph, group: list, anchor_rank: int, expr: str, frames: FrameSource,
              single: bool = False) -> Answer:
        """``group`` = [Segment]; marks are 1..n in the order given."""
        f = frames.get(anchor_rank)
        marks, rows, facts = [], [], []
        for k, s in enumerate(group, 1):
            o = min(s.obs, key=lambda x: abs(x.rank - anchor_rank))
            marks.append((k, o.box))
            rows.append((k, history_crops(g, s, frames)))
            facts.append(candidate_fact(k, s, g))
        ov = overview(f, marks, self.overview_width) if f is not None else None
        gal = gallery(rows)
        keys = [str(k) for k in range(1, len(group) + 1)]
        parts = []
        if ov is not None:
            parts += [("text", f"Overview frame (t = {anchor_rank / g.cfg.fps:.1f} s):"), ("image", ov)]
        parts += [("text", "Close-ups, one row per mark:"), ("image", gal)]
        if self.facts:
            parts.append(("text", "Measured facts (camera motion removed):\n" + drone_fact(g) + "\n" + "\n".join(facts)))
        example = "{" + ", ".join(f'"{k}": "yes|no"' for k in keys) + "}"
        parts.append(("text", f'Expression: "{expr}"\nAnswer for every mark: {example}'))
        res = self.backend.chat(SYSTEM_SINGLE if single else SYSTEM_SET, parts,
                                max_new_tokens=min(self.max_new_tokens, 8 + 8 * len(keys)),
                                schema=yes_no_schema(keys))
        p = field_yes_probability(res.tokens, keys)
        probs = {s.sid: p[str(k)] for k, s in enumerate(group, 1) if str(k) in p}
        imgs = [im for im in (ov, gal) if im is not None]
        return Answer("refer", probs, res.latency_s, res.prompt_tokens, res.completion_tokens,
                      self._cost(imgs), len(group), res.text)

    def link(self, g: TrackletGraph, a: Segment, b: Segment, frames: FrameSource) -> Answer:
        gal = gallery([("A", history_crops(g, a, frames, at="end")), ("B", history_crops(g, b, frames, at="start"))])
        gap = b.t0 - a.t1
        fa, fb = candidate_fact("A", a, g), candidate_fact("B", b, g)
        parts = [("text", "Row A: first track (end). Row B: second track (start)."), ("image", gal),
                 ("text", f"{fa}\n{fb}\nGap between the tracks: {gap:.1f} s.\n"
                          'Are A and B the same object? {"same": "yes|no"}')]
        res = self.backend.chat(SYSTEM_LINK, parts, max_new_tokens=16, schema=yes_no_schema(["same"]))
        p = field_yes_probability(res.tokens, ["same"])
        probs = {(a.sid, b.sid): p["same"]} if "same" in p else {}
        return Answer("link", probs, res.latency_s, res.prompt_tokens, res.completion_tokens,
                      self._cost([gal]), 2, res.text)


class OracleVerifier:
    """NOT a VLM. Answers from ground truth with controlled errors, for plumbing tests and for
    the "perfect verifier" ceiling. ``truth_refer(seg) -> fraction of frames referred`` and
    ``truth_same(a, b) -> bool | None`` are supplied by the caller (see ``pipeline.GTOracle``).

    Accuracy is ``acc`` per candidate; set-level calls get ``set_bonus`` extra accuracy, which
    models comparative judgement (0 by default, i.e. no advantage is assumed).
    """

    def __init__(self, truth_refer, truth_same, acc: float = 0.9, conf: float = 0.85, set_bonus: float = 0.0,
                 seed: int = 0, latency_base: float = 0.35, latency_per_cand: float = 0.04):
        self.truth_refer, self.truth_same = truth_refer, truth_same
        self.acc, self.conf, self.set_bonus = acc, conf, set_bonus
        self.rng = random.Random(seed)
        self.lb, self.lc = latency_base, latency_per_cand

    def _p(self, truth: bool, acc: float) -> float:
        correct = self.rng.random() < acc
        c = min(max(self.conf + self.rng.gauss(0, 0.07), 0.55), 0.99)
        yes = truth if correct else not truth
        return c if yes else 1 - c

    def refer(self, g, group, anchor_rank, expr, frames, single: bool = False) -> Answer:
        acc = min(self.acc + (0 if single or len(group) == 1 else self.set_bonus), 0.999)
        probs = {s.sid: self._p(self.truth_refer(s) >= 0.5, acc) for s in group}
        return Answer("refer", probs, self.lb + self.lc * len(group), 0, 4 * len(group),
                      300 + 60 * len(group), len(group))

    def link(self, g, a, b, frames) -> Answer:
        t = self.truth_same(a, b)
        p = 0.5 if t is None else self._p(bool(t), self.acc)
        return Answer("link", {(a.sid, b.sid): p}, self.lb + 2 * self.lc, 0, 4, 200, 2)


class CachedVerifier:
    """Exact replay cache: a request is identified by its full content (expression, anchor frame,
    each member's track id and observed extent, prompt kind, facts on/off), so the same prompt
    is answered once across lag sweeps and methods. Deterministic decoding makes this exact.
    The stored latency is replayed, so the logical clock is unchanged."""

    def __init__(self, inner, path: str | None = None):
        import json
        import os
        self.inner, self.path = inner, path
        self.mem: dict = {}
        self.hits = self.misses = 0
        if path and os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.mem[r["k"]] = r
        self._f = open(path, "a") if path else None

    @property
    def facts(self):
        return getattr(self.inner, "facts", None)

    @facts.setter
    def facts(self, v):
        if hasattr(self.inner, "facts"):
            self.inner.facts = v

    @staticmethod
    def _seg_key(s: Segment) -> str:
        return f"{s.gid}:{s.obs[0].rank}-{s.obs[-1].rank}"

    def _store(self, k: str, ans: Answer, probs: dict) -> None:
        import json
        r = {"k": k, "p": probs, "lat": ans.latency_s, "pt": ans.prompt_tokens, "ct": ans.completion_tokens,
             "vt": ans.visual_tokens, "n": ans.n_cands}
        self.mem[k] = r
        if self._f:
            self._f.write(json.dumps(r) + "\n")
            self._f.flush()

    def refer(self, g, group, anchor_rank, expr, frames, single: bool = False) -> Answer:
        keys = [self._seg_key(s) for s in group]
        k = f"R|{expr}|{anchor_rank}|{int(single)}|{int(bool(self.facts))}|{','.join(keys)}"
        r = self.mem.get(k)
        if r is not None:
            self.hits += 1
            probs = {s.sid: r["p"][kk] for s, kk in zip(group, keys) if kk in r["p"]}
            return Answer("refer", probs, r["lat"], r["pt"], r["ct"], r["vt"], r["n"], meta={"cached": True})
        self.misses += 1
        ans = self.inner.refer(g, group, anchor_rank, expr, frames, single=single)
        self._store(k, ans, {kk: ans.probs[s.sid] for s, kk in zip(group, keys) if s.sid in ans.probs})
        return ans

    def link(self, g, a, b, frames) -> Answer:
        k = f"L|{self._seg_key(a)}|{self._seg_key(b)}"
        r = self.mem.get(k)
        if r is not None:
            self.hits += 1
            return Answer("link", {(a.sid, b.sid): r["p"]["same"]} if "same" in r["p"] else {}, r["lat"],
                          r["pt"], r["ct"], r["vt"], r["n"], meta={"cached": True})
        self.misses += 1
        ans = self.inner.link(g, a, b, frames)
        p = ans.probs.get((a.sid, b.sid))
        self._store(k, ans, {} if p is None else {"same": p})
        return ans

    def close(self):
        if self._f:
            self._f.close()
