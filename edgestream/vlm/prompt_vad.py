"""Open-set VAD prompt: fixed taxonomy prefix + hypothesis-conditioned event cards.

    [1] system + full taxonomy (one line per category)      <- identical for every call: KV-cached
    [2] the 1-3 cards selected by the cheap-tier hypothesis   <- retrieval keyed by candidate type
    [3] overview with Set-of-Mark boxes + per-object crops
    [4] memory context (scene statistics, trajectories, why it was flagged)
    [5] question

The hypothesis only decides which cards are spelled out; the answer may be any category,
"other_anomaly" or "normal", so a wrong cheap-tier guess (e.g. "running" for a robbery)
does not cap recall.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from ..memory.memory import visual_tokens
from ..types import Candidate
from .cards import EventCard
from .prompt import VLMRequest, draw_marks

SYSTEM_VAD = (
    "You are a video-surveillance analyst for aerial (drone) and fixed cameras. Decide whether "
    "the object(s) marked with numbers in square brackets are involved in an ABNORMAL event. "
    "Most activity is normal: answer \"no\" with event_type \"normal\" unless the images show the "
    "event. Use \"uncertain\" only if the images are too small or blurred to judge.\n"
    "Categories:\n{taxonomy}\n"
    "Reply with one JSON object: {{\"verdict\": \"yes|no|uncertain\", \"event_type\": \"<category id "
    "or normal>\", \"severity\": \"low|medium|high\", \"description\": \"<= 25 words\"}}."
)


def load_hypotheses(cards_dir: str | Path) -> dict[str, list[str]]:
    p = Path(cards_dir) / "_hypotheses.yaml"
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def canonical_category(value: str, cards: dict[str, EventCard]) -> str:
    """Map a free-form answer ("Fighting", "trespassing on lawn") to a card id or 'normal'."""
    v = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if v in cards:
        return v
    if v in ("normal", "none", "no", "no_anomaly", ""):
        return "normal"
    for cid, card in cards.items():
        if v == re.sub(r"[^a-z0-9]+", "_", card.name.lower()).strip("_"):
            return cid
    for cid in cards:                      # prefix / containment fallback
        if v.startswith(cid) or cid.startswith(v):
            return cid
    return "other_anomaly" if "other_anomaly" in cards else v


class VADPromptBuilder:
    def __init__(self, cards: dict[str, EventCard], hypotheses: dict[str, list[str]],
                 overview_width: int = 448, max_new_tokens: int = 64, max_marked: int = 3,
                 max_cards: int = 3):
        self.cards = cards
        self.hyp = hypotheses
        self.overview_width = overview_width
        self.max_new_tokens = max_new_tokens
        self.max_marked = max_marked
        self.max_cards = max_cards
        tax = "\n".join(f"- {cid}: {c.short or c.name}" for cid, c in cards.items()) + \
            "\n- normal: ordinary activity for this place"
        self.system = SYSTEM_VAD.format(taxonomy=tax)
        self.card_ids = list(cards) + ["normal"]

    def hypotheses_for(self, cand: Candidate) -> list[str]:
        grp = (cand.evidence or {}).get("class_group")
        keyed = self.hyp.get(f"{cand.type_hint}@{grp}") if grp else None
        ids = [c for c in (keyed or self.hyp.get(cand.type_hint, [])) if c in self.cards]
        if not ids:
            ids = ["other_anomaly"] if "other_anomaly" in self.cards else [next(iter(self.cards))]
        return ids[: self.max_cards]

    def build(self, cand: Candidate, ctx: dict, frame, boxes: list) -> VLMRequest:
        hyps = self.hypotheses_for(cand)
        card = self.cards[hyps[0]]
        cards_txt = "Most likely categories here (judge them carefully):\n\n" + \
            "\n\n".join(self.cards[h].render() for h in hyps)
        overview = draw_marks(frame, boxes[: self.max_marked], 1.0, self.overview_width)
        parts = [("text", cards_txt), ("text", "Image 1: current overview, marked objects in colour."),
                 ("image", overview)]
        vis = visual_tokens(overview.shape[1], overview.shape[0])
        for k, trk in enumerate(ctx["tracks"][: self.max_marked]):
            if not trk["crops"]:
                continue
            times = ", ".join(f"t={tc:.1f}s" for _, _, tc in trk["crops"])
            parts.append(("text", f"Crops of [{k + 1}] ({trk['cls_name']}) at {times}, oldest first:"))
            for img, (w, h), _ in trk["crops"]:
                parts.append(("image", img))
                vis += visual_tokens(w, h)
        lines = [f"Scene: {ctx['stats_text']}"]
        for k, trk in enumerate(ctx["tracks"][: self.max_marked]):
            lines.append(f"[{k + 1}] {trk['cls_name']}: {trk['trajectory']}.")
        if not ctx["tracks"] and boxes:
            lines.append("[1] is an image region with motion or change that the detector could not "
                         "classify (it may be an animal, an object or a vehicle type it does not know).")
        if ctx["scene_log"]:
            lines.append("Recent notes: " + " | ".join(ctx["scene_log"]))
        lines.append(f"Why it was flagged: {cand.type_hint.replace('_', ' ')} for {cand.duration:.0f}s, "
                     f"cues {cand.evidence}.")
        parts.append(("text", "\n".join(lines)))
        subject = "[1]" if len(boxes) <= 1 else "the marked objects"
        parts.append(("text", f"Question: is {subject} involved in an abnormal event? If yes, which "
                              "category? Answer with the JSON object only."))
        return VLMRequest(cand=cand, card=card, system=self.system, parts=parts, visual_tokens=vis,
                          max_new_tokens=self.max_new_tokens,
                          meta={"t": cand.t_last, "n_images": sum(1 for p in parts if p[0] == "image"),
                                "card_ids": self.card_ids, "default_event": hyps[0], "hypotheses": hyps})
