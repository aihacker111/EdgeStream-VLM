"""Prompt builder: 5 blocks in a fixed order so the static prefix can be KV-cached.

    [1] system (fixed)  [2] event card (fixed per type)  [3] images  [4] memory context  [5] question

Objects are referred to by numbers drawn on the images (Set-of-Mark): [1], [2], ...
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..memory.memory import visual_tokens
from ..types import Candidate
from .cards import EventCard

SYSTEM = (
    "You verify traffic events from a fixed or aerial camera. Judge ONLY the object(s) "
    "marked with numbers in square brackets. Directions are in image coordinates. "
    "If the evidence is insufficient, answer \"uncertain\". Reply with one JSON object: "
    "{\"verdict\": \"yes|no|uncertain\", \"event_type\": \"<event id or none>\", "
    "\"severity\": \"low|medium|high\", \"description\": \"<= 25 words\"}."
)

MARK_COLORS = [(0, 0, 255), (0, 200, 255), (255, 0, 255)]


@dataclass
class VLMRequest:
    cand: Candidate
    card: EventCard
    system: str
    parts: list                    # [("text", str) | ("image", np.ndarray BGR)]
    visual_tokens: int
    max_new_tokens: int
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(p[1] for p in self.parts if p[0] == "text")

    def images(self) -> list:
        return [p[1] for p in self.parts if p[0] == "image"]


def draw_marks(frame: np.ndarray, boxes: list, scale: float, width: int) -> np.ndarray:
    """Overview image: resize first, then draw [k] marks so they stay visible on small objects."""
    hh, ww = frame.shape[:2]
    s = width / float(ww)
    img = cv2.resize(frame, (width, int(round(hh * s))), interpolation=cv2.INTER_AREA)
    k_scale = scale * s
    for k, (x, y, w, h) in enumerate(boxes):
        c = MARK_COLORS[k % len(MARK_COLORS)]
        x0, y0, x1, y1 = (int(round(v * k_scale)) for v in (x, y, x + w, y + h))
        pad = 3
        cv2.rectangle(img, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), c, 2)
        cv2.putText(img, f"[{k + 1}]", (x0, max(y0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return img


class PromptBuilder:
    def __init__(self, cards: dict[str, EventCard], overview_width: int = 448,
                 max_new_tokens: int = 64, max_marked: int = 3):
        self.cards = cards
        self.overview_width = overview_width
        self.max_new_tokens = max_new_tokens
        self.max_marked = max_marked

    def build(self, cand: Candidate, ctx: dict, frame_scaled: np.ndarray, frame_scale: float,
              boxes: list) -> VLMRequest:
        card = self.cards[cand.type_hint]
        overview = draw_marks(frame_scaled, boxes[: self.max_marked], frame_scale, self.overview_width)
        parts = [("text", card.render()),
                 ("text", "Image 1: current overview, marked objects in colour."),
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
            link = f" (possibly the same vehicle as #{trk['linked_to']} seen earlier)" if trk["linked_to"] else ""
            lines.append(f"[{k + 1}] {trk['cls_name']}: {trk['trajectory']}{link}.")
        if ctx["scene_log"]:
            lines.append("Recent notes: " + " | ".join(ctx["scene_log"]))
        lines.append(f"Why it was flagged: {cand.type_hint} rule, {cand.duration:.0f}s, "
                     f"cues {cand.evidence}.")
        parts.append(("text", "\n".join(lines)))
        subject = "[1]" if len(boxes) <= 1 else "the marked objects"
        parts.append(("text", f"Question: does {subject} match the event \"{card.id}\"? "
                              "Answer with the JSON object only."))
        return VLMRequest(cand=cand, card=card, system=SYSTEM, parts=parts, visual_tokens=vis,
                          max_new_tokens=self.max_new_tokens,
                          meta={"t": cand.t_last, "n_images": sum(1 for p in parts if p[0] == "image")})
