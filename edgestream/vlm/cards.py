"""Event cards: one small YAML per event type, loaded only when that type is verified."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EventCard:
    id: str
    name: str
    definition: str
    positive_cues: list = field(default_factory=list)
    negative_cues: list = field(default_factory=list)
    severity: str = "medium"
    version: int = 1
    short: str = ""                     # one-line definition for the taxonomy list

    def render(self) -> str:
        pos = "\n".join(f"  - {c}" for c in self.positive_cues)
        neg = "\n".join(f"  - {c}" for c in self.negative_cues)
        return (f"EVENT: {self.name} (id: {self.id})\n"
                f"Definition: {self.definition.strip()}\n"
                f"Signs that it IS this event:\n{pos}\n"
                f"Signs that it is NOT (answer \"no\"):\n{neg}")


def load_cards(directory: str | Path) -> dict[str, EventCard]:
    cards = {}
    for p in sorted(Path(directory).glob("*.yaml")):
        if p.name.startswith("_"):          # _hypotheses.yaml etc. are not cards
            continue
        with open(p) as f:
            raw = yaml.safe_load(f)
        card = EventCard(**{k: raw[k] for k in EventCard.__dataclass_fields__ if k in raw})
        cards[card.id] = card
    if not cards:
        raise FileNotFoundError(f"No event cards in {directory}")
    return cards


def output_schema(card_ids: list[str]) -> dict:
    """JSON schema used for constrained decoding (vLLM/SGLang/XGrammar/Outlines)."""
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["yes", "no", "uncertain"]},
            "event_type": {"type": "string", "enum": list(card_ids) + ["none"]},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            "description": {"type": "string", "maxLength": 160},
        },
        "required": ["verdict", "event_type", "severity", "description"],
        "additionalProperties": False,
    }
