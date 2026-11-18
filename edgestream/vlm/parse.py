"""Robust parsing of the VLM's JSON answer (needed when constrained decoding is unavailable)."""
from __future__ import annotations

import json
import re

_VERDICTS = {"yes", "no", "uncertain"}
_SEVERITIES = {"low", "medium", "high"}


def extract_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    start = None
    return None


def normalize(obj: dict | None, card_id: str, default_severity: str) -> tuple[dict, bool]:
    if not obj:
        return {"verdict": "uncertain", "event_type": card_id, "severity": default_severity,
                "description": ""}, False
    verdict = str(obj.get("verdict", "uncertain")).strip().lower()
    ok = verdict in _VERDICTS
    sev = str(obj.get("severity", default_severity)).strip().lower()
    return {
        "verdict": verdict if ok else "uncertain",
        "event_type": str(obj.get("event_type", card_id)),
        "severity": sev if sev in _SEVERITIES else default_severity,
        "description": str(obj.get("description", ""))[:200],
    }, ok
