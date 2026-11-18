"""Turn verdicts into events: merge, cool down, save evidence, route by severity."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from ..harness.logger import JsonlLogger
from ..harness.stream import FrameRingBuffer
from ..types import Candidate, Event, Verdict


@dataclass
class EventConfig:
    merge_gap_s: float = 10.0          # same type + overlapping tracks within this gap -> same event
    cooldown_s: float = 30.0           # per (type, cell) for non-high severity
    min_confidence: float = 0.0        # drop "yes" verdicts below this (when confidence is known)
    clip_pre_s: float = 3.0
    save_clips: bool = True


class EventAggregator:
    def __init__(self, cfg: EventConfig):
        self.cfg = cfg
        self.events: list[Event] = []
        self._next = 1
        self._cooldown: dict[tuple, float] = {}

    def add(self, t: float, cand: Candidate, v: Verdict, event_type: str | None = None) -> tuple[Event | None, bool]:
        """Returns (event, is_new). ``event`` is None when the verdict produces no alert."""
        if v.verdict != "yes":
            return None, False
        if v.confidence is not None and v.confidence < self.cfg.min_confidence:
            return None, False
        typ = event_type or cand.type_hint
        for ev in reversed(self.events):
            if ev.event_type == typ and t - ev.t_end <= self.cfg.merge_gap_s \
                    and set(ev.track_ids) & set(cand.track_ids):
                ev.t_end = max(ev.t_end, cand.t_last)
                ev.track_ids = sorted(set(ev.track_ids) | set(cand.track_ids))
                ev.cand_ids.append(cand.cand_id)
                return ev, False
        key = (typ, tuple(cand.cell))
        if v.severity != "high" and t - self._cooldown.get(key, -1e9) < self.cfg.cooldown_s:
            return None, False
        self._cooldown[key] = t
        ev = Event(event_id=self._next, event_type=typ, severity=v.severity, t_start=cand.t_start,
                   t_end=cand.t_last, t_alert=t, track_ids=list(cand.track_ids),
                   description=v.description, confidence=v.confidence, cand_ids=[cand.cand_id])
        self._next += 1
        self.events.append(ev)
        return ev, True


def save_evidence(ev: Event, ring: FrameRingBuffer, out_dir: Path, fps: float, pre_s: float,
                  boxes_by_t: dict | None = None) -> dict:
    """Write a short clip (t_start - pre_s .. t_alert) and a keyframe from the ring buffer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = ring.window(ev.t_start - pre_s, ev.t_alert)
    if not frames:
        return {}
    clip_path = out_dir / f"event_{ev.event_id:04d}.mp4"
    h, w = frames[0][2].shape[:2]
    vw = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    key_img = None
    for t, scale, img in frames:
        for gid, (x, y, bw, bh) in (boxes_by_t or {}).get(round(t, 3), {}).items():
            if gid in ev.track_ids:
                p0 = (int(x * scale), int(y * scale))
                p1 = (int((x + bw) * scale), int((y + bh) * scale))
                cv2.rectangle(img, p0, p1, (0, 0, 255), 2)
        cv2.putText(img, f"{ev.event_type} #{ev.event_id} t={t:.1f}s", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        vw.write(img)
        key_img = img
    vw.release()
    key_path = out_dir / f"event_{ev.event_id:04d}.jpg"
    cv2.imwrite(str(key_path), key_img)
    return {"clip": clip_path.name, "keyframe": key_path.name, "n_frames": len(frames)}


class EventRouter:
    """Routes each event to the people who act on it (one JSONL channel each)."""

    def __init__(self, log: JsonlLogger):
        self.log = log

    def route(self, ev: Event) -> list[str]:
        channels = ["analytics"]
        if ev.severity == "high":
            channels.insert(0, "command_center")
        if ev.event_type in ("counter_flow", "stopped_in_flow", "speeding"):
            channels.append("review_queue")
        for ch in channels:
            self.log.write(f"route_{ch}", ev.to_dict())
        return channels
