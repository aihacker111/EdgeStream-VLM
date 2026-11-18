"""Shared data types for the reasoning half of the pipeline (after perception)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Candidate:
    """A possible event proposed by the cheap tier, waiting for the gate."""
    cand_id: int
    type_hint: str                     # event-card id, e.g. "counter_flow"
    track_ids: list[int]               # global ids from perception
    t_start: float
    t_last: float
    s_rule: float                      # [0,1] how strongly the geometric rule fires
    s_anom: float                      # [0,1] statistical anomaly strength
    severity_prior: float              # [0,1] from the event card
    cell: tuple[int, int] = (0, 0)     # flow-field cell where it happens
    quality: float = 1.0               # mean perception track quality (TQS)
    recovered_frac: float = 0.0        # share of recent boxes that were QAM-interpolated
    kappa: Optional[float] = None      # appearance reliability (None without ReID)
    evidence: dict = field(default_factory=dict)
    priority: float = 0.0              # filled by the gate
    times_sent: int = 0
    last_sent_t: Optional[float] = None
    last_sent_priority: float = 0.0
    box: Optional[tuple] = None        # tlwh for candidates without a track (motion blobs, static change)
    vlm_p: Optional[float] = None      # last VLM P(anomaly) for this candidate (None = never verified)
    vlm_type: Optional[str] = None     # last VLM category

    @property
    def duration(self) -> float:
        return max(0.0, self.t_last - self.t_start)

    def key(self) -> tuple:
        return (self.type_hint, tuple(sorted(self.track_ids)))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 2)
        return d


@dataclass
class Verdict:
    """Parsed VLM answer for one candidate."""
    cand_id: int
    verdict: str                       # yes | no | uncertain
    event_type: str
    severity: str                      # low | medium | high
    description: str
    confidence: Optional[float]        # from token logprobs when the backend exposes them
    parse_ok: bool
    raw_text: str
    backend: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    visual_tokens: int


@dataclass
class Event:
    event_id: int
    event_type: str
    severity: str
    t_start: float
    t_end: float
    t_alert: float                     # logical time the alert was raised
    track_ids: list[int]
    description: str
    confidence: Optional[float]
    cand_ids: list[int] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
