"""Logical cost accounting, independent of how fast the GPU happens to be."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BudgetLedger:
    vlm_calls: int = 0
    visual_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_s: float = 0.0
    by_type: dict = field(default_factory=dict)
    by_reason: dict = field(default_factory=dict)   # "candidate" | "query" | ...

    def record(self, event_type: str, visual: int, prompt: int, completion: int,
               wall_s: float, reason: str = "candidate") -> None:
        self.vlm_calls += 1
        self.visual_tokens += int(visual)
        self.prompt_tokens += int(prompt)
        self.completion_tokens += int(completion)
        self.wall_s += float(wall_s)
        self.by_type[event_type] = self.by_type.get(event_type, 0) + 1
        self.by_reason[reason] = self.by_reason.get(reason, 0) + 1

    def summary(self, duration_s: float) -> dict:
        minutes = max(duration_s / 60.0, 1e-9)
        return {
            "vlm_calls": self.vlm_calls,
            "vlm_calls_per_min": round(self.vlm_calls / minutes, 2),
            "visual_tokens": self.visual_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "completion_tokens_per_call": round(self.completion_tokens / max(self.vlm_calls, 1), 1),
            "vlm_wall_s": round(self.wall_s, 2),
            "by_type": self.by_type,
            "by_reason": self.by_reason,
        }
