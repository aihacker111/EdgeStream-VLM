"""JSONL logs: one file per channel, everything needed to recompute metrics offline."""
from __future__ import annotations

import json
from pathlib import Path


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


class JsonlLogger:
    def __init__(self, out_dir: str | Path):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._files = {}

    def write(self, channel: str, record: dict) -> None:
        f = self._files.get(channel)
        if f is None:
            f = open(self.dir / f"{channel}.jsonl", "w")
            self._files[channel] = f
        f.write(json.dumps(record, default=_default, ensure_ascii=False) + "\n")

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()
