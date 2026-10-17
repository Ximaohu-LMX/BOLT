"""Opt-in, append-only phase timing for efficiency experiments.

Set EFFICIENCY_PROFILE=1 and EFFICIENCY_TIMING_FILE=<path>. The default code
path is a no-op and experiment semantics are unchanged.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _enabled() -> bool:
    return os.getenv("EFFICIENCY_PROFILE", "").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }


class PhaseTimer:
    def __init__(self) -> None:
        self.enabled = _enabled()
        self.path = os.getenv("EFFICIENCY_TIMING_FILE", "")
        self.started = time.perf_counter()
        self.previous = self.started
        self.sequence = 0

    def _sync(self) -> None:
        if not self.enabled or os.getenv("EFFICIENCY_SYNC_CUDA", "1") == "0":
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def mark(self, phase: str, **extra: object) -> None:
        if not self.enabled or not self.path:
            return
        self._sync()
        now = time.perf_counter()
        row = {
            "schema_version": 1,
            "run_id": os.getenv("FORMAL_RUN_ID", ""),
            "task_id": os.getenv("EFFICIENCY_TASK_ID", ""),
            "repeat_id": os.getenv("EFFICIENCY_REPEAT_ID", ""),
            "cache_protocol": os.getenv("EFFICIENCY_CACHE_PROTOCOL", ""),
            "sequence": self.sequence,
            "phase": phase,
            "elapsed_since_previous_seconds": now - self.previous,
            "elapsed_since_start_seconds": now - self.started,
            **extra,
        }
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.previous = now
        self.sequence += 1


timer = PhaseTimer()
