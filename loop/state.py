"""Loop state persistence.

Module #8 extraction (2026-06-01): LoopState was previously defined inside
``orchestrator.py``. It is the small JSON document that records how far the
3-agent loop has advanced (iteration, budgets used, lifecycle status). The
class is intentionally a pure dataclass — no DB / agent / GPU dependencies —
so it can be loaded from a watchdog or status CLI without booting the full
``Orchestrator`` stack.

Persistence routes through ``utils.atomic_write.atomic_write_json`` so a
crash mid-write cannot leave the loop unable to restart from a half-written
``loop_state.json``. The old in-orchestrator implementation used a plain
``open()/json.dump()`` which had been linked to one of the earlier "stuck on
restart" incidents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..utils.atomic_write import atomic_write_json, safe_load_json


@dataclass
class LoopState:
    """In-memory representation of ``loop_state.json``.

    Field names must stay byte-for-byte compatible with the JSON on disk —
    the dashboard, watchdog, and ``anna_v2 status`` all read this file
    directly. Adding a new field is safe (older readers ignore unknowns);
    renaming or removing is not.
    """

    iteration: int = 0
    total_gpu_hours_used: float = 0.0
    total_claude_usd: float = 0.0
    status: str = "idle"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "total_gpu_hours_used": self.total_gpu_hours_used,
            "total_claude_usd": self.total_claude_usd,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoopState":
        return cls(
            iteration=d.get("iteration", 0),
            total_gpu_hours_used=d.get("total_gpu_hours_used", 0.0),
            total_claude_usd=d.get("total_claude_usd", 0.0),
            status=d.get("status", "idle"),
        )


def load_state(path: str) -> LoopState:
    """Load loop state, returning a default if file is absent or corrupt.

    ``safe_load_json`` quarantines a corrupt JSON file (renamed with a
    ``.corrupted.<ts>`` suffix) instead of crashing — the loop should
    restart fresh rather than refuse to boot if disk goes bad.
    """
    raw = safe_load_json(path, default=None)
    if not isinstance(raw, dict):
        return LoopState()
    return LoopState.from_dict(raw)


def save_state(path: str, state: LoopState) -> None:
    """Atomically persist loop state to ``path``.

    tmp + fsync + rename inside ``atomic_write_json``. Crash-safe.
    """
    atomic_write_json(path, state.to_dict())
