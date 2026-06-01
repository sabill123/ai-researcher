"""STEP 7: Persist loop state at the end of an iteration.

Extracted from ``Orchestrator.run`` (module #8, 2026-06-01). The old inline
code did ``self.state.iteration = iteration + 1; self._save_state()`` — we
preserve that, but now the save routes through the new atomic
``loop.state.save_state`` helper (tmp + fsync + rename) so a crash between
iterations cannot leave ``loop_state.json`` half-written.

ReflectiveMemory persistence happens inside each ``memory.add_*`` /
``memory.apply_*`` call (see judge.py), which already uses
``atomic_write_json``. We do not re-save it here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runner import IterationContext


def save_checkpoint(ctx: "IterationContext") -> None:
    """Advance the iteration counter and persist loop_state.json atomically."""
    ctx.state.iteration = ctx.iteration + 1
    ctx.save_state()
