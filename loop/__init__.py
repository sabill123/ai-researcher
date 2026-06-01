"""Loop package: per-iteration step extraction of the 3-agent ANNA loop.

Module #8 refactor (2026-06-01). Previously every Researcher → Engineer →
Wait → Collect → Judge step lived inside a single 1266-LOC god-function on
``Orchestrator``. The new shape:

    Orchestrator
        owns: config, db, gpu_manager, memory, agents, sandbox, runner, tree
        run(): outer while + iteration loop + flock + SIGTERM, delegates to
        IterationRunner.run_one(ctx).

    IterationRunner (loop/runner.py)
        run_one(ctx) → executes one Researcher→Engineer→Wait→Collect→Judge
        pass. All step bodies are imported from ``loop.steps.*`` so they can
        be unit-tested without booting the full orchestrator.

    IterationContext (loop/runner.py)
        dataclass that carries shared handles into each step. Keeps step
        signatures uniform: ``def run_X(ctx, ...)``.

The step modules are intentionally thin — each is a verbatim cut from the
old orchestrator, only renamed from ``self.X`` to ``ctx.X``. Behavior is
unchanged.
"""

from .state import LoopState, load_state, save_state
from .runner import IterationContext, IterationRunner

__all__ = [
    "LoopState",
    "load_state",
    "save_state",
    "IterationContext",
    "IterationRunner",
]
