"""STEP 1: Researcher agent invocation.

Extracted from ``Orchestrator.run`` (module #8, 2026-06-01). The Researcher
runs every ``researcher_interval`` iterations OR whenever the Judge has
emitted explicit ``researcher_guidance`` in its previous direction. The
result is stored on ``ctx.research_context`` so Engineer.propose can pick
it up downstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..runner import IterationContext
    from ...experiment_db import ExperimentRecord


def run_researcher(
    ctx: "IterationContext",
    *,
    best: Optional["ExperimentRecord"],
    current_best_mae: float,
) -> None:
    """Conditionally run the Researcher and update ``ctx.research_context``.

    Mutates ``ctx.research_context`` in place — the orchestrator's
    persistent cross-iteration handle. No-op when neither the cadence
    nor the Judge's guidance asks for new research.
    """
    iteration = ctx.iteration

    need_research = (
        iteration % ctx.config.researcher_interval == 0
        or (
            ctx.judge_direction
            and ctx.judge_direction.get("next_direction", {}).get(
                "researcher_guidance"
            )
        )
    )
    if not need_research:
        return

    print(f"\n--- STEP 1: RESEARCHER ---")
    weak_actions = ctx.get_weak_actions(best)
    judge_guidance = None
    if ctx.judge_direction:
        judge_guidance = ctx.judge_direction.get("next_direction", {}).get(
            "researcher_guidance"
        )

    ctx.research_context = ctx.researcher.research(
        current_best_mae=current_best_mae,
        weak_actions=weak_actions,
        judge_guidance=judge_guidance,
        existing_insights=ctx.memory.format_for_llm(),
        completed_techniques=ctx.memory.get_completed_techniques(),
    )
    if ctx.research_context:
        techniques = ctx.research_context.get("techniques", [])
        print(f"  Found {len(techniques)} techniques:")
        for t in techniques:
            print(
                f"    - {t.get('name', 'unnamed')}: "
                f"{t.get('core_idea', '')[:80]}"
            )
