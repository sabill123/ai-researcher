"""STEP 6: Judge agent reviews the iteration and writes back insights.

Extracted from ``Orchestrator._run_judge`` (module #8, 2026-06-01). Three
pieces of context are computed in-flight and passed to Judge.judge():

    * parent_comparisons — text showing each child experiment's MAE delta
      vs. its parent ("+0.012 *** IMPROVED ***" markers).
    * stagnation_diagnosis — auto-generated when worst-action concentration
      or parent reuse exceeds thresholds; surfaces "you are stuck" so the
      Judge can override its own ``next_direction``.
    * vlm_baseline — reference numbers for the VLM auto-labeler.

The post-judge bookkeeping (retrospectives, insight re-examination,
new_insights) writes through ``ReflectiveMemory`` (atomic_write under the
hood) and updates ``ctx.judge_direction`` which the NEXT iteration's
Researcher/Engineer steps read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ...experiment_db import ExperimentRecord

if TYPE_CHECKING:
    from ..runner import IterationContext


def run_judge(
    ctx: "IterationContext",
    *,
    latest_results: List[ExperimentRecord],
    current_best_mae: float,
) -> None:
    """Run Judge agent on completed experiments. Mutates ``ctx.judge_direction``."""
    latest_text = ctx.db.format_experiments_for_prompt(latest_results)
    top_text = ctx.db.format_experiments_for_prompt(ctx.get_top_valid_k(15))

    updated_best = ctx.get_best_valid_experiment()
    updated_best_mae = (
        updated_best.avg_score_mae if updated_best else current_best_mae
    )

    # Include VLM baseline info for Judge context
    vlm_context = ctx.get_vlm_baseline_context()

    # Build parent-vs-child comparison text
    parent_comparisons = []
    for exp in latest_results:
        if exp.parent_experiment_id:
            parent = ctx.db.get_experiment(exp.parent_experiment_id)
            if parent and parent.avg_score_mae and exp.avg_score_mae:
                delta = parent.avg_score_mae - exp.avg_score_mae
                parent_comparisons.append(
                    f"  {exp.exp_name}: parent={parent.exp_name} "
                    f"(parent_MAE={parent.avg_score_mae:.4f}, "
                    f"child_MAE={exp.avg_score_mae:.4f}, "
                    f"delta={delta:+.4f}"
                    f"{'  *** IMPROVED ***' if delta > 0 else ''})"
                )
    parent_comparison_text = (
        "\n".join(parent_comparisons) if parent_comparisons else None
    )

    # Inject stagnation diagnosis so Judge can break the local-search loop.
    # NOTE: research_context MUST stay a dict (Judge calls .get on it) — the
    # stagnation text is passed as a separate argument, never concatenated in.
    stagnation_text = ctx.compute_stagnation_diagnosis()
    if stagnation_text:
        print("\n  [stagnation diagnosis attached to Judge]")

    judge_result = ctx.judge.judge(
        completed_experiments=top_text,
        latest_experiments=latest_text,
        existing_insights=ctx.memory.format_for_llm(),
        research_context=ctx.research_context,
        current_best_mae=updated_best_mae,
        iteration=ctx.state.iteration,
        vlm_baseline=vlm_context,
        parent_comparisons=parent_comparison_text,
        stagnation_diagnosis=stagnation_text,
    )

    if judge_result:
        analysis = judge_result.get("analysis", "")
        if analysis:
            print(f"\n  Analysis: {analysis[:300]}...")

        # NEW (2026-05-14): Judge now emits per-experiment retrospectives
        # and explicit re-examination of past insights, so that
        # documented insights become a live feedback signal instead of
        # inert text. Persist both before adding any new insights.
        retros = judge_result.get("retrospectives", [])
        if retros:
            ctx.memory.add_retrospectives_from_judge(retros)
        reexam = judge_result.get("insight_reexamination", [])
        if reexam:
            ctx.memory.apply_insight_reexamination(reexam)
            cited = sum(1 for r in reexam if isinstance(r, dict))
            print(
                f"  Re-examined {cited} past insights "
                f"(confirmed/contradicted/retired counts updated)"
            )

        new_insights = judge_result.get("new_insights", [])
        if new_insights:
            ctx.memory.add_insights_from_judge(new_insights)
            print(f"  Added {len(new_insights)} new insights")

        ctx.judge_direction = judge_result
        direction = judge_result.get("next_direction", {})
        print(f"  Strategy: {direction.get('strategy', 'unknown')}")
        if direction.get("engineer_guidance"):
            print(
                f"  Engineer guidance: "
                f"{direction['engineer_guidance'][:150]}"
            )
        if direction.get("researcher_guidance"):
            print(
                f"  Researcher guidance: "
                f"{direction['researcher_guidance'][:150]}"
            )
        if direction.get("cited_insights"):
            cited_short = "; ".join(
                (c[:80] + "...") if len(c) > 80 else c
                for c in direction["cited_insights"][:3]
            )
            print(f"  Cited insights driving direction: {cited_short}")
