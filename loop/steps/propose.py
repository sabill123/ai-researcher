"""STEP 2: Engineer proposes.

Extracted from ``Orchestrator.run`` (module #8, 2026-06-01). Selects a
parent experiment (via UCB1-ish exploration), assembles the prompt
context (top-k experiments, insights with VLM baseline appended, parent
retrospectives), and asks the Engineer to produce N proposals.

Behavior is identical to the inlined block — only the variable names that
were ``self.X`` are now ``ctx.X``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..runner import IterationContext
    from ...experiment_db import ExperimentRecord


def run_engineer_propose(
    ctx: "IterationContext",
    *,
    current_best_mae: float,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]],
           Optional["ExperimentRecord"], Optional[str]]:
    """Return ``(proposals, parent_info, parent_exp, parent_code_dir)``.

    parent_info is the dict passed to ``Engineer.propose`` (may be None when
    no valid parent exists, in which case Engineer is told to start from
    baseline). parent_exp + parent_code_dir are returned because the
    Implement step needs them to seed the sandbox copy.
    """
    iteration = ctx.iteration

    print(f"\n--- STEP 2: ENGINEER (Propose) ---")
    top_experiments = ctx.db.format_experiments_for_prompt(
        ctx.get_top_valid_k(15)
    )
    engineer_guidance = None
    if ctx.judge_direction:
        engineer_guidance = ctx.judge_direction.get("next_direction", {}).get(
            "engineer_guidance"
        )

    # Include VLM baseline info in insights for Engineer context
    insights_text = ctx.memory.format_for_llm()
    vlm_ctx = ctx.get_vlm_baseline_context()
    if vlm_ctx:
        insights_text += (
            f"\n\n## VLM Auto-Labeling Baseline (Reference)\n{vlm_ctx}"
        )

    # Select parent experiment for code inheritance
    parent_exp = ctx.get_parent_experiment()
    parent_code_dir: Optional[str] = None
    parent_info: Optional[Dict[str, Any]] = None
    if parent_exp and parent_exp.experiment_dir:
        parent_code_dir = os.path.join(parent_exp.experiment_dir, "code")
        if not os.path.isdir(parent_code_dir):
            parent_code_dir = None
            parent_exp = None
        else:
            # Build parent info dict for Engineer context
            diff_summary = ""
            if parent_exp.code_diff:
                diff_lines = parent_exp.code_diff.split("\n")
                added = sum(
                    1 for l in diff_lines
                    if l.startswith("+") and not l.startswith("+++")
                )
                removed = sum(
                    1 for l in diff_lines
                    if l.startswith("-") and not l.startswith("---")
                )
                diff_summary = f"+{added}/-{removed} lines"
            parent_info = {
                "exp_name": parent_exp.exp_name,
                "avg_score_mae": parent_exp.avg_score_mae,
                "code_change_type": parent_exp.code_change_type,
                "code_diff_summary": diff_summary or "N/A",
                "per_action_results": parent_exp.per_action_results,
                "proposal_rationale": parent_exp.proposal_rationale,
            }
            print(
                f"  Parent experiment: {parent_exp.exp_name} "
                f"(MAE={parent_exp.avg_score_mae:.4f})"
            )

    if not parent_exp:
        print(f"  No valid parent — using baseline code")

    # NEW (2026-05-14): hand the Engineer the Judge-written
    # retrospectives for the candidate parent and a few recent runs,
    # so the next proposal explicitly addresses prior failure modes
    # of the parent rather than silently repeating them.
    parent_names_for_retro = [parent_info["exp_name"]] if parent_info else []
    parent_retro_text = ctx.memory.format_retrospectives_for_llm(
        parent_exp_names=parent_names_for_retro, limit=8
    )

    proposals = ctx.engineer.propose(
        top_experiments=top_experiments,
        research_context=ctx.research_context,
        judge_guidance=engineer_guidance,
        existing_insights=insights_text,
        current_best_mae=current_best_mae,
        iteration=iteration,
        num_proposals=ctx.config.max_parallel,
        parent_experiment=parent_info,
        parent_retrospective_text=parent_retro_text,
    )

    if proposals:
        for p in proposals:
            print(
                f"  Proposed: {p.get('exp_name')} "
                f"({p.get('code_change_type', 'unknown')})"
            )
            print(f"    Rationale: {p.get('rationale', '')[:100]}")

    return proposals or [], parent_info, parent_exp, parent_code_dir
