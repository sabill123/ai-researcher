"""STEP 5: Collect results for each launched experiment.

Extracted from ``Orchestrator.run`` (module #8, 2026-06-01). For every
record we either reuse the results already gathered during early-stop
(wait.py marked them with status="early_stopped") or pull them now from
the on-disk training log. The tree is updated with each completed result
so the Judge gets accurate ``next_direction`` stats.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List

from ...experiment_db import ExperimentRecord

if TYPE_CHECKING:
    from ..runner import IterationContext


def collect_results(
    ctx: "IterationContext",
    *,
    launched: List[ExperimentRecord],
    current_best_mae: float,
) -> List[ExperimentRecord]:
    """Return the list of experiments with valid avg_score_mae.

    Side effects (verbatim from the old inline code):
        * DB rows transition to ``completed`` or ``failed`` here.
        * ``ctx.tree.record_experiment`` is called per completed record.
        * Per-action MAE and "NEW BEST!" banner are printed.
    """
    latest_results: List[ExperimentRecord] = []
    for exp in launched:
        exp = ctx.db.get_experiment(exp.id)
        if exp.status == "failed":
            print(f"  {exp.exp_name}: FAILED")
            continue

        # Early-stopped experiments already had results collected
        if exp.status == "early_stopped" and exp.avg_score_mae:
            print(
                f"  {exp.exp_name}: avg_score_mae = "
                f"{exp.avg_score_mae:.4f} (early stopped)"
            )
        else:
            results = ctx.runner.collect_results(exp)
            if results is None:
                ctx.db.update_status(
                    exp.id, "failed",
                    completed_at=datetime.now().isoformat(),
                )
                print(f"  {exp.exp_name}: FAILED (no results)")
                continue
            ctx.db.update_status(
                exp.id, "completed",
                completed_at=datetime.now().isoformat(),
            )

        exp = ctx.db.get_experiment(exp.id)
        print(f"  {exp.exp_name}: avg_score_mae = {exp.avg_score_mae:.4f}")

        if exp.per_action_results:
            for k, v in sorted(exp.per_action_results.items()):
                if k.startswith("test_") and k.endswith("_score_mae"):
                    action = k[len("test_"):-len("_score_mae")]
                    print(f"    {action}: {v:.4f}")

        # Record in tree
        improvement = current_best_mae - (exp.avg_score_mae or 999)
        ctx.tree.record_experiment(
            category=exp.code_change_type,
            experiment_id=exp.id,
            mae_improvement=improvement,
            result_mae=exp.avg_score_mae or 999,
        )

        if exp.avg_score_mae and exp.avg_score_mae < current_best_mae:
            print(
                f"  *** NEW BEST! Improved by "
                f"{current_best_mae - exp.avg_score_mae:.4f} ***"
            )

        latest_results.append(exp)

    return latest_results
