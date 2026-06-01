"""STEP 3: Engineer implements proposals, validates sandbox, launches on GPU.

Extracted from ``Orchestrator.run`` (module #8, 2026-06-01). For each
proposal we:

    1. Create a sandbox dir (inheriting parent code when present).
    2. Build the DB record with parent linkage.
    3. If the proposal contains code changes, ask Engineer to write them,
       restore read-only files, validate the sandbox, capture the diff,
       and reject empty diffs (no actual change → no point training).
    4. Persist proposal.json, score with EI×conf/√cost, insert into DB.
    5. Reserve a GPU (waiting if necessary) and launch the experiment.

Tracking the running subprocess on ``ctx.running_procs`` is the same
contract the wait/collect steps rely on — same key (record.id), same
value type (``_DetachedProcess`` or ``subprocess.Popen``).
"""

from __future__ import annotations

import json
import math
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ...experiment_db import ExperimentRecord

if TYPE_CHECKING:
    from ..runner import IterationContext


def run_engineer_implement(
    ctx: "IterationContext",
    *,
    proposals: List[Dict[str, Any]],
    parent_exp: Optional[ExperimentRecord],
    parent_code_dir: Optional[str],
) -> List[ExperimentRecord]:
    """Implement, validate, and launch every viable proposal.

    Returns the list of ExperimentRecords that successfully launched. A
    proposal that fails any of the gates (implement / validate / empty
    diff / GPU timeout) is dropped without entering this list.
    """
    iteration = ctx.iteration

    print(f"\n--- STEP 3: ENGINEER (Implement & Launch) ---")
    launched: List[ExperimentRecord] = []

    for proposal in proposals:
        exp_name = proposal.get("exp_name", f"exp_{iteration:03d}_unnamed")

        # Force critical config overrides
        cfg = proposal.get("config", {})
        cfg["head_type"] = "feature_fusion"  # ALWAYS use feature_fusion (never linear)
        cfg["batch_size"] = min(cfg.get("batch_size", 12), 12)  # Cap at 12 to avoid OOM
        proposal["config"] = cfg

        # Create sandbox — inherit code from parent experiment
        exp_dir = ctx.sandbox.create_sandbox(
            exp_name, parent_code_dir=parent_code_dir
        )
        code_dir = os.path.join(exp_dir, "code")

        # Create experiment record with parent linkage
        record = ExperimentRecord(
            exp_name=exp_name,
            status="proposed",
            priority=0.0,
            expected_improvement=proposal.get("expected_improvement", 0.01),
            confidence=proposal.get("confidence", 0.5),
            config=proposal.get("config", {}),
            experiment_dir=exp_dir,
            proposed_by="engineer",
            parent_experiment_id=parent_exp.id if parent_exp else None,
            code_change_type=proposal.get("code_change_type", "hyperparameter"),
            research_context=(
                json.dumps(ctx.research_context)
                if ctx.research_context else None
            ),
            proposal_rationale=proposal.get("rationale", ""),
            iteration=iteration,
        )

        # Implement code changes (if not pure hyperparameter)
        if (proposal.get("code_change_type") != "hyperparameter"
                and proposal.get("changes")):
            base_dir = parent_code_dir if parent_code_dir else ctx.config.scripts_dir
            success = ctx.engineer.implement(
                proposal, code_dir,
                baseline_code_dir=base_dir,
            )
            if not success:
                print(f"  Implementation failed for {exp_name}")
                record.status = "failed"
                ctx.db.add_experiment(record)
                continue

            # Restore read-only files
            ctx.sandbox.restore_readonly_files(exp_dir)

            # Validate sandbox
            is_valid, errors = ctx.sandbox.validate_sandbox(exp_dir)
            if not is_valid:
                print(f"  Validation failed for {exp_name}:")
                for err in errors:
                    print(f"    - {err}")
                record.status = "failed"
                record.proposal_rationale += (
                    f" | Validation errors: {'; '.join(errors)}"
                )
                ctx.db.add_experiment(record)
                continue

            # Save diff — reject if empty (no actual code changes)
            diff = ctx.sandbox.save_diff(exp_dir)
            if not diff or not diff.strip():
                print(f"  No code changes detected for {exp_name} — skipping")
                record.status = "failed"
                record.proposal_rationale += " | No code changes implemented"
                ctx.db.add_experiment(record)
                continue
            record.code_diff = diff
            record.status = "validated"
        else:
            record.status = "validated"

        # Save proposal.json
        proposal_path = os.path.join(exp_dir, "proposal.json")
        with open(proposal_path, "w") as f:
            json.dump(proposal, f, indent=2, ensure_ascii=False)

        # Score and add to DB
        cost = ctx.runner.estimate_cost_hours(record)
        ei = max(record.expected_improvement, 0.001)
        conf = max(record.confidence, 0.1)
        record.priority = (ei * conf) / math.sqrt(max(cost, 0.1))
        ctx.db.add_experiment(record)

        # Launch on GPU
        available_gpus = ctx.gpu_manager.get_available_gpus(
            min_free_memory_mb=ctx.config.min_free_memory_mb,
            max_utilization_pct=ctx.config.max_gpu_utilization_pct,
        )
        if not available_gpus:
            print("  Waiting for GPU...")
            gpu_id = ctx.gpu_manager.wait_for_gpu(
                min_free_memory_mb=ctx.config.min_free_memory_mb,
                max_utilization_pct=ctx.config.max_gpu_utilization_pct,
                timeout_seconds=ctx.config.gpu_wait_timeout,
                poll_interval=ctx.config.gpu_poll_interval,
            )
            if gpu_id is None:
                print("  GPU timeout. Skipping remaining.")
                break
            available_gpus = [gpu_id]

        gpu_id = available_gpus[0]
        try:
            proc = ctx.runner.launch_experiment(record, gpu_id)
            ctx.running_procs[record.id] = proc
            launched.append(record)
            print(f"  Launched {exp_name} on GPU {gpu_id} (pid={proc.pid})")
        except Exception as e:
            print(f"  Failed to launch {exp_name}: {e}")
            ctx.db.update_status(record.id, "failed")

    return launched
