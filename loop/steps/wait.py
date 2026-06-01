"""STEP 4: Wait for launched experiments to finish.

Extracted from ``Orchestrator._wait_for_completion`` and
``Orchestrator._ai_early_stop_check`` (module #8, 2026-06-01). The body is
verbatim — every status transition, every release_gpu call, and the
zombie-detection branch all behave identically. The only differences are:

    * ``self`` → ``ctx`` (no behavior change).
    * ``self._running_procs`` lives on the orchestrator and is reached via
      ``ctx.running_procs`` (same dict object — orchestrator passes its own
      handle when building the context).

Why this stayed one big function:
    The early-stop / zombie / natural-exit / timeout paths share the same
    "release GPU + remove from pending + set DB status" tail. Splitting it
    risks drifting one path's cleanup. A future refactor should route
    everything through ``ExperimentRun.__exit__`` (experiment_run.py) which
    centralizes that cleanup; this PR is a pure relocation.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Tuple

from ...agents.base_agent import call_claude_cli
from ...experiment_db import ExperimentRecord

if TYPE_CHECKING:
    from ..runner import IterationContext


def wait_for_completion(
    ctx: "IterationContext",
    *,
    experiments: List[ExperimentRecord],
) -> None:
    """Block until every launched experiment exits, times out, or is stopped."""
    pending = set(exp.id for exp in experiments if exp.id in ctx.running_procs)
    timeout = ctx.config.experiment_timeout_hours * 3600
    start_time = time.time()

    # Track last epoch we did an AI check for each experiment
    last_ai_check_epoch: Dict[str, int] = {}

    # Get current best for context
    best_exp = ctx.get_best_valid_experiment()
    current_best_mae = best_exp.avg_score_mae if best_exp else 999.0

    while pending:
        if time.time() - start_time > timeout:
            print("  Experiment timeout reached. Marking remaining as failed.")
            for exp_id in pending:
                ctx.db.update_status(
                    exp_id, "failed",
                    completed_at=datetime.now().isoformat(),
                )
                proc = ctx.running_procs.pop(exp_id, None)
                if proc:
                    proc.terminate()
                exp = ctx.db.get_experiment(exp_id)
                if exp and exp.gpu_id is not None:
                    ctx.gpu_manager.release_gpu(exp.gpu_id)
            break

        done = set()
        early_stopped = set()

        for exp_id in pending:
            proc = ctx.running_procs.get(exp_id)
            if proc is None or proc.poll() is not None:
                done.add(exp_id)
                rc = getattr(proc, "returncode", None) if proc else None
                if rc is not None and rc != 0:
                    ctx.db.update_status(
                        exp_id, "failed",
                        completed_at=datetime.now().isoformat(),
                    )
                exp = ctx.db.get_experiment(exp_id)
                if exp and exp.gpu_id is not None:
                    ctx.gpu_manager.release_gpu(exp.gpu_id)
                ctx.running_procs.pop(exp_id, None)
                continue

            # --- Zombie process detection (PID died but proc object lingering) ---
            exp = ctx.db.get_experiment(exp_id)
            if exp and exp.pid and not ctx.runner.check_process_alive(exp.pid):
                print(
                    f"  [ZOMBIE] {exp.exp_name} (pid={exp.pid}) is dead — "
                    f"cleaning up"
                )
                done.add(exp_id)
                results = ctx.runner.collect_results(exp)
                if results:
                    ctx.db.update_status(
                        exp_id, "early_stopped",
                        completed_at=datetime.now().isoformat(),
                    )
                else:
                    ctx.db.update_status(
                        exp_id, "failed",
                        completed_at=datetime.now().isoformat(),
                    )
                if exp.gpu_id is not None:
                    ctx.gpu_manager.release_gpu(exp.gpu_id)
                ctx.running_procs.pop(exp_id, None)
                continue

            # --- AI-based early stopping check ---
            exp = ctx.db.get_experiment(exp_id)
            if not exp:
                continue

            curve = ctx.runner.get_training_curve(exp)
            if not curve:
                continue

            current_ep = curve[-1][0]
            last_checked = last_ai_check_epoch.get(exp_id, 0)

            # Check every 10 epochs after min_epochs, avoid redundant calls
            if (current_ep >= ctx.config.early_stop_min_epochs
                    and current_ep - last_checked >= 10):
                last_ai_check_epoch[exp_id] = current_ep
                should_stop, reason = _ai_early_stop_check(
                    exp, curve, current_best_mae
                )
                if should_stop:
                    print(f"  [AI EARLY STOP] {exp.exp_name}: {reason}")
                    early_stopped.add(exp_id)

        # Kill early-stopped experiments and collect their results
        for exp_id in early_stopped:
            proc = ctx.running_procs.pop(exp_id, None)
            if proc:
                proc.terminate()
                time.sleep(2)  # Give process time to flush logs
            exp = ctx.db.get_experiment(exp_id)
            if exp and exp.gpu_id is not None:
                ctx.gpu_manager.release_gpu(exp.gpu_id)
            # Collect results from training log before marking status
            results = ctx.runner.collect_results(exp)
            if results:
                ctx.db.update_status(
                    exp_id, "early_stopped",
                    completed_at=datetime.now().isoformat(),
                )
            else:
                ctx.db.update_status(
                    exp_id, "failed",
                    completed_at=datetime.now().isoformat(),
                )
            done.add(exp_id)

        pending -= done

        if pending:
            for exp_id in pending:
                exp = ctx.db.get_experiment(exp_id)
                best_mae, best_ep, current_ep = (
                    ctx.runner.get_intermediate_best_mae(exp)
                )
                progress = ctx.runner.get_training_progress(exp)
                mae_info = (
                    f" (best_mae={best_mae:.4f}@ep{best_ep})"
                    if best_mae else ""
                )
                if progress:
                    print(f"  [{exp.exp_name}] {progress}{mae_info}")
            time.sleep(ctx.config.experiment_poll_interval)


def _ai_early_stop_check(
    exp: ExperimentRecord,
    curve: list,
    current_best_mae: float,
) -> Tuple[bool, str]:
    """Ask Claude to analyze training curve and decide whether to early-stop.

    Returns (should_stop: bool, reason: str). Verbatim from the old
    ``Orchestrator._ai_early_stop_check`` — prompt text intentionally
    unchanged to keep the Monitor's conservative stop bias.
    """
    # Format training curve concisely
    curve_text = (
        "Epoch | Avg MAE | 눈_살짝 | 눈_질끈 | 이마_주름 | 입_이 | 입_우 | 안면_무표정\n"
    )
    for epoch_num, avg_mae, action_maes in curve:
        row = f"  {epoch_num:3d}  | {avg_mae:.4f} "
        for action in [
            "눈_살짝감기", "눈_질끈감기", "이마_주름",
            "입_이", "입_우", "안면_무표정",
        ]:
            row += f"| {action_maes.get(action, 0):.4f} "
        curve_text += row + "\n"

    best_epoch, best_mae = min(curve, key=lambda x: x[1])[:2]
    current_epoch = curve[-1][0]

    prompt = f"""You are a CONSERVATIVE training monitor for a facial palsy severity prediction model.
Your DEFAULT should be CONTINUE. Only STOP if the evidence is overwhelming.

## Experiment: {exp.exp_name}
## Code change type: {exp.code_change_type}
## Proposal rationale: {exp.proposal_rationale[:200] if exp.proposal_rationale else 'N/A'}

## Current Best (other experiments): {current_best_mae:.4f}
## Target: 0.49 (lower is better)
## Total epochs planned: 100

## Training Curve (Score MAE — lower is better):
{curve_text}
## Summary:
- Current epoch: {current_epoch}/100
- Best MAE so far: {best_mae:.4f} at epoch {best_epoch}
- Gap to current best: {best_mae - current_best_mae:+.4f}
- Epochs since best: {current_epoch - best_epoch}

## STRICT Decision Criteria (ALL must be true to STOP):
1. At least 40 epochs completed
2. No improvement for the last 15+ epochs (not just 5-10)
3. Best MAE is more than 0.10 worse than current best (not marginal gaps)
4. No individual action showing significant improvement trend

## CONTINUE if ANY of these are true:
- Experiment is below epoch 40
- ANY individual action MAE is still improving (even if avg is flat)
- Best MAE is within 0.08 of current best (close enough to potentially beat it)
- Loss is still decreasing (model is still learning, even if test metrics lag)
- Epochs since best < 20

## IMPORTANT: Previous experiments showed that early stopping was too aggressive.
## Many experiments were stopped at epoch 20-30 when they might have improved further.
## When in doubt, ALWAYS choose CONTINUE.

## Response Format (MUST be exactly one of):
DECISION: CONTINUE
REASON: <one sentence>

or

DECISION: STOP
REASON: <one sentence>"""

    print(f"  [Monitor] Analyzing {exp.exp_name} at epoch {current_epoch}...")
    response = call_claude_cli(
        prompt=prompt,
        model="haiku",
        max_budget_usd=0.05,
        timeout_seconds=30,
        tools="",
    )

    if not response:
        return False, ""  # Default to continue if AI call fails

    response = response.strip()
    if "DECISION: STOP" in response:
        # Extract reason
        reason = "AI determined experiment should stop"
        for line in response.split("\n"):
            if line.strip().startswith("REASON:"):
                reason = line.strip().replace("REASON:", "").strip()
                break
        return True, reason

    return False, ""
