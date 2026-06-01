"""
VLM-based severity evaluation for ANNA v2.
Runs VLM auto-labeling on labeled data and stores results as a baseline experiment.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, Optional

from .config import SystemConfig
from .experiment_db import ExperimentDB, ExperimentRecord

VLM_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "exp_vlm_autolabel", "run_vlm_alignment.py"
)

# Best model from alignment test (2026-03-19)
DEFAULT_VLM_MODEL = "claude-opus-4-6"

# Action name mapping for consistent metric keys
ACTIONS = [
    "눈_살짝감기", "눈_질끈감기", "이마_주름",
    "입_이", "입_우", "안면_무표정",
]


def run_vlm_baseline(
    config: SystemConfig,
    db: ExperimentDB,
    iteration: int,
    model: str = DEFAULT_VLM_MODEL,
    samples_per_action: int = 0,  # 0 = full evaluation
) -> Optional[ExperimentRecord]:
    """Run VLM evaluation and store as a baseline experiment.

    Returns the experiment record, or None if already exists or failed.
    """
    # Check if VLM baseline for this model already exists
    existing = db.get_all_experiments()
    for exp in existing:
        if (exp.code_change_type == "vlm_baseline"
                and exp.config.get("vlm_model") == model
                and exp.status == "completed"):
            print(f"  VLM baseline for {model} already exists: {exp.exp_name} (MAE={exp.avg_score_mae:.4f})")
            return exp

    if not os.path.exists(VLM_SCRIPT):
        print(f"  WARNING: VLM script not found: {VLM_SCRIPT}")
        return None

    # Create experiment record
    mode = "full" if samples_per_action == 0 else "quick"
    exp_name = f"vlm_baseline_{model.replace('-', '_').replace('.', '')}_{mode}"

    record = ExperimentRecord(
        exp_name=exp_name,
        status="running",
        priority=0.0,
        expected_improvement=0.0,
        confidence=1.0,
        config={
            "vlm_model": model,
            "vlm_mode": mode,
            "samples_per_action": samples_per_action,
            "api_base_url": "https://gateway.letsur.ai/v1",
        },
        proposed_by="vlm_baseline",
        code_change_type="vlm_baseline",
        proposal_rationale=f"VLM auto-labeling baseline using {model}",
        iteration=iteration,
        started_at=datetime.now().isoformat(),
    )
    db.insert_experiment(record)

    print(f"  Running VLM baseline: {exp_name} (model={model}, mode={mode})")

    # Build command
    cmd = [
        sys.executable, VLM_SCRIPT,
        "--mode", mode,
        "--model", model,
    ]
    if samples_per_action > 0:
        cmd.extend(["--samples", str(samples_per_action)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
            cwd=os.path.dirname(VLM_SCRIPT),
        )

        if result.returncode != 0:
            print(f"  VLM baseline failed: {result.stderr[:200]}")
            db.update_status(record.id, "failed",
                             completed_at=datetime.now().isoformat())
            return None

        # Parse results from output directory
        output_dir = os.path.join(os.path.dirname(VLM_SCRIPT), "output")
        metrics = _parse_vlm_results(output_dir, model, mode)

        if metrics is None:
            print(f"  VLM baseline: no results found")
            db.update_status(record.id, "failed",
                             completed_at=datetime.now().isoformat())
            return None

        avg_mae = metrics.get("avg_score_mae", 999.0)
        per_action = metrics.get("per_action_results", {})

        db.update_results(
            record.id,
            avg_score_mae=avg_mae,
            per_action_results=per_action,
            results_dir=output_dir,
        )
        db.update_status(record.id, "completed",
                         completed_at=datetime.now().isoformat())

        print(f"  VLM baseline completed: avg_score_mae={avg_mae:.4f}")
        for action in ACTIONS:
            key = f"test_{action}_score_mae"
            if key in per_action:
                print(f"    {action}: {per_action[key]:.4f}")

        return db.get_experiment(record.id)

    except subprocess.TimeoutExpired:
        print(f"  VLM baseline timed out")
        db.update_status(record.id, "failed",
                         completed_at=datetime.now().isoformat())
        return None
    except Exception as e:
        print(f"  VLM baseline error: {e}")
        db.update_status(record.id, "failed",
                         completed_at=datetime.now().isoformat())
        return None


def _parse_vlm_results(output_dir: str, model: str, mode: str) -> Optional[Dict]:
    """Parse the latest VLM alignment results JSON."""
    if not os.path.exists(output_dir):
        return None

    # Find the most recent matching output file
    prefix = f"vlm_alignment_{mode}_"
    candidates = []
    for f in os.listdir(output_dir):
        if f.startswith(prefix) and f.endswith(".json"):
            candidates.append(f)

    if not candidates:
        # Try broader match
        for f in os.listdir(output_dir):
            if f.startswith("vlm_alignment_") and f.endswith(".json"):
                candidates.append(f)

    if not candidates:
        return None

    latest = sorted(candidates)[-1]
    path = os.path.join(output_dir, latest)

    with open(path) as f:
        data = json.load(f)

    # Extract metrics in ANNA-compatible format
    overall = data.get("overall", {})
    if not overall and model in data:
        overall = data[model].get("overall", {})

    per_action_data = data.get("per_action", {})
    if not per_action_data and model in data:
        per_action_data = data[model].get("per_action", {})

    if not overall:
        return None

    avg_mae = overall.get("mae", 999.0)

    # Convert to ANNA per_action_results format: test_{action}_score_mae
    per_action_results = {}
    for action in ACTIONS:
        action_data = per_action_data.get(action, {})
        action_mae = action_data.get("mae", avg_mae)
        per_action_results[f"test_{action}_score_mae"] = action_mae

    per_action_results["vlm_model"] = model
    per_action_results["vlm_overall_mae"] = avg_mae
    per_action_results["vlm_exact_match"] = overall.get("exact_match", 0)
    per_action_results["vlm_within_1"] = overall.get("within_1", 0)
    per_action_results["vlm_spearman_r"] = overall.get("spearman_r", 0)
    per_action_results["source"] = "vlm_auto_labeling"

    return {
        "avg_score_mae": avg_mae,
        "per_action_results": per_action_results,
    }
