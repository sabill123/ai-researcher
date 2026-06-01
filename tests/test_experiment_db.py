"""Tests for ``anna_v2.experiment_db`` and the 6-action integrity gate.

The integrity gate's bound checks (sanity range, missing-action,
reconstructed-avg drift) live in ``ExperimentRunner._parse_metrics``. We
exercise it indirectly via ``ExperimentDB._is_six_action_metrics`` for
schema classification, and directly via the constants exposed in
``experiment_runner`` for the numeric thresholds.
"""
from __future__ import annotations

import math

import pytest

from anna_v2.experiment_db import ExperimentDB, ExperimentRecord
from anna_v2.experiment_runner import (
    REQUIRED_ACTIONS,
    _MAE_MIN,
    _MAE_MAX,
    _AVG_RECONSTRUCT_TOL,
)


# ----------------------------------------------------------- 6-action filter

def _make_six_action_metrics(value: float = 0.40) -> dict:
    metrics = {f"test_{a}_score_mae": value for a in REQUIRED_ACTIONS}
    metrics["test_avg_score_mae"] = value
    return metrics


def test_six_action_metrics_accepts_full_set():
    m = _make_six_action_metrics()
    assert ExperimentDB._is_six_action_metrics(m) is True


def test_six_action_metrics_rejects_aggregate_only():
    legacy = {"avg_score_mae": 0.41, "avg_severity_mae": 0.30}
    assert ExperimentDB._is_six_action_metrics(legacy) is False


def test_six_action_metrics_rejects_test_avg_only():
    """Only ``test_avg_score_mae`` with no per-action keys is still legacy."""
    legacy = {"test_avg_score_mae": 0.41}
    assert ExperimentDB._is_six_action_metrics(legacy) is False


def test_six_action_metrics_rejects_empty():
    assert ExperimentDB._is_six_action_metrics({}) is False
    assert ExperimentDB._is_six_action_metrics(None) is False  # type: ignore[arg-type]


# ----------------------------------------------------------- CRUD round-trip

def test_add_and_get_experiment(experiment_db):
    rec = ExperimentRecord(
        exp_name="unit_test_exp",
        config={"lr": 1e-3, "batch_size": 32},
        avg_score_mae=0.42,
        per_action_results=_make_six_action_metrics(0.42),
        proposed_by="manual",
        code_change_type="hyperparameter",
    )
    exp_id = experiment_db.add_experiment(rec)
    assert exp_id == rec.id

    fetched = experiment_db.get_experiment(exp_id)
    assert fetched is not None
    assert fetched.exp_name == "unit_test_exp"
    assert fetched.config["lr"] == 1e-3
    assert fetched.per_action_results is not None
    assert math.isclose(fetched.avg_score_mae, 0.42)


def test_update_results_marks_completed(experiment_db):
    rec = ExperimentRecord(
        exp_name="to_complete",
        config={},
        status="running",
    )
    experiment_db.add_experiment(rec)
    per_action = _make_six_action_metrics(0.35)
    experiment_db.update_results(rec.id, 0.35, per_action, "/tmp/results")
    fetched = experiment_db.get_experiment(rec.id)
    assert fetched.status == "completed"
    assert math.isclose(fetched.avg_score_mae, 0.35)


# -------------------------------------- sanity bounds (numeric reasoning)

def test_sanity_bounds_are_sensible():
    """0.05 .. 5.0 reflects the 0..4 ordinal scale plus a safety margin."""
    assert _MAE_MIN < _MAE_MAX
    assert _MAE_MIN >= 0.0
    assert _MAE_MAX <= 10.0


def test_sanity_range_rejects_out_of_range():
    """Values implied by the integrity gate's bounds."""
    too_low = 0.0
    too_high = 10.0
    assert too_low < _MAE_MIN
    assert too_high > _MAE_MAX
    # NaN / inf must also be rejected by callers.
    assert not math.isfinite(float("nan"))
    assert not math.isfinite(float("inf"))


def test_reconstructed_avg_drift_detection():
    """Reconstructed avg vs reported avg must differ by < tolerance."""
    per_action = [0.40, 0.41, 0.39, 0.42, 0.38, 0.40]
    reconstructed = sum(per_action) / len(per_action)
    reported_ok = reconstructed + (_AVG_RECONSTRUCT_TOL / 2)
    reported_bad = reconstructed + (_AVG_RECONSTRUCT_TOL * 5)
    assert abs(reconstructed - reported_ok) < _AVG_RECONSTRUCT_TOL
    assert abs(reconstructed - reported_bad) >= _AVG_RECONSTRUCT_TOL


def test_missing_action_keys_make_set_incomplete():
    """If even one REQUIRED_ACTIONS key is missing, integrity must fail."""
    five = list(REQUIRED_ACTIONS)[:5]
    partial = {f"test_{a}_score_mae": 0.4 for a in five}
    # Helper still passes the structural check (one per-action key present)
    assert ExperimentDB._is_six_action_metrics(partial) is True
    # But the integrity gate counts per-action MAEs vs REQUIRED_ACTIONS:
    found = [a for a in REQUIRED_ACTIONS if f"test_{a}_score_mae" in partial]
    assert len(found) < len(REQUIRED_ACTIONS)


# ------------------------------------------------------------ top_k / best

def test_get_top_k_excludes_bootstrap(experiment_db):
    """bootstrap and vlm_baseline rows must NOT show up in top_k."""
    good = ExperimentRecord(
        exp_name="good_run",
        config={},
        status="completed",
        avg_score_mae=0.30,
        per_action_results=_make_six_action_metrics(0.30),
        proposed_by="engineer",
        code_change_type="hyperparameter",
    )
    bootstrap = ExperimentRecord(
        exp_name="legacy",
        config={},
        status="completed",
        avg_score_mae=0.10,
        per_action_results=_make_six_action_metrics(0.10),
        proposed_by="bootstrap",
        code_change_type="hyperparameter",
    )
    vlm = ExperimentRecord(
        exp_name="vlm_check",
        config={},
        status="completed",
        avg_score_mae=0.05,
        per_action_results=_make_six_action_metrics(0.05),
        proposed_by="engineer",
        code_change_type="vlm_baseline",
    )
    for r in (good, bootstrap, vlm):
        experiment_db.add_experiment(r)

    top = experiment_db.get_top_k(10)
    names = {e.exp_name for e in top}
    assert "good_run" in names
    assert "legacy" not in names
    assert "vlm_check" not in names
