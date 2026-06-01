"""Tests for the loop helper functions used by the iteration steps.

These helpers live in ``anna_v2.loop.helpers`` and feed downstream agent
prompts (Researcher / Engineer / Judge). They mix DB reads with a v1-era
``db._conn()`` call pattern, so the tests construct a small shim that
adapts the v2 ExperimentDB's attribute-style ``_conn`` to the
context-manager-callable interface helpers expect.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import List

import pytest

from anna_v2.experiment_db import ExperimentDB, ExperimentRecord
from anna_v2.loop.helpers import (
    compute_stagnation_diagnosis,
    get_parent_experiment,
    get_weak_actions,
)


# ----------------------------------------------------------- DB compat shim

class _CompatDB:
    """Wrap an ExperimentDB so ``db._conn()`` works the way helpers expect.

    The v1 ExperimentDB exposed ``_conn`` as a method returning a fresh
    connection context-manager. The v2 module stores the connection as
    an attribute. ``loop/helpers.py`` was carried over from v1 and still
    calls ``with db._conn() as conn:`` — to keep the helpers testable we
    delegate every attribute lookup to the underlying DB but override
    ``_conn`` to be callable.
    """

    def __init__(self, db: ExperimentDB):
        self._db = db

    def __getattr__(self, item):
        return getattr(self._db, item)

    @contextmanager
    def _conn(self):
        # Use the underlying connection directly; do NOT close it here.
        yield self._db._conn


# ------------------------------------------------------------ fixtures

@pytest.fixture()
def compat_db(experiment_db):
    return _CompatDB(experiment_db)


def _make_rec(
    *,
    exp_name: str,
    mae: float,
    per_action: dict,
    proposed_by: str = "engineer",
    parent_id: str = None,
    status: str = "completed",
) -> ExperimentRecord:
    return ExperimentRecord(
        exp_name=exp_name,
        status=status,
        config={},
        avg_score_mae=mae,
        per_action_results=per_action,
        proposed_by=proposed_by,
        code_change_type="hyperparameter",
        parent_experiment_id=parent_id,
    )


def _per_action(values):
    keys = ["눈_살짝감기", "눈_질끈감기", "이마_주름", "입_이", "입_우", "안면_무표정"]
    return {f"test_{k}_score_mae": v for k, v in zip(keys, values)}


# ------------------------------------------------------------ get_weak_actions

def test_get_weak_actions_default_when_no_best():
    """Without a best experiment, fall back to a fixed default focus list."""
    result = get_weak_actions(db=None, best_exp=None)
    assert result == ["입_우", "입_이"]


def test_get_weak_actions_returns_three_worst(compat_db, experiment_db):
    """The three highest per-action MAEs should make the focus list."""
    per_action = _per_action([0.10, 0.20, 0.30, 0.40, 0.90, 0.80])
    best = _make_rec(exp_name="best", mae=0.45, per_action=per_action)
    experiment_db.add_experiment(best)
    actions = get_weak_actions(compat_db, best)
    assert isinstance(actions, list)
    assert len(actions) <= 3
    # 입_우 (0.90) and 입_이 (0.80) and 이마_주름 (0.30) are the top 3.
    assert "입_우" in actions
    assert "입_이" in actions


def test_get_weak_actions_ignores_ci_suffix_keys(compat_db, experiment_db):
    """`test_입_이_score_mae_ci_upper` must not produce phantom actions."""
    per_action = _per_action([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    per_action["test_입_이_score_mae_ci_upper"] = 1.5  # noise key
    best = _make_rec(exp_name="best_ci", mae=0.35, per_action=per_action)
    experiment_db.add_experiment(best)
    actions = get_weak_actions(compat_db, best)
    for a in actions:
        assert "ci_upper" not in a, f"phantom action leaked into list: {a}"
        assert "ci_lower" not in a


# ------------------------------------------------------ stagnation diagnosis

def test_compute_stagnation_diagnosis_returns_empty_when_no_history(
    compat_db, system_config,
):
    """With <10 completed experiments, diagnosis returns ''."""
    assert compute_stagnation_diagnosis(compat_db, system_config) == ""


def test_compute_stagnation_diagnosis_emits_lines_with_enough_history(
    compat_db, experiment_db, system_config, tmp_path,
):
    """With >=10 completed experiments, the diagnosis should be populated."""
    # Build a fake sandbox dir so has_valid_sandbox_code can succeed for at
    # least one experiment. We point experiment_dir at tmp_path and pre-create
    # all the sandbox files the config expects.
    sandbox = tmp_path / "sandbox" / "code"
    sandbox.mkdir(parents=True)
    for fname in system_config.sandbox_files:
        (sandbox / fname).write_text("# stub\n")

    for i in range(12):
        rec = _make_rec(
            exp_name=f"exp_{i:03d}",
            mae=0.50 + 0.001 * i,
            per_action=_per_action([0.40, 0.45, 0.30, 0.50, 0.90, 0.55]),
            parent_id=("exp_000" if i > 0 else None),
        )
        # Mark a completion timestamp so the ORDER BY completed_at finds them.
        from datetime import datetime
        rec.completed_at = datetime.now().isoformat()
        # Only the first record gets a valid sandbox dir.
        if i == 0:
            rec.experiment_dir = str(tmp_path / "sandbox")
        experiment_db.add_experiment(rec)

    text = compute_stagnation_diagnosis(compat_db, system_config)
    assert isinstance(text, str)
    assert text.startswith("=== STAGNATION DIAGNOSIS")
    assert "Worst-action concentration" in text or "Current best MAE" in text


# ----------------------------------------------------- get_parent_experiment

def test_get_parent_experiment_none_when_no_valid_candidates(
    compat_db, system_config, experiment_db,
):
    """If no candidate has a valid sandbox, parent selection returns None."""
    rec = _make_rec(
        exp_name="no_sandbox",
        mae=0.30,
        per_action=_per_action([0.30, 0.30, 0.30, 0.30, 0.30, 0.30]),
    )
    experiment_db.add_experiment(rec)
    chosen = get_parent_experiment(compat_db, system_config)
    assert chosen is None


def test_get_parent_experiment_picks_valid_candidate(
    compat_db, system_config, experiment_db, tmp_path, capsys,
):
    """A candidate with a valid sandbox should be selected by UCB1."""
    sandbox = tmp_path / "exp_dir" / "code"
    sandbox.mkdir(parents=True)
    for fname in system_config.sandbox_files:
        (sandbox / fname).write_text("# stub\n")

    rec = _make_rec(
        exp_name="valid_parent",
        mae=0.30,
        per_action=_per_action([0.30, 0.30, 0.30, 0.30, 0.30, 0.30]),
    )
    rec.experiment_dir = str(tmp_path / "exp_dir")
    experiment_db.add_experiment(rec)

    chosen = get_parent_experiment(compat_db, system_config)
    assert chosen is not None
    assert chosen.exp_name == "valid_parent"
