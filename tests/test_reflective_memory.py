"""Tests for ``anna_v2.reflective_memory.ReflectiveMemory``.

Covers:
* prune() grace window: brand-new insights survive even when the cap is
  exceeded.
* prune() weighted score: cited insights outrank uncited ones with the
  same confidence.
* corrupt JSON on disk is quarantined and the memory boots empty.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timedelta

import pytest

from anna_v2.reflective_memory import Insight, ReflectiveMemory


def _make_insight(
    observation: str,
    *,
    confidence: float = 0.5,
    cited_count: int = 0,
    age_seconds: int = 0,
    parameter_affected: str = "",
) -> Insight:
    """Build an Insight with an explicit created_at so we can simulate age.

    A negative ``age_seconds`` would put creation in the future (always in
    grace). A positive value backdates ``created_at`` by that many seconds
    so the insight falls outside the grace window when desired.
    """
    created = (datetime.now() - timedelta(seconds=age_seconds)).isoformat()
    return Insight(
        observation=observation,
        evidence=[],
        confidence=confidence,
        cited_count=cited_count,
        created_at=created,
        last_updated=created,
        parameter_affected=parameter_affected or None,
    )


def test_new_insights_protected_by_grace_window(memory_path):
    mem = ReflectiveMemory(memory_path)
    # Fill the memory beyond the prune cap with brand-new insights.
    for i in range(40):
        mem.add_insight(_make_insight(
            f"fresh observation {i}",
            confidence=0.1,
            parameter_affected=f"param_{i}",  # avoid _find_similar merges
        ))
    # All 40 are within the 30-minute grace window so prune cannot evict.
    mem.prune(max_insights=10)
    assert len(mem.insights) == 40, (
        f"grace window should preserve fresh insights, got {len(mem.insights)}"
    )


def test_prune_evicts_old_uncited_low_confidence(memory_path):
    mem = ReflectiveMemory(memory_path)
    # Backdate by 2 hours -> outside grace.
    old_age = 2 * 60 * 60
    for i in range(35):
        mem.add_insight(_make_insight(
            f"old observation {i}",
            confidence=0.2,
            age_seconds=old_age,
            parameter_affected=f"param_{i}",
        ))
    mem.prune(max_insights=20)
    assert len(mem.insights) == 20


def test_cited_count_outranks_uncited(memory_path):
    """prune() ranks by ``cited_count*2 + confidence`` — cited beats uncited
    even with lower confidence."""
    mem = ReflectiveMemory(memory_path)
    # Two old insights: one cited many times with low confidence, one
    # never cited with high confidence. The cited one MUST survive.
    cited = _make_insight(
        "cited insight",
        confidence=0.30,
        cited_count=5,
        age_seconds=2 * 3600,
        parameter_affected="cited_param",
    )
    uncited = _make_insight(
        "uncited insight",
        confidence=0.55,
        cited_count=0,
        age_seconds=2 * 3600,
        parameter_affected="uncited_param",
    )
    mem.add_insight(cited)
    mem.add_insight(uncited)
    # Add filler old insights to force pruning to bite.
    for i in range(50):
        mem.add_insight(_make_insight(
            f"filler {i}",
            confidence=0.10,
            age_seconds=2 * 3600,
            parameter_affected=f"filler_param_{i}",
        ))
    mem.prune(max_insights=10)
    observations = {ins.observation for ins in mem.insights}
    assert "cited insight" in observations, (
        f"prune dropped the cited insight; kept={observations}"
    )


def test_corrupt_memory_file_quarantined_on_load(memory_path):
    # Write a torn JSON file at the memory path.
    with open(memory_path, "w") as fh:
        fh.write("[ {bad json}")
    mem = ReflectiveMemory(memory_path)
    # Boots with an empty list rather than crashing.
    assert mem.insights == []
    # A *.corrupted.<ts> sibling must exist.
    siblings = glob.glob(f"{memory_path}.corrupted.*")
    assert len(siblings) >= 1


def test_bootstrap_idempotent(memory_path):
    mem = ReflectiveMemory(memory_path)
    mem.bootstrap()
    n1 = len(mem.insights)
    assert n1 > 0
    mem.bootstrap()  # second call must be a no-op
    assert len(mem.insights) == n1


def test_round_trip_persists_to_disk(memory_path):
    mem = ReflectiveMemory(memory_path)
    mem.add_insight(_make_insight(
        "persisted obs",
        confidence=0.9,
        cited_count=3,
        parameter_affected="lr",
    ))
    # New instance over the same file must see the insight.
    mem2 = ReflectiveMemory(memory_path)
    obs = [i.observation for i in mem2.insights]
    assert "persisted obs" in obs
    cited = next(i for i in mem2.insights if i.observation == "persisted obs")
    assert cited.cited_count == 3
