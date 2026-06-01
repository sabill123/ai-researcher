"""Shared pytest fixtures for the ANNA v2 test suite.

The fixtures here keep each test isolated by giving callers their own
temporary directory and a fresh in-memory-ish SQLite ExperimentDB. All
persistent paths are routed through ``tmp_path`` so a failing test can
never corrupt the real ``anna_v2/data/`` files.
"""
from __future__ import annotations

import os
import sys

import pytest

# Make ``anna_v2`` importable when pytest is run from the project root
# without an installed package. The tests live at anna_v2/tests/, so we
# add the project root (parent of anna_v2) to sys.path once on import.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@pytest.fixture()
def tmp_data_dir(tmp_path):
    """Return a fresh data directory under pytest's tmp_path."""
    d = tmp_path / "data"
    d.mkdir()
    return str(d)


@pytest.fixture()
def db_path(tmp_data_dir):
    """Filesystem path for a throwaway ExperimentDB."""
    return os.path.join(tmp_data_dir, "experiments.db")


@pytest.fixture()
def experiment_db(db_path):
    """Build and tear down an ExperimentDB on a temp path."""
    from anna_v2.experiment_db import ExperimentDB
    db = ExperimentDB(db_path)
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def memory_path(tmp_data_dir):
    return os.path.join(tmp_data_dir, "reflective_memory.json")


@pytest.fixture()
def gpu_state_path(tmp_data_dir):
    return os.path.join(tmp_data_dir, "gpu_state.json")


@pytest.fixture()
def system_config(tmp_path):
    """Return a SystemConfig pointed at a tmp project root.

    We override ``project_root`` to a subdir under the allowed base path
    so ``assert_safe_path`` would pass; tests that don't need the path
    safety check can ignore this.
    """
    from anna_v2.config import SystemConfig, ALLOWED_BASE_PATH

    # Use the allowed prefix so any safety asserts pass even though the
    # subdir is unique per test.
    root = os.path.join(ALLOWED_BASE_PATH, "research-2026-01")
    cfg = SystemConfig(project_root=root)
    return cfg
