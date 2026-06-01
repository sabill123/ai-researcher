"""Tests for ``anna_v2.gpu_manager.GPUManager`` reservation persistence.

Covers:
* a reservation made by one instance survives across process restarts
  (modelled by constructing a second instance over the same state path).
* ``validate_reservations`` drops entries whose owner PID is dead.
* ``release_gpu`` / ``release_by_experiment`` clear in-memory and disk
  state.
"""
from __future__ import annotations

import os

import pytest

from anna_v2.gpu_manager import GPUManager, _STATE_SCHEMA_VERSION
from anna_v2.utils.atomic_write import safe_load_json


@pytest.fixture()
def mgr(gpu_state_path):
    """Build a GPUManager pinned to a temp state path with no GPU excluded."""
    return GPUManager(max_gpus=8, excluded_gpus=(), state_path=gpu_state_path)


def test_reserve_persists_to_disk(mgr, gpu_state_path):
    ok = mgr.reserve_gpu(2, "exp_aaaa")
    assert ok is True
    raw = safe_load_json(gpu_state_path, default=None)
    assert isinstance(raw, dict)
    assert raw["version"] == _STATE_SCHEMA_VERSION
    assert "2" in raw["reservations"]
    assert raw["reservations"]["2"]["experiment_id"] == "exp_aaaa"
    assert raw["reservations"]["2"]["pid"] == os.getpid()


def test_reservation_restored_on_new_instance(gpu_state_path):
    m1 = GPUManager(excluded_gpus=(), state_path=gpu_state_path)
    m1.reserve_gpu(3, "exp_bbbb")
    # Simulate restart by building a second manager over the same file.
    m2 = GPUManager(excluded_gpus=(), state_path=gpu_state_path)
    # Our own PID is still alive so the reservation should survive
    # validate_reservations(), which runs in the constructor.
    assert 3 in m2._reservations
    assert m2._reservations[3].experiment_id == "exp_bbbb"


def test_double_reserve_returns_false(mgr):
    assert mgr.reserve_gpu(4, "exp_first") is True
    assert mgr.reserve_gpu(4, "exp_second") is False
    assert mgr._reservations[4].experiment_id == "exp_first"


def test_validate_releases_dead_pid(gpu_state_path):
    mgr = GPUManager(excluded_gpus=(), state_path=gpu_state_path)
    # Reserve under an obviously dead PID.
    # PID 1 is init/systemd — almost certainly alive. Use a very large PID
    # that is not running and very unlikely to be allocated.
    dead_pid = 999_999_990
    mgr.reserve_gpu(5, "exp_dead", pid=dead_pid)
    assert 5 in mgr._reservations
    released = mgr.validate_reservations()
    assert 5 in released
    assert 5 not in mgr._reservations
    # Disk state must also be cleared.
    raw = safe_load_json(gpu_state_path, default={"reservations": {}})
    assert "5" not in raw.get("reservations", {})


def test_release_gpu_clears_disk(mgr, gpu_state_path):
    mgr.reserve_gpu(6, "exp_release_me")
    mgr.release_gpu(6)
    assert 6 not in mgr._reservations
    raw = safe_load_json(gpu_state_path, default={"reservations": {}})
    assert "6" not in raw.get("reservations", {})


def test_release_by_experiment(mgr):
    mgr.reserve_gpu(0, "exp_multi")
    mgr.reserve_gpu(2, "exp_multi")
    mgr.reserve_gpu(3, "exp_other")
    mgr.release_by_experiment("exp_multi")
    assert 0 not in mgr._reservations
    assert 2 not in mgr._reservations
    assert 3 in mgr._reservations


def test_excluded_gpus_never_appear_available(gpu_state_path, monkeypatch):
    """``get_available_gpus`` must skip excluded ids regardless of nvidia-smi."""
    # Stub get_gpu_status to advertise GPUs 0..3 as free.
    from anna_v2 import gpu_manager as gm_mod
    from anna_v2.gpu_manager import GPUStatus

    fake = [
        GPUStatus(id=i, memory_total_mb=24000, memory_used_mb=0,
                  memory_free_mb=24000, utilization_pct=0)
        for i in range(4)
    ]
    monkeypatch.setattr(
        gm_mod.GPUManager, "get_gpu_status", lambda self: fake,
    )
    mgr = GPUManager(excluded_gpus=(1, 3), state_path=gpu_state_path)
    avail = mgr.get_available_gpus(min_free_memory_mb=1000, max_utilization_pct=10)
    assert 1 not in avail
    assert 3 not in avail
    assert set(avail) == {0, 2}
