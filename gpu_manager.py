"""
GPU availability detection and allocation for shared server.

Reservations are persisted to ``data/gpu_state.json`` via atomic writes so
that an orchestrator crash/restart does not strand GPUs. On startup the
persisted reservations are loaded and validated against the recorded
reserver PID — entries whose owner process is dead are released
automatically (``validate_reservations``).

``get_available_gpus`` calls ``validate_reservations`` on every invocation
so a stale reservation cannot block scheduling indefinitely. The persisted
schema is::

    {
        "version": 1,
        "reservations": {
            "<gpu_id>": {
                "experiment_id": "<exp_id>",
                "pid":           <reserver_pid_int>,
                "reserved_at":   "<iso8601 ts>"
            },
            ...
        }
    }

The public API (``reserve_gpu``, ``release_gpu``, ``release_by_experiment``,
``get_available_gpus``, ``wait_for_gpu``, ``format_status``, ``get_gpu_status``)
is unchanged. ``reserve_gpu`` gains an optional ``pid`` kwarg that defaults
to ``os.getpid()`` for backward compatibility.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from .config import SystemConfig
from .utils.atomic_write import atomic_write_json, safe_load_json
from .utils.logging_setup import get_logger

logger = get_logger(__name__)

# Persisted state schema version. Bump if the on-disk shape changes.
_STATE_SCHEMA_VERSION = 1

# Default persistence location relative to anna_v2 package dir. We resolve
# this lazily (in __init__) rather than at import time so tests can override
# via the ``state_path`` constructor arg.
_DEFAULT_STATE_FILENAME = "gpu_state.json"


@dataclass
class GPUStatus:
    id: int
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_pct: int


@dataclass
class Reservation:
    """Schema for one persisted GPU reservation."""

    experiment_id: str
    pid: int
    reserved_at: str  # ISO8601

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "pid": self.pid,
            "reserved_at": self.reserved_at,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Reservation":
        return cls(
            experiment_id=str(d.get("experiment_id", "")),
            pid=int(d.get("pid", 0)),
            reserved_at=str(d.get("reserved_at", "")),
        )


def _default_state_path() -> str:
    """Resolve the default gpu_state.json path under <project_root>/anna_v2/data/."""
    cfg = SystemConfig()
    return os.path.join(cfg.anna_v2_dir, "data", _DEFAULT_STATE_FILENAME)


def _pid_alive(pid: int) -> bool:
    """Return True if ``pid`` exists and is signalable by the current user.

    Uses ``os.kill(pid, 0)`` which sends no signal but performs the
    permission/existence check the kernel uses for real signals. A
    PermissionError means the PID is alive but owned by another user — we
    treat that as alive (it's not our place to release someone else's GPU
    just because we can't signal it).
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # PID exists but is owned by another user — treat as alive.
        return True
    except OSError:
        # Defensive: any other OS error, err on the side of "alive" so we
        # don't accidentally double-allocate a GPU.
        return True


class GPUManager:
    def __init__(
        self,
        max_gpus: int = 8,
        excluded_gpus=(1, 7),
        state_path: Optional[str] = None,
    ):
        """Construct the manager and restore any persisted reservations.

        Args:
            max_gpus: maximum number of GPUs on the host (informational).
            excluded_gpus: GPU ids the scheduler must never touch. Defaults
                to ``(1, 7)`` — GPU 1 is the housemate, GPU 7 is the
                production self-monitoring inference worker. Pass an empty
                tuple to schedule on every GPU.
            state_path: override the persistence path (mainly for tests).
                Defaults to ``<anna_v2_dir>/data/gpu_state.json``.
        """
        self.max_gpus = max_gpus
        # GPUs the orchestrator must never schedule on (housemate / production
        # inference worker). Skipped early in get_available_gpus(). Stored as
        # a set for O(1) membership checks.
        self._excluded = set(excluded_gpus)
        self._state_path = state_path or _default_state_path()

        # gpu_id -> Reservation. Restored from disk on construction.
        self._reservations: Dict[int, Reservation] = {}
        self._restore_from_disk()

    # ------------------------------------------------------------- persistence
    def _restore_from_disk(self) -> None:
        """Load persisted reservations. Corrupt/missing file -> empty state."""
        raw = safe_load_json(self._state_path, default=None)
        if not raw:
            logger.info("GPUManager: no persisted state at %s", self._state_path)
            return

        version = raw.get("version") if isinstance(raw, dict) else None
        if version != _STATE_SCHEMA_VERSION:
            logger.warning(
                "GPUManager: state schema version=%r != %d; ignoring persisted state",
                version, _STATE_SCHEMA_VERSION,
            )
            return

        entries = raw.get("reservations", {}) or {}
        restored = 0
        for gid_str, entry in entries.items():
            try:
                gid = int(gid_str)
                self._reservations[gid] = Reservation.from_dict(entry)
                restored += 1
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "GPUManager: skipping malformed reservation gpu=%r: %s",
                    gid_str, exc,
                )

        logger.info(
            "GPUManager: restored %d reservation(s) from %s",
            restored, self._state_path,
        )
        # Immediately drop reservations whose owner process is dead.
        self.validate_reservations()

    def _persist(self) -> None:
        """Atomically write the current reservation table to disk."""
        payload = {
            "version": _STATE_SCHEMA_VERSION,
            "reservations": {
                str(gid): res.to_dict()
                for gid, res in self._reservations.items()
            },
        }
        try:
            atomic_write_json(self._state_path, payload)
        except OSError as exc:
            # Persistence failure must not crash the scheduler — in-memory
            # state is still correct, we just lose crash-recovery for this
            # update. Log loudly so operators notice.
            logger.error(
                "GPUManager: failed to persist state to %s: %s",
                self._state_path, exc,
            )

    # -------------------------------------------------------------- validation
    def validate_reservations(self) -> List[int]:
        """Drop reservations whose owner PID is no longer alive.

        Called automatically from ``get_available_gpus`` so callers never
        have to remember to invoke it. Also safe to call manually (e.g.
        from a watchdog).

        Returns:
            List of GPU ids that were released.
        """
        released: List[int] = []
        for gid, res in list(self._reservations.items()):
            if not _pid_alive(res.pid):
                logger.warning(
                    "GPUManager: releasing GPU %d (exp=%s pid=%d dead)",
                    gid, res.experiment_id, res.pid,
                )
                del self._reservations[gid]
                released.append(gid)

        if released:
            self._persist()
        return released

    # ------------------------------------------------------------ nvidia-smi
    def get_gpu_status(self) -> List[GPUStatus]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                logger.warning(
                    "nvidia-smi failed rc=%d stderr=%s",
                    result.returncode, result.stderr.strip(),
                )
                return []

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue
                gpus.append(GPUStatus(
                    id=int(parts[0]),
                    memory_total_mb=int(parts[1]),
                    memory_used_mb=int(parts[2]),
                    memory_free_mb=int(parts[3]),
                    utilization_pct=int(parts[4]),
                ))
            return gpus
        except Exception as exc:
            logger.exception("nvidia-smi query raised: %s", exc)
            return []

    # ------------------------------------------------------------- scheduling
    def get_available_gpus(self, min_free_memory_mb: int = 20000,
                           max_utilization_pct: int = 10) -> List[int]:
        # Always sweep dead reservations first so a crashed orchestrator
        # cannot strand GPUs across restarts.
        self.validate_reservations()

        gpus = self.get_gpu_status()
        available = []
        for gpu in gpus:
            if gpu.id in self._reservations:
                continue
            if gpu.id in self._excluded:
                continue
            if gpu.memory_free_mb >= min_free_memory_mb and \
               gpu.utilization_pct <= max_utilization_pct:
                available.append(gpu.id)
        return available

    def reserve_gpu(
        self,
        gpu_id: int,
        experiment_id: str,
        pid: Optional[int] = None,
    ) -> bool:
        """Reserve ``gpu_id`` for ``experiment_id``.

        Records the reserver process PID (defaults to ``os.getpid()``) so
        ``validate_reservations`` can release the slot if the reserver
        dies before calling ``release_gpu``.

        Returns False if the GPU is already reserved.
        """
        if gpu_id in self._reservations:
            return False
        owner_pid = pid if pid is not None else os.getpid()
        self._reservations[gpu_id] = Reservation(
            experiment_id=experiment_id,
            pid=int(owner_pid),
            reserved_at=datetime.now().isoformat(timespec="seconds"),
        )
        self._persist()
        logger.info(
            "GPUManager: reserved GPU %d for exp=%s (pid=%d)",
            gpu_id, experiment_id, owner_pid,
        )
        return True

    def release_gpu(self, gpu_id: int):
        res = self._reservations.pop(gpu_id, None)
        if res is not None:
            self._persist()
            logger.info(
                "GPUManager: released GPU %d (exp=%s)",
                gpu_id, res.experiment_id,
            )

    def release_by_experiment(self, experiment_id: str):
        to_remove = [
            gid for gid, res in self._reservations.items()
            if res.experiment_id == experiment_id
        ]
        for gid in to_remove:
            del self._reservations[gid]
        if to_remove:
            self._persist()
            logger.info(
                "GPUManager: released %d GPU(s) for exp=%s: %s",
                len(to_remove), experiment_id, to_remove,
            )

    def wait_for_gpu(self, min_free_memory_mb: int = 20000,
                     max_utilization_pct: int = 10,
                     timeout_seconds: int = 1800,
                     poll_interval: int = 60) -> Optional[int]:
        elapsed = 0
        while elapsed < timeout_seconds:
            available = self.get_available_gpus(min_free_memory_mb, max_utilization_pct)
            if available:
                return available[0]
            time.sleep(poll_interval)
            elapsed += poll_interval
        return None

    # ------------------------------------------------------------- diagnostic
    def format_status(self) -> str:
        gpus = self.get_gpu_status()
        if not gpus:
            return "Failed to query GPU status"

        lines = [f"{'GPU':>4} {'Used':>8} {'Free':>8} {'Total':>8} {'Util':>6} {'Reserved':>20}"]
        lines.append("-" * 65)
        for gpu in gpus:
            res = self._reservations.get(gpu.id)
            if res is None:
                reserved_label = "-"
            else:
                reserved_label = f"{res.experiment_id} (pid={res.pid})"
            lines.append(
                f"{gpu.id:>4} {gpu.memory_used_mb:>7}M {gpu.memory_free_mb:>7}M "
                f"{gpu.memory_total_mb:>7}M {gpu.utilization_pct:>5}% {reserved_label:>20}"
            )
        return "\n".join(lines)
