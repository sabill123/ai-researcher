"""ExperimentRun: context manager for a single GPU training job lifecycle.

Wraps GPU reservation, subprocess launch, status bookkeeping, and the
SIGTERM → wait → SIGKILL → GPU release escalation behind a single
``with`` block. The exception path is the load-bearing case: today's
orchestrator has three separate code paths (timeout / early-stop /
zombie cleanup) that each call ``terminate()`` directly; missing one
leaks the GPU reservation and leaves a stale ``running`` row in the DB.
This wrapper closes that gap by funneling every exit through ``__exit__``.

External contracts that must not change:
    * DB schema (status transitions: proposed → running → completed/early_stopped/failed)
    * GPU reservation API (reserve_gpu / release_gpu / release_by_experiment)
    * Sandbox layout (experiment_dir/code, experiment_dir/results, training.log, train.pid)
    * Subprocess launch flags (nohup + setsid + CUDA_VISIBLE_DEVICES)

stdlib + anna_v2.utils only. Type hints reference the existing classes
via TYPE_CHECKING to avoid import cycles with experiment_runner /
experiment_db that already import from each other.
"""

from __future__ import annotations

import os
import signal
import time
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional

from .utils.logging_setup import get_logger

if TYPE_CHECKING:
    from .experiment_db import ExperimentDB, ExperimentRecord
    from .experiment_runner import ExperimentRunner, _DetachedProcess
    from .gpu_manager import GPUManager


logger = get_logger(__name__)


# Tunables — kept as module constants so tests and watchdog can import them.
TERMINATE_GRACE_SECONDS = 10  # SIGTERM → wait this long → SIGKILL
KILL_SETTLE_SECONDS = 2  # SIGKILL → wait this long → release GPU
POLL_INTERVAL_SECONDS = 30  # default wait_completion poll cadence


class ExperimentRun:
    """Manage one training subprocess's full lifecycle.

    Typical use::

        run = ExperimentRun(db, gpu_manager, runner, experiment, gpu_id)
        with run:
            metrics = run.wait_completion(timeout_sec=4 * 3600)
        # GPU released, DB finalized, even on exception

    Attributes:
        experiment: the ExperimentRecord this run owns.
        gpu_id: GPU index reserved for this run.
        proc: _DetachedProcess once ``__enter__`` has launched it.
        final_status: set by ``__exit__`` or explicit ``finalize`` —
            one of {completed, early_stopped, failed, killed}.
    """

    def __init__(
        self,
        db: "ExperimentDB",
        gpu_manager: "GPUManager",
        runner: "ExperimentRunner",
        experiment: "ExperimentRecord",
        gpu_id: int,
    ) -> None:
        self.db = db
        self.gpu_manager = gpu_manager
        self.runner = runner
        self.experiment = experiment
        self.gpu_id = gpu_id

        self.proc: Optional["_DetachedProcess"] = None
        self.final_status: Optional[str] = None
        self._entered = False
        self._released = False

    # ------------------------------------------------------------------ enter
    def __enter__(self) -> "ExperimentRun":
        """Reserve GPU, mark running, launch subprocess.

        ``launch_experiment`` in the current runner already does the
        GPU reservation + DB status update internally, so we don't
        double-reserve. We do verify the reservation landed and record
        ``_entered`` so ``__exit__`` knows to release on the way out.
        """
        if self._entered:
            raise RuntimeError("ExperimentRun already entered")

        logger.info(
            "ExperimentRun.enter exp=%s gpu=%d",
            self.experiment.exp_name, self.gpu_id,
        )
        self.proc = self.runner.launch_experiment(self.experiment, self.gpu_id)
        self._entered = True
        return self

    # ----------------------------------------------------------------- wait
    def wait_completion(
        self,
        timeout_sec: float,
        *,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        on_poll: Optional[Callable[["ExperimentRun"], Optional[str]]] = None,
    ) -> Optional[str]:
        """Block until the process exits, times out, or ``on_poll`` says stop.

        Args:
            timeout_sec: wall-clock budget. On expiry returns "timeout"
                and the caller is expected to let ``__exit__`` clean up.
            poll_interval: seconds between liveness checks.
            on_poll: optional hook called every poll iteration with
                ``self``. If it returns a non-None string (e.g.
                "early_stopped"), wait returns that immediately so the
                caller can drive the early-stop path.

        Returns:
            "exited" on natural process exit
            "timeout" on budget exhaustion
            <string from on_poll> on hook-driven termination
        """
        if not self._entered or self.proc is None:
            raise RuntimeError("wait_completion() requires entered context")

        deadline = time.monotonic() + max(0.0, timeout_sec)
        while True:
            if self.proc.poll() is not None:
                return "exited"

            if on_poll is not None:
                verdict = on_poll(self)
                if verdict:
                    return verdict

            if time.monotonic() >= deadline:
                return "timeout"

            time.sleep(poll_interval)

    # ------------------------------------------------------------- terminate
    def terminate_then_kill(
        self,
        *,
        grace_seconds: int = TERMINATE_GRACE_SECONDS,
    ) -> bool:
        """Send SIGTERM, wait, escalate to SIGKILL if still alive.

        Returns True if the process is gone by the time we return, False
        if even SIGKILL failed to clear it (rare; usually a kernel-D
        state). Safe to call when proc is already dead.
        """
        if self.proc is None:
            return True

        if self.proc.poll() is not None:
            return True

        logger.info(
            "ExperimentRun.terminate exp=%s pid=%s SIGTERM",
            self.experiment.exp_name, self.proc.pid,
        )
        try:
            self.proc.terminate()
        except (OSError, ProcessLookupError):
            pass

        # Poll for graceful exit
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return True
            time.sleep(0.5)

        # Escalate
        logger.warning(
            "ExperimentRun.terminate exp=%s pid=%s SIGKILL (graceful timeout)",
            self.experiment.exp_name, self.proc.pid,
        )
        try:
            self.proc.kill()
        except (OSError, ProcessLookupError):
            pass

        # Give the kernel a moment
        time.sleep(KILL_SETTLE_SECONDS)
        return self.proc.poll() is not None

    # ------------------------------------------------------------- finalize
    def finalize(self, status: str) -> None:
        """Record final status and release the GPU reservation.

        Idempotent: calling twice (e.g. once explicitly, again in
        ``__exit__``) is a no-op the second time. The status passed on
        the first call wins.
        """
        if self._released:
            return

        if status not in (
            "completed", "early_stopped", "failed",
            "killed", "timeout",
        ):
            # Defensive — accept unknown statuses but log them; never
            # crash the cleanup path.
            logger.warning("ExperimentRun.finalize unexpected status=%s", status)

        self.final_status = status
        try:
            self.db.update_status(
                self.experiment.id,
                status,
                completed_at=datetime.now().isoformat(),
            )
        except Exception:
            logger.exception(
                "ExperimentRun.finalize db.update_status failed exp=%s",
                self.experiment.exp_name,
            )

        try:
            self.gpu_manager.release_by_experiment(self.experiment.id)
        except Exception:
            logger.exception(
                "ExperimentRun.finalize release_by_experiment failed exp=%s",
                self.experiment.exp_name,
            )

        self._released = True

    # --------------------------------------------------------------- exit
    def __exit__(self, exc_type, exc, tb) -> bool:
        """Always: stop the process, release GPU, record final status.

        Returns False so any exception in the ``with`` body propagates;
        we only suppress on an explicit ``return True`` (which we never
        want — losing context on an orchestrator-level error is bad).
        """
        try:
            still_alive = self.proc is not None and self.proc.poll() is None
            if still_alive:
                killed = self.terminate_then_kill()
                if self.final_status is None:
                    self.final_status = "killed" if killed else "failed"
            else:
                if self.final_status is None:
                    # Process exited on its own with no explicit verdict — let
                    # the collector decide later; we still need a DB row that
                    # isn't 'running'. Use 'completed' so downstream filters
                    # pick it up; result collection will refine.
                    self.final_status = "completed"

            self.finalize(self.final_status)
        except Exception:
            logger.exception(
                "ExperimentRun.__exit__ cleanup raised exp=%s",
                self.experiment.exp_name,
            )

        return False
