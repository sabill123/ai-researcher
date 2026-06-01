"""
Experiment runner: launches training subprocess and collects results.
"""

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import SystemConfig, assert_safe_path
from .experiment_db import ExperimentDB, ExperimentRecord
from .gpu_manager import GPUManager
from .utils.logging_setup import get_logger, touch_heartbeat

logger = get_logger(__name__)

# Canonical 6 actions used by the baseline 6-action training pipeline.
# Used by the integrity gate to confirm test_metrics.json has all expected
# per-action MAEs before marking an experiment 'completed'.
REQUIRED_ACTIONS: Tuple[str, ...] = (
    "눈_살짝감기",
    "눈_질끈감기",
    "이마_주름",
    "입_이",
    "입_우",
    "안면_무표정",
)

# Per-action MAE sanity bounds. Below 0.05 is implausibly good (likely a
# bug or empty test set); above 5.0 is past the 0..4 ordinal scale and
# almost certainly garbage (NaN cast, swapped logits, etc.).
_MAE_MIN = 0.05
_MAE_MAX = 5.0

# Tolerance for reconstructed avg vs reported avg_score_mae. 0.01 is
# generous given typical 4-decimal logging precision.
_AVG_RECONSTRUCT_TOL = 0.01


def _resolve_heartbeat_dir(config: SystemConfig) -> str:
    """Where touch_heartbeat writes ``<name>.touch`` files.

    Co-located with the rest of the orchestrator's mutable state
    (``<anna_v2_dir>/data/heartbeats``) so the watchdog only needs to
    know one root.
    """
    return os.path.join(config.anna_v2_dir, "data", "heartbeats")


class _DetachedProcess:
    """Lightweight wrapper to track a nohup-detached process by PID."""

    # Seconds to wait between SIGTERM and the SIGKILL escalation. Long
    # enough for a well-behaved Python trainer to flush logs and tear
    # down DataLoader workers; short enough that an unkillable hang
    # doesn't stall the orchestrator's planner loop.
    _TERM_GRACE_SEC = 10.0
    # Seconds to wait after SIGKILL before releasing. The kernel reaps
    # almost immediately; this is a safety margin for the GPU driver
    # and any zombie cleanup.
    _KILL_GRACE_SEC = 2.0

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self) -> Optional[int]:
        """Return None if running, 0 if finished."""
        try:
            os.kill(self.pid, 0)
            return None  # Still running
        except (OSError, ProcessLookupError):
            return 0  # Finished

    def _wait_dead(self, timeout: float) -> bool:
        """Poll up to ``timeout`` seconds for the process to exit.

        Returns True if the process is gone, False if it's still alive.
        Uses a 0.2s poll cadence — fine-grained enough to keep total
        wait close to ``timeout`` without busy-spinning.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.poll() is not None:
                return True
            time.sleep(0.2)
        return self.poll() is not None

    def terminate(self):
        """Graceful-then-forceful shutdown with logging at each step.

        Sequence: SIGTERM → wait up to 10s → if alive, SIGKILL → wait
        up to 2s → release. Every transition is logged so postmortems
        can tell whether a stuck experiment exited cleanly or had to
        be force-killed (which often signals a CUDA-side wedge).
        """
        # Step 1: graceful SIGTERM
        try:
            os.kill(self.pid, signal.SIGTERM)
            logger.info("SIGTERM sent to pid=%s", self.pid)
        except (OSError, ProcessLookupError) as exc:
            logger.info(
                "SIGTERM skipped for pid=%s (already gone: %s)",
                self.pid, exc,
            )
            return

        # Step 2: wait for graceful exit
        if self._wait_dead(self._TERM_GRACE_SEC):
            logger.info(
                "pid=%s exited within %.1fs of SIGTERM",
                self.pid, self._TERM_GRACE_SEC,
            )
            return

        # Step 3: escalate to SIGKILL
        logger.warning(
            "pid=%s still alive after SIGTERM grace (%.1fs); escalating to SIGKILL",
            self.pid, self._TERM_GRACE_SEC,
        )
        try:
            os.kill(self.pid, signal.SIGKILL)
            logger.info("SIGKILL sent to pid=%s", self.pid)
        except (OSError, ProcessLookupError) as exc:
            logger.info(
                "SIGKILL skipped for pid=%s (exited between checks: %s)",
                self.pid, exc,
            )
            return

        # Step 4: brief settle window so the kernel can reap and the
        # GPU driver can release VRAM before the caller schedules a
        # replacement experiment on the same device.
        if self._wait_dead(self._KILL_GRACE_SEC):
            logger.info(
                "pid=%s reaped within %.1fs of SIGKILL",
                self.pid, self._KILL_GRACE_SEC,
            )
        else:
            logger.error(
                "pid=%s still observable after SIGKILL+%.1fs; releasing anyway",
                self.pid, self._KILL_GRACE_SEC,
            )

    def kill(self):
        try:
            os.kill(self.pid, signal.SIGKILL)
            logger.info("explicit SIGKILL sent to pid=%s", self.pid)
        except (OSError, ProcessLookupError):
            pass


class ExperimentRunner:
    def __init__(self, config: SystemConfig, db: ExperimentDB, gpu_manager: GPUManager):
        self.config = config
        self.db = db
        self.gpu_manager = gpu_manager

    # Valid argparse choices for train.py (baseline 6-action)
    VALID_CHOICES = {
        "model_type": ["baseline", "baseline_shared_head", "simple_baseline"],
        "severity_loss_fn": ["CE", "SORD", "MSE", "WK", "OLL", "CE+MSE", "CE+WK", "CE+OLL", "MSE+WK", "MSE+OLL"],
        "head_type": ["linear", "feature_fusion"],
    }

    def _safe_choice(self, cfg: dict, key: str, default: str) -> str:
        """Ensure config value is in valid argparse choices."""
        val = cfg.get(key, default)
        valid = self.VALID_CHOICES.get(key)
        if valid and val not in valid:
            print(f"  WARNING: config['{key}']={val} is not valid, using default '{default}'")
            return default
        return val

    def build_command(self, experiment: ExperimentRecord) -> List[str]:
        """Build training command for an experiment."""
        cfg = experiment.config
        exp_dir = experiment.experiment_dir
        code_dir = os.path.join(exp_dir, "code") if exp_dir else self.config.scripts_dir
        results_dir = os.path.join(exp_dir, "results") if exp_dir else self.config.results_base_dir

        cmd = [
            sys.executable, os.path.join(code_dir, "train.py"),
            "--data_dir", self.config.data_dir,
            "--target_actions", self.config.target_actions,
            "--exp", experiment.exp_name,
            "--output_dir", results_dir,
            "--epochs", str(self.config.epochs),
            "--no_wandb",
            "--seed", str(cfg.get("seed", 42)),
            "--model_type", self._safe_choice(cfg, "model_type", "baseline"),
            "--head_type", self._safe_choice(cfg, "head_type", "feature_fusion"),
            "--do_action_classification", str(cfg.get("do_action_classification", True)),
            "--severity_loss_fn", self._safe_choice(cfg, "severity_loss_fn", "MSE+WK"),
            "--batch_size", str(cfg.get("batch_size", 12)),
            "--lr", str(cfg.get("lr", 1e-4)),
            "--weight_decay", str(cfg.get("weight_decay", 0.0001)),
        ]

        return cmd

    def launch_experiment(self, experiment: ExperimentRecord, gpu_id: int) -> subprocess.Popen:
        """Launch training as subprocess on the given GPU using nohup for persistence."""
        exp_dir = experiment.experiment_dir
        assert_safe_path(exp_dir, "experiment_dir")

        code_dir = os.path.join(exp_dir, "code")
        log_path = os.path.join(exp_dir, "training.log")
        pid_path = os.path.join(exp_dir, "train.pid")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        cmd = self.build_command(experiment)
        cmd_str = " ".join(cmd)

        # Use nohup + setsid so process survives parent death
        # Write PID to file for tracking
        shell_cmd = (
            f"nohup {cmd_str} > {log_path} 2>&1 & "
            f"echo $! > {pid_path}"
        )
        proc = subprocess.Popen(
            shell_cmd,
            cwd=code_dir,
            env=env,
            shell=True,
            start_new_session=True,
        )
        proc.wait()  # Wait for shell to finish launching

        # Read actual training PID from file
        actual_pid = proc.pid
        if os.path.exists(pid_path):
            try:
                with open(pid_path) as f:
                    actual_pid = int(f.read().strip())
            except (ValueError, IOError):
                pass

        self.db.update_status(
            experiment.id, "running",
            gpu_id=gpu_id,
            pid=actual_pid,
            started_at=datetime.now().isoformat(),
        )
        self.gpu_manager.reserve_gpu(gpu_id, experiment.id)

        logger.info(
            "launched experiment id=%s name=%s gpu=%s pid=%s",
            experiment.id, experiment.exp_name, gpu_id, actual_pid,
        )

        # Heartbeat: tell the watchdog the runner is alive and just
        # successfully spawned a child. Failure here must never abort
        # the launch — touch_heartbeat already swallows OSError.
        touch_heartbeat("experiment_runner", _resolve_heartbeat_dir(self.config))

        # Return a lightweight object that tracks the PID
        return _DetachedProcess(actual_pid)

    def check_process_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def collect_results(self, experiment: ExperimentRecord) -> Optional[Dict[str, float]]:
        """Parse results after experiment completion or early stop.

        Checks for (in priority order):
        1. test_metrics.json — final metrics after full training
        2. best_test_metrics.json — best epoch metrics (available during training)
        3. training.log — fallback: parse best epoch metrics from log
        """
        exp_dir = experiment.experiment_dir
        if not exp_dir:
            return None

        results_dir = os.path.join(exp_dir, "results")

        # Try JSON files first (if results dir exists)
        if os.path.exists(results_dir):
            # Find the experiment output folder (may have exp_name subfolder)
            search_dir = results_dir
            exp_subdir = os.path.join(results_dir, experiment.exp_name)
            if os.path.exists(exp_subdir):
                search_dir = exp_subdir

            # Find timestamped subdirectory
            subdirs = []
            for name in os.listdir(search_dir):
                full = os.path.join(search_dir, name)
                if os.path.isdir(full) and name[0].isdigit():
                    subdirs.append(name)

            if not subdirs:
                for fname in ["test_metrics.json", "best_test_metrics.json"]:
                    metrics_path = os.path.join(results_dir, fname)
                    if os.path.exists(metrics_path):
                        return self._parse_metrics(experiment, metrics_path, results_dir)
            else:
                latest = sorted(subdirs)[-1]
                actual_results_dir = os.path.join(search_dir, latest)
                for fname in ["test_metrics.json", "best_test_metrics.json"]:
                    metrics_path = os.path.join(actual_results_dir, fname)
                    if os.path.exists(metrics_path):
                        return self._parse_metrics(experiment, metrics_path, actual_results_dir)

        # Fallback: extract best epoch results from training.log
        return self._collect_from_training_log(experiment)

    def _collect_from_training_log(self, experiment: ExperimentRecord) -> Optional[Dict[str, float]]:
        """Fallback: extract best epoch results from training.log when no JSON files exist."""
        curve = self.get_training_curve(experiment)
        if not curve:
            return None

        # Find best epoch (lowest avg_score_mae)
        best_entry = min(curve, key=lambda x: x[1])
        best_epoch, best_mae, per_action = best_entry[0], best_entry[1], best_entry[2]

        # Build metrics dict matching JSON format
        metrics = {}
        for action, mae_val in per_action.items():
            metrics[f"test_{action}_score_mae"] = mae_val
        metrics["best_epoch"] = best_epoch
        metrics["best_avg_score_mae"] = best_mae
        metrics["source"] = "training_log_fallback"

        avg_score_mae = best_mae
        results_dir = experiment.experiment_dir or ""

        self.db.update_results(
            experiment.id,
            avg_score_mae=avg_score_mae,
            per_action_results=metrics,
            results_dir=results_dir,
        )
        print(f"  [collect] Recovered results from training.log: avg_score_mae={avg_score_mae:.4f} (epoch {best_epoch})")
        return metrics

    def _parse_metrics(self, experiment: ExperimentRecord,
                       metrics_path: str, results_dir: str) -> Optional[Dict[str, float]]:
        with open(metrics_path) as f:
            metrics = json.load(f)

        # ------------------------------------------------------------------
        # 6-action integrity gate (2026-06-01)
        #
        # Before declaring an experiment 'completed', confirm the metrics
        # blob actually contains a valid score for every required action.
        # Historically a partial / corrupted test_metrics.json silently
        # produced a misleading avg_score_mae (sometimes from 1-2 actions
        # only), polluting ANNA's leaderboard and parent selection.
        #
        # Gate checks (any failure -> status='incomplete', avg=None):
        #   1. Every action in REQUIRED_ACTIONS has a *_score_mae key.
        #   2. Each value is a finite number with _MAE_MIN <= v <= _MAE_MAX
        #      (rejects NaN, inf, sentinel zeros, broken huge values).
        #   3. mean(6 per-action MAEs) matches the reported avg_score_mae
        #      within _AVG_RECONSTRUCT_TOL when an avg is present in the
        #      file. Mismatch usually means the file was written mid-run
        #      or the actions list drifted from train.py.
        # ------------------------------------------------------------------

        errors: List[str] = []
        per_action_mae: Dict[str, float] = {}

        for action in REQUIRED_ACTIONS:
            key = f"test_{action}_score_mae"
            if key not in metrics:
                errors.append(f"missing key {key}")
                continue
            val = metrics[key]
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                errors.append(f"{key} is not numeric (got {type(val).__name__})")
                continue
            fval = float(val)
            if not math.isfinite(fval):
                errors.append(f"{key} is non-finite ({fval})")
                continue
            if fval < _MAE_MIN or fval > _MAE_MAX:
                errors.append(
                    f"{key}={fval:.4f} outside [{_MAE_MIN}, {_MAE_MAX}]"
                )
                continue
            per_action_mae[action] = fval

        avg_score_mae: Optional[float] = None
        reconstructed_avg: Optional[float] = None

        if len(per_action_mae) == len(REQUIRED_ACTIONS):
            reconstructed_avg = sum(per_action_mae.values()) / len(REQUIRED_ACTIONS)

            # Cross-check against the avg field if the writer provided
            # one. We accept either 'avg_score_mae' or 'test_avg_score_mae'
            # because train.py variants have used both names.
            reported_avg = None
            for cand_key in ("avg_score_mae", "test_avg_score_mae"):
                v = metrics.get(cand_key)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                    reported_avg = float(v)
                    break

            if reported_avg is not None:
                drift = abs(reconstructed_avg - reported_avg)
                if drift >= _AVG_RECONSTRUCT_TOL:
                    errors.append(
                        f"reconstructed avg {reconstructed_avg:.4f} disagrees "
                        f"with reported {reported_avg:.4f} (drift={drift:.4f} "
                        f">= tol={_AVG_RECONSTRUCT_TOL})"
                    )

            if not errors:
                avg_score_mae = reconstructed_avg

        if errors:
            error_message = "6-action integrity gate failed: " + "; ".join(errors)
            logger.error(
                "[%s] %s (metrics_path=%s)",
                experiment.exp_name, error_message, metrics_path,
            )
            # Annotate the metrics blob with the failure so downstream
            # readers (dashboard, judge) can see why we rejected it.
            annotated = dict(metrics)
            annotated["_integrity_gate"] = {
                "status": "failed",
                "errors": errors,
                "metrics_path": metrics_path,
                "checked_at": datetime.now().isoformat(),
            }
            try:
                self.db.update_status(
                    experiment.id,
                    "incomplete",
                    avg_score_mae=None,
                    per_action_results_json=json.dumps(annotated, ensure_ascii=False),
                    results_dir=results_dir,
                    error_message=error_message,
                )
            except Exception as exc:
                # Older schemas may not have an error_message column;
                # retry without it so the status flip still lands.
                logger.warning(
                    "update_status(incomplete) full call failed for %s (%s); "
                    "retrying without error_message",
                    experiment.exp_name, exc,
                )
                try:
                    self.db.update_status(
                        experiment.id,
                        "incomplete",
                        avg_score_mae=None,
                        per_action_results_json=json.dumps(annotated, ensure_ascii=False),
                        results_dir=results_dir,
                    )
                except Exception as exc2:
                    logger.error(
                        "update_status(incomplete) fallback failed for %s: %s",
                        experiment.exp_name, exc2,
                    )
            return annotated

        # Gate passed — persist the validated result.
        if avg_score_mae is not None:
            try:
                self.db.update_results(
                    experiment.id,
                    avg_score_mae=avg_score_mae,
                    per_action_results=metrics,
                    results_dir=results_dir,
                )
                logger.info(
                    "[%s] integrity gate OK avg_score_mae=%.4f",
                    experiment.exp_name, avg_score_mae,
                )
            except Exception as e:
                logger.error(
                    "[collect] db.update_results failed for %s: %s",
                    experiment.exp_name, e,
                )
        return metrics

    def estimate_cost_hours(self, experiment: ExperimentRecord) -> float:
        base_minutes_per_epoch = 8.0
        batch_size = experiment.config.get("batch_size", 16)
        if batch_size == 32:
            base_minutes_per_epoch *= 0.6
        elif batch_size == 8:
            base_minutes_per_epoch *= 1.5

        # Use actual epochs for completed/early_stopped experiments
        if experiment.status in ("completed", "early_stopped", "failed"):
            curve = self.get_training_curve(experiment)
            actual_epochs = curve[-1][0] if curve else 0
            if actual_epochs > 0:
                return (base_minutes_per_epoch * actual_epochs) / 60.0
            # No training data — assume minimal cost
            return 0.1

        # For running/planned experiments, use full budget
        return (base_minutes_per_epoch * self.config.epochs) / 60.0

    def get_intermediate_best_mae(self, experiment: ExperimentRecord) -> Tuple[Optional[float], int, int]:
        """Parse training log for intermediate best avg_score_mae.

        Returns (best_avg_mae, best_epoch, current_epoch).
        Returns (None, 0, 0) if no results found.
        """
        curve = self.get_training_curve(experiment)
        if not curve:
            return None, 0, 0

        best_entry = min(curve, key=lambda x: x[1])
        best_epoch, best_mae = best_entry[0], best_entry[1]
        latest_epoch = curve[-1][0]
        return best_mae, best_epoch, latest_epoch

    def get_training_curve(self, experiment: ExperimentRecord) -> List[Tuple[int, float, Dict[str, float]]]:
        """Parse training log for full epoch-by-epoch training curve.

        Returns list of (epoch_num, avg_score_mae, {action: score_mae}).
        """
        if not experiment.experiment_dir:
            return []
        log_path = os.path.join(experiment.experiment_dir, "training.log")
        if not os.path.exists(log_path):
            return []

        actions = ["눈_살짝감기", "눈_질끈감기", "이마_주름", "입_이", "입_우", "안면_무표정"]
        current_epoch_maes = {}
        epoch_results = []
        current_epoch_num = 0

        try:
            with open(log_path) as f:
                for line in f:
                    m = re.search(r'Epoch \[(\d+)/\d+\]', line)
                    if m:
                        current_epoch_num = int(m.group(1))

                    for action in actions:
                        key = f"test_{action}_score_mae"
                        if key in line:
                            m2 = re.search(rf'{re.escape(key)}:\s*([\d.]+)', line)
                            if m2:
                                current_epoch_maes[action] = float(m2.group(1))

                    if len(current_epoch_maes) == len(actions):
                        avg = sum(current_epoch_maes.values()) / len(actions)
                        epoch_results.append((current_epoch_num, avg, dict(current_epoch_maes)))
                        current_epoch_maes = {}
        except Exception:
            return []

        return epoch_results

    def get_training_progress(self, experiment: ExperimentRecord) -> Optional[str]:
        if not experiment.experiment_dir:
            return None
        log_path = os.path.join(experiment.experiment_dir, "training.log")
        if not os.path.exists(log_path):
            return None

        last_epoch = None
        last_test = None
        try:
            with open(log_path) as f:
                for line in f:
                    if "Epoch [" in line and "Iter [" in line:
                        last_epoch = line.strip()
                    if "New best model" in line:
                        last_test = line.strip()
        except Exception:
            return None

        parts = []
        if last_epoch:
            parts.append(last_epoch[:80])
        if last_test:
            parts.append(last_test)
        return " | ".join(parts) if parts else None
