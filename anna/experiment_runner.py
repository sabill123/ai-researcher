"""
Experiment runner: launches training subprocess and collects results.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional

from .config import SystemConfig, assert_safe_path
from .experiment_db import ExperimentDB, ExperimentRecord
from .gpu_manager import GPUManager


class ExperimentRunner:
    def __init__(self, config: SystemConfig, db: ExperimentDB, gpu_manager: GPUManager):
        self.config = config
        self.db = db
        self.gpu_manager = gpu_manager

    # Valid argparse choices for train_cloc_v2.py
    VALID_CHOICES = {
        "model_type": ["separate", "shared"],
        "severity_loss_fn": ["CE", "MSE", "MSE+WK"],
        "cloc_type": ["cloc", "ordinal"],
        "margin_mode": ["single", "multi"],
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
            sys.executable, os.path.join(code_dir, "train_cloc_v2.py"),
            "--data_dir", self.config.data_dir,
            "--target_actions", self.config.target_actions,
            "--exp", experiment.exp_name,
            "--output_dir", results_dir,
            "--epochs", str(self.config.epochs),
            "--no_wandb",
            "--seed", str(cfg.get("seed", 42)),
            "--model_type", self._safe_choice(cfg, "model_type", "separate"),
            "--do_action_classification", str(cfg.get("do_action_classification", True)),
            "--severity_loss_fn", self._safe_choice(cfg, "severity_loss_fn", "MSE+WK"),
            "--use_cloc", str(cfg.get("use_cloc", True)),
            "--cloc_type", self._safe_choice(cfg, "cloc_type", "cloc"),
            "--cloc_weight", str(cfg.get("cloc_weight", 0.5)),
            "--cloc_temperature", str(cfg.get("cloc_temperature", 0.1)),
            "--margin_mode", self._safe_choice(cfg, "margin_mode", "multi"),
            "--learnable_margins", str(cfg.get("learnable_margins", True)),
            "--ordinal_alpha", str(cfg.get("ordinal_alpha", 1.0)),
            "--batch_size", str(cfg.get("batch_size", 16)),
            "--lr", str(cfg.get("lr", 1e-4)),
            "--weight_decay", str(cfg.get("weight_decay", 0.0001)),
        ]

        margins = cfg.get("initial_margins", [2.0, 1.0, 0.8, 0.6])
        cmd.append("--initial_margins")
        cmd.extend([str(m) for m in margins])

        return cmd

    def launch_experiment(self, experiment: ExperimentRecord, gpu_id: int) -> subprocess.Popen:
        """Launch training as subprocess on the given GPU."""
        exp_dir = experiment.experiment_dir
        assert_safe_path(exp_dir, "experiment_dir")

        code_dir = os.path.join(exp_dir, "code")
        log_path = os.path.join(exp_dir, "training.log")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"

        cmd = self.build_command(experiment)

        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            cwd=code_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        self.db.update_status(
            experiment.id, "running",
            gpu_id=gpu_id,
            pid=proc.pid,
            started_at=datetime.now().isoformat(),
        )
        self.gpu_manager.reserve_gpu(gpu_id, experiment.id)

        return proc

    def check_process_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def collect_results(self, experiment: ExperimentRecord) -> Optional[Dict[str, float]]:
        """Parse results after experiment completion."""
        exp_dir = experiment.experiment_dir
        if not exp_dir:
            return None

        results_dir = os.path.join(exp_dir, "results")
        if not os.path.exists(results_dir):
            return None

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
            # Maybe test_metrics.json is directly in results_dir
            metrics_path = os.path.join(results_dir, "test_metrics.json")
            if os.path.exists(metrics_path):
                return self._parse_metrics(experiment, metrics_path, results_dir)
            return None

        latest = sorted(subdirs)[-1]
        actual_results_dir = os.path.join(search_dir, latest)
        metrics_path = os.path.join(actual_results_dir, "test_metrics.json")

        if not os.path.exists(metrics_path):
            return None

        return self._parse_metrics(experiment, metrics_path, actual_results_dir)

    def _parse_metrics(self, experiment: ExperimentRecord,
                       metrics_path: str, results_dir: str) -> Optional[Dict[str, float]]:
        with open(metrics_path) as f:
            metrics = json.load(f)

        score_mae_values = [
            v for k, v in metrics.items() if k.endswith("_score_mae")
        ]
        avg_score_mae = sum(score_mae_values) / len(score_mae_values) if score_mae_values else None

        if avg_score_mae is not None:
            self.db.update_results(
                experiment.id,
                avg_score_mae=avg_score_mae,
                per_action_results=metrics,
                results_dir=results_dir,
            )

        return metrics

    def estimate_cost_hours(self, experiment: ExperimentRecord) -> float:
        base_minutes_per_epoch = 8.0
        batch_size = experiment.config.get("batch_size", 16)
        if batch_size == 32:
            base_minutes_per_epoch *= 0.6
        elif batch_size == 8:
            base_minutes_per_epoch *= 1.5
        return (base_minutes_per_epoch * self.config.epochs) / 60.0

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
