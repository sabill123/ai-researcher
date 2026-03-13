"""
System configuration for ANNA v2.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class SystemConfig:
    project_root: str = "/home/jaeseokhan/2025-02/khu/research-2026-01"

    @property
    def scripts_dir(self) -> str:
        return os.path.join(self.project_root, "scripts")

    @property
    def results_base_dir(self) -> str:
        return os.path.join(self.scripts_dir, "results_cloc_v2")

    @property
    def anna_v2_dir(self) -> str:
        return os.path.join(self.project_root, "anna_v2")

    @property
    def experiments_dir(self) -> str:
        return os.path.join(self.project_root, "experiments")

    @property
    def db_path(self) -> str:
        return os.path.join(self.anna_v2_dir, "data", "experiments.db")

    @property
    def memory_path(self) -> str:
        return os.path.join(self.anna_v2_dir, "data", "reflective_memory.json")

    @property
    def state_path(self) -> str:
        return os.path.join(self.anna_v2_dir, "data", "loop_state.json")

    @property
    def pretrained_ckpts_dir(self) -> str:
        return os.path.join(self.project_root, "pretrained_ckpts")

    data_dir: str = "/home/jaeseokhan/2025-02/khu/research-2026-01/integrated_dataset"
    target_actions: str = "target"

    # Training
    epochs: int = 100
    max_parallel: int = 2

    # GPU management
    min_free_memory_mb: int = 20000
    max_gpu_utilization_pct: int = 10
    gpu_poll_interval: int = 60
    gpu_wait_timeout: int = 1800

    # Budget
    budget_gpu_hours: float = 300.0
    budget_claude_usd: float = 50.0

    # Experiment timeouts
    experiment_timeout_hours: float = 4.0
    experiment_poll_interval: int = 60

    # Agent settings
    researcher_interval: int = 3  # Run researcher every N iterations
    human_review_interval: int = 5

    # Target
    target_avg_score_mae: float = 0.49

    # Files to copy into experiment sandbox
    sandbox_files: List[str] = field(default_factory=lambda: [
        "train_cloc_v2.py",
        "model.py",
        "losses.py",
        "cloc_loss.py",
        "dataset.py",
        "backbone.py",
        "utils.py",
        "__init__.py",
    ])

    # Files the engineer is NOT allowed to modify (read-only in sandbox)
    readonly_files: List[str] = field(default_factory=lambda: [
        "utils.py",  # Metric computation — must not change
    ])


ALLOWED_BASE_PATH = "/home/jaeseokhan/2025-02/khu"


def assert_safe_path(path: str, label: str = "path"):
    """Verify a path is within the allowed project directory."""
    real = os.path.realpath(path)
    if not real.startswith(ALLOWED_BASE_PATH):
        raise RuntimeError(
            f"SAFETY VIOLATION: {label} '{real}' is outside "
            f"allowed path '{ALLOWED_BASE_PATH}'. Operation blocked."
        )
