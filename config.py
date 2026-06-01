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
        return os.path.join(self.project_root, "baseline_6action")

    @property
    def results_base_dir(self) -> str:
        return os.path.join(self.scripts_dir, "results")

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
    target_actions: tuple = (
        '눈_살짝감기',
        '눈_질끈감기',
        '이마_주름',
        '입_이',
        '입_우',
        '안면_무표정',
    )

    # Training
    epochs: int = 100
    max_parallel: int = 4  # 4× parallel — 8 A5000 GPUs, ~5 free after housemate's usage
    # GPUs that ANNA must NEVER schedule on. GPU 1 = housemate, GPU 7 = the
    # production self-monitoring app's inference worker.
    excluded_gpus: tuple = (1, 7)

    # GPU management
    min_free_memory_mb: int = 16000  # A5000=24GB, 실험당 ~12GB 사용
    max_gpu_utilization_pct: int = 30  # 학습 중 GPU는 당연히 사용됨
    gpu_poll_interval: int = 60
    gpu_wait_timeout: int = 1800

    # Budget
    budget_gpu_hours: float = 5000.0
    budget_claude_usd: float = 500.0

    # Experiment timeouts
    experiment_timeout_hours: float = 18.0  # 실측 ~9.4 min/epoch × 100 = ~15.7h, +margin (이전 15h는 iter 354에서 4/4 timeout으로 fail 유발)
    experiment_poll_interval: int = 60

    # AI-based early stopping — Claude analyzes training curve and decides
    early_stop_min_epochs: int = 30        # 30 epoch 이전에는 early stop 금지

    # Agent settings
    researcher_interval: int = 3  # Run researcher every N iterations
    human_review_interval: int = 10000  # effectively disabled — let ANNA run autonomously

    # Target
    target_avg_score_mae: float = 0.49

    # Sanity check ranges for per-action MAE values
    per_action_sanity_range: tuple = (0.05, 5.0)
    # Tolerance when reconstructing avg MAE from per-action MAEs
    reconstructed_avg_tol: float = 0.01

    # Agent invocation timeouts (seconds)
    timeout_researcher_sec: int = 900
    timeout_engineer_propose_sec: int = 1800
    timeout_engineer_implement_sec: int = 1200
    timeout_judge_sec: int = 600
    # Per-experiment training timeout (seconds) — mirrors experiment_timeout_hours
    timeout_experiment_sec: int = 18 * 3600

    # Watchdog thresholds
    watchdog_crash_loop_threshold: int = 5
    watchdog_giveup_24h: int = 20

    # Reflective memory / retrospective settings
    reflective_insight_grace_min: int = 30
    retrospective_cap: int = 200

    # Files to copy into experiment sandbox
    sandbox_files: List[str] = field(default_factory=lambda: [
        "train.py",
        "model.py",
        "losses.py",
        "dataset.py",
        "backbone.py",
        "utils.py",
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
