"""
Main orchestrator: 3-agent research loop.
Researcher → Engineer → Execute → Judge → Loop

Inspired by MARS (Modular Agent with Reflective Search) and AI-Scientist-v2.

Module #8 refactor (2026-06-01):
    The 1266-LOC god-function that previously lived here has been split into
    one module per phase under ``anna_v2/loop/``. Orchestrator now owns only:

        * Composition root (DB, GPU mgr, memory, agents, runner, sandbox, tree).
        * Outer ``run()`` while-loop driving ``IterationRunner.run_one(ctx)``.
        * Bootstrapping (``initialize``), restart recovery
          (``_resume_running_experiments``), and one-time VLM baseline.
        * Status/report dump-only entry points used by the CLI.
        * Process-level safety: single-instance ``flock`` and SIGTERM handler.

    All pure-read query helpers (best/top-k/parent/weak-actions/stagnation
    diagnosis/VLM context) moved to ``loop/helpers.py``. Orchestrator
    re-exposes them as bound methods so cli.py and the watchdog keep working
    unchanged.
"""

from __future__ import annotations

import fcntl
import os
import signal
from datetime import datetime
from typing import Dict, List, Optional

from .config import SystemConfig, assert_safe_path
from .experiment_db import ExperimentDB, ExperimentRecord
from .experiment_runner import ExperimentRunner
from .experiment_sandbox import ExperimentSandbox
from .experiment_tree import ExperimentTree
from .gpu_manager import GPUManager
from .reflective_memory import ReflectiveMemory
from .agents.researcher import ResearcherAgent
from .agents.engineer import EngineerAgent
from .agents.judge import JudgeAgent
from .loop import (
    IterationContext,
    IterationRunner,
    LoopState,
    load_state,
    save_state,
)
from .loop import helpers as _helpers
from .loop.steps.judge import run_judge as _run_judge_step
from .loop.steps.wait import wait_for_completion as _wait_for_completion

# Re-export LoopState so external callers using ``from anna_v2.orchestrator
# import LoopState`` keep working after the module #8 move.
__all__ = ["Orchestrator", "LoopState"]


class Orchestrator:
    def __init__(self, config: SystemConfig):
        self.config = config
        assert_safe_path(config.project_root, "project_root")

        self.db = ExperimentDB(config.db_path)
        self.gpu_manager = GPUManager(
            excluded_gpus=getattr(config, "excluded_gpus", ()),
        )
        self.runner = ExperimentRunner(config, self.db, self.gpu_manager)
        self.sandbox = ExperimentSandbox(config)
        self.memory = ReflectiveMemory(config.memory_path)
        self.tree = ExperimentTree()

        self.researcher = ResearcherAgent()
        self.engineer = EngineerAgent()
        self.judge = JudgeAgent()

        self.state = load_state(self.config.state_path)
        self._running_procs: Dict[str, object] = {}
        self._research_context: Optional[Dict] = None
        self._judge_direction: Optional[Dict] = None
        self._lock_fd: Optional[int] = None

    # ------------------------------------------------------------------ state
    def _save_state(self) -> None:
        save_state(self.config.state_path, self.state)

    # ------------------------------------------------- query-helper shims
    # Thin wrappers so step modules (via ctx) and cli/watchdog (via
    # Orchestrator) share a single implementation in loop/helpers.py.
    def _get_best_valid_experiment(self) -> Optional[ExperimentRecord]:
        return _helpers.get_best_valid_experiment(self.db, self.config)

    def _get_top_valid_k(self, k: int) -> List[ExperimentRecord]:
        return _helpers.get_top_valid_k(self.db, self.config, k)

    def _get_parent_experiment(self) -> Optional[ExperimentRecord]:
        return _helpers.get_parent_experiment(self.db, self.config)

    def _get_weak_actions(
        self, best_exp: Optional[ExperimentRecord]
    ) -> List[str]:
        return _helpers.get_weak_actions(self.db, best_exp)

    def _get_vlm_baseline_context(self) -> Optional[str]:
        return _helpers.get_vlm_baseline_context(self.db)

    def _compute_stagnation_diagnosis(self) -> str:
        return _helpers.compute_stagnation_diagnosis(self.db, self.config)

    # ----------------------------------------------------------- single-instance
    def _acquire_lock(self) -> bool:
        """Acquire ``orchestrator.lock`` via ``fcntl.flock``; False if held.

        Held for the process lifetime. The lock file lives next to
        ``loop_state.json`` for triage visibility. A held lock means
        another anna_v2 process is already driving the loop — refuse to
        start so we don't double-launch experiments onto the same GPU.
        """
        lock_path = os.path.join(
            os.path.dirname(self.config.state_path), "orchestrator.lock"
        )
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        self._lock_fd = fd
        return True

    def _install_signal_handler(self) -> None:
        """Persist ``state.status='sigterm_shutdown'`` then re-raise default.

        Does NOT kill running experiments — they keep training and are
        re-picked-up by ``_resume_running_experiments`` on next start.
        """
        def _handler(signum, _frame):
            try:
                self.state.status = "sigterm_shutdown"
                self._save_state()
            except Exception:
                # Never raise from a signal handler.
                pass
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    # --------------------------------------------------------------- init
    def initialize(self):
        """One-time setup: import existing results and bootstrap memory."""
        print("Importing existing experiments...")
        n = self.db.import_existing_results(self.config.results_base_dir)
        print(f"  Imported {n} experiments")

        print("Bootstrapping reflective memory...")
        self.memory.bootstrap()
        self.memory.prune(max_insights=30)
        print(f"  {len(self.memory.insights)} insights loaded")

        best = self._get_best_valid_experiment()
        if best:
            print(f"  Best result: {best.exp_name} = {best.avg_score_mae:.4f}")
        print(f"  Total experiments in DB: {self.db.get_total_count()}")

        self._cleanup_stale_runs()

    def _ensure_vlm_baseline(self):
        """Run VLM baseline evaluation if not already done."""
        try:
            from .vlm_evaluator import run_vlm_baseline
            print("\n--- VLM BASELINE CHECK ---")
            result = run_vlm_baseline(
                config=self.config,
                db=self.db,
                iteration=self.state.iteration,
                samples_per_action=30,
            )
            if result and result.avg_score_mae:
                print(f"  VLM baseline MAE: {result.avg_score_mae:.4f}")
        except Exception as e:
            print(f"  VLM baseline skipped: {e}")

    def _resume_running_experiments(self):
        """On restart, pick up experiments still training from prior session.

        Two sub-cases:
          1. Process alive → re-attach via ``_DetachedProcess``, then wait
             with the standard early-stop + zombie detection loop.
          2. Process dead → try one last result collection, else mark failed.

        Once any survivors finish we still run the Judge on them so the
        memory captures the resume-window outcomes before the main loop
        starts a fresh iteration.
        """
        from .experiment_runner import _DetachedProcess

        running = self.db.get_running_experiments()
        if not running:
            return

        alive = []
        for exp in running:
            if exp.pid and self.runner.check_process_alive(exp.pid):
                alive.append(exp)
                self._running_procs[exp.id] = _DetachedProcess(exp.pid)
                print(
                    f"  Resuming tracking: {exp.exp_name} "
                    f"(pid={exp.pid}, gpu={exp.gpu_id})"
                )
            else:
                print(
                    f"  Cleaning stale experiment: {exp.exp_name} "
                    f"(pid={exp.pid})"
                )
                results = self.runner.collect_results(exp)
                if results is None:
                    self.db.update_status(
                        exp.id, "failed",
                        completed_at=datetime.now().isoformat(),
                    )
                if exp.gpu_id is not None:
                    self.gpu_manager.release_gpu(exp.gpu_id)

        if alive:
            print(
                f"\n  Found {len(alive)} running experiments — "
                f"waiting with early stopping..."
            )
            ctx = self._build_iteration_context()
            _wait_for_completion(ctx, experiments=alive)

            best_exp = self._get_best_valid_experiment()
            current_best_mae = best_exp.avg_score_mae if best_exp else 999.0

            for exp in alive:
                exp = self.db.get_experiment(exp.id)
                if exp.status == "running":
                    results = self.runner.collect_results(exp)
                    if results is None:
                        self.db.update_status(
                            exp.id, "failed",
                            completed_at=datetime.now().isoformat(),
                        )
                    else:
                        exp = self.db.get_experiment(exp.id)
                        print(
                            f"  {exp.exp_name}: avg_score_mae = "
                            f"{exp.avg_score_mae:.4f}"
                        )
                        if (exp.avg_score_mae
                                and exp.avg_score_mae < current_best_mae):
                            print(f"  *** NEW BEST! ***")

            print(f"\n--- JUDGE (for resumed experiments) ---")
            completed_resumed = [self.db.get_experiment(e.id) for e in alive]
            completed_resumed = [
                e for e in completed_resumed if e and e.avg_score_mae
            ]
            if completed_resumed:
                _run_judge_step(
                    ctx,
                    latest_results=completed_resumed,
                    current_best_mae=current_best_mae,
                )
                self._judge_direction = ctx.judge_direction

    def _cleanup_stale_runs(self):
        """Mark dead-but-still-running DB rows as failed and free their GPUs."""
        running = self.db.get_running_experiments()
        for exp in running:
            if exp.pid and not self.runner.check_process_alive(exp.pid):
                print(
                    f"  Cleaning stale experiment: {exp.exp_name} "
                    f"(pid={exp.pid})"
                )
                results = self.runner.collect_results(exp)
                if results is None:
                    self.db.update_status(
                        exp.id, "failed",
                        completed_at=datetime.now().isoformat(),
                    )
                if exp.gpu_id is not None:
                    self.gpu_manager.release_gpu(exp.gpu_id)

    # --------------------------------------------------------------- run loop
    def _build_iteration_context(self) -> IterationContext:
        """Build a per-iteration handle bundle passed to step modules."""
        return IterationContext(
            config=self.config,
            db=self.db,
            gpu_manager=self.gpu_manager,
            runner=self.runner,
            sandbox=self.sandbox,
            memory=self.memory,
            tree=self.tree,
            researcher=self.researcher,
            engineer=self.engineer,
            judge=self.judge,
            state=self.state,
            iteration=self.state.iteration,
            research_context=self._research_context,
            judge_direction=self._judge_direction,
            running_procs=self._running_procs,
            get_best_valid_experiment=self._get_best_valid_experiment,
            get_top_valid_k=self._get_top_valid_k,
            get_parent_experiment=self._get_parent_experiment,
            get_weak_actions=self._get_weak_actions,
            get_vlm_baseline_context=self._get_vlm_baseline_context,
            compute_stagnation_diagnosis=self._compute_stagnation_diagnosis,
            save_state=self._save_state,
        )

    def run(self, max_iterations: int = 50):
        """Main 3-agent research loop."""
        if not self._acquire_lock():
            print(
                "Another anna_v2 process is already running "
                "(orchestrator.lock held). Refusing to start a second "
                "instance — see _acquire_lock docstring."
            )
            return
        self._install_signal_handler()

        if self.state.iteration == 0 and self.db.get_total_count() == 0:
            self.initialize()

        self._resume_running_experiments()
        self._ensure_vlm_baseline()

        runner = IterationRunner()
        for iteration in range(self.state.iteration, max_iterations):
            self.state.iteration = iteration
            self.state.status = "running"
            self._save_state()

            print(f"\n{'=' * 70}")
            print(f"ANNA v2 — ITERATION {iteration}")
            print(f"{'=' * 70}")

            remaining_gpu = self._compute_remaining_gpu_budget()
            if remaining_gpu < 10.0:
                print("GPU budget exhausted. Stopping.")
                self.state.status = "budget_exhausted"
                self._save_state()
                return

            print(f"GPU budget remaining: {remaining_gpu:.1f} hours")

            ctx = self._build_iteration_context()
            runner.run_one(ctx)
            # Pull cross-iteration carry-overs back onto self.
            self._research_context = ctx.research_context
            self._judge_direction = ctx.judge_direction

            best = self._get_best_valid_experiment()

            # Human review checkpoint
            if (iteration + 1) % self.config.human_review_interval == 0:
                self.state.status = "awaiting_review"
                self._save_state()
                print(f"\n*** HUMAN REVIEW CHECKPOINT ***")
                print(f"Iteration {iteration + 1} completed.")
                print(
                    f"Best: {best.avg_score_mae:.4f}"
                    if best else "No results yet"
                )
                print("Run 'python -m anna_v2 resume' to continue.")
                return

            # Target check
            if best and best.avg_score_mae <= self.config.target_avg_score_mae:
                print(f"\n*** TARGET REACHED: {best.avg_score_mae:.4f} ***")
                self.state.status = "target_reached"
                self._save_state()
                return

        self.state.status = "max_iterations_reached"
        self._save_state()

    def _compute_remaining_gpu_budget(self) -> float:
        all_completed = self.db.get_all_completed()
        running = self.db.get_running_experiments()
        used = 0.0
        for exp in all_completed + running:
            if exp.proposed_by == "bootstrap":
                continue
            used += self.runner.estimate_cost_hours(exp)
        return max(0, self.config.budget_gpu_hours - used)

    # ---------------------------------------------------- CLI entry points
    def resume(self):
        state = load_state(self.config.state_path)
        if state.status == "awaiting_review":
            print(f"Resuming from iteration {state.iteration}")
            self.state = state
            self.state.status = "running"
            self._save_state()
            self._cleanup_stale_runs()
            self.run(max_iterations=5000)
        elif state.status == "idle":
            self.run(max_iterations=5000)
        else:
            print(f"Cannot resume: status is '{state.status}'")

    def show_status(self):
        state = load_state(self.config.state_path)
        best = self._get_best_valid_experiment()
        running = self.db.get_running_experiments()
        total = self.db.get_total_count()
        completed = len(self.db.get_all_completed())
        remaining = self._compute_remaining_gpu_budget()

        print(f"Status: {state.status}")
        print(f"Iteration: {state.iteration}")
        print(f"Experiments: {completed}/{total} completed")
        print(f"GPU budget remaining: {remaining:.1f} hours")
        if best:
            print(f"Best: {best.exp_name} = {best.avg_score_mae:.4f}")
        print(f"Target: {self.config.target_avg_score_mae}")
        print(f"Insights: {len(self.memory.insights)}")

        print(f"\n{self.tree.get_direction_stats()}")

        if running:
            print(f"\nRunning ({len(running)}):")
            for exp in running:
                progress = self.runner.get_training_progress(exp)
                print(
                    f"  {exp.exp_name} on GPU {exp.gpu_id}: "
                    f"{progress or 'no progress info'}"
                )

        print(f"\n{self.gpu_manager.format_status()}")

    def show_report(self):
        """Generate comprehensive report."""
        all_completed = self.db.get_all_completed()
        best = self._get_best_valid_experiment()
        total = self.db.get_total_count()

        print("=" * 70)
        print("ANNA v2 RESEARCH REPORT")
        print("=" * 70)
        print(f"Total experiments: {total}")
        print(f"Completed: {len(all_completed)}")
        print(
            f"Best Avg Score MAE: {best.avg_score_mae:.4f}"
            if best else "No completed experiments"
        )
        print(f"Target: {self.config.target_avg_score_mae}")
        print()

        top10 = self.db.get_top_k(10)
        if top10:
            print("--- Top 10 Experiments ---")
            for i, exp in enumerate(top10, 1):
                print(
                    f"  {i}. {exp.exp_name}: {exp.avg_score_mae:.4f} "
                    f"[{exp.code_change_type}]"
                )
                if exp.per_action_results:
                    for k, v in sorted(exp.per_action_results.items()):
                        if k.startswith("test_") and k.endswith("_score_mae"):
                            action = k[len("test_"):-len("_score_mae")]
                            print(f"       {action}: {v:.4f}")
            print()

        for ct in [
            "architecture", "loss", "augmentation",
            "training", "hyperparameter",
        ]:
            exps = self.db.get_experiments_by_change_type(ct)
            if exps:
                best_ct = exps[0]
                print(
                    f"  Best {ct}: {best_ct.exp_name} = "
                    f"{best_ct.avg_score_mae:.4f} ({len(exps)} total)"
                )

        print(f"\n{self.tree.get_direction_stats()}")

        print(f"\n--- Research Insights ({len(self.memory.insights)}) ---")
        for insight in self.memory.get_insights(min_confidence=0.5):
            print(f"  [{insight.confidence:.2f}] {insight.observation}")
            print(f"         -> {insight.recommendation}")

        print("=" * 70)
