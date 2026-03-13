"""
Main orchestrator: 3-agent research loop.
Researcher → Engineer → Execute → Judge → Loop

Inspired by MARS (Modular Agent with Reflective Search) and AI-Scientist-v2.
"""

import json
import math
import os
import subprocess
import time
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


class LoopState:
    def __init__(self, iteration: int = 0, total_gpu_hours_used: float = 0.0,
                 total_claude_usd: float = 0.0, status: str = "idle"):
        self.iteration = iteration
        self.total_gpu_hours_used = total_gpu_hours_used
        self.total_claude_usd = total_claude_usd
        self.status = status

    def to_dict(self):
        return {
            "iteration": self.iteration,
            "total_gpu_hours_used": self.total_gpu_hours_used,
            "total_claude_usd": self.total_claude_usd,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            iteration=d.get("iteration", 0),
            total_gpu_hours_used=d.get("total_gpu_hours_used", 0.0),
            total_claude_usd=d.get("total_claude_usd", 0.0),
            status=d.get("status", "idle"),
        )


class Orchestrator:
    def __init__(self, config: SystemConfig):
        self.config = config
        assert_safe_path(config.project_root, "project_root")

        self.db = ExperimentDB(config.db_path)
        self.gpu_manager = GPUManager()
        self.runner = ExperimentRunner(config, self.db, self.gpu_manager)
        self.sandbox = ExperimentSandbox(config)
        self.memory = ReflectiveMemory(config.memory_path)
        self.tree = ExperimentTree()

        self.researcher = ResearcherAgent()
        self.engineer = EngineerAgent()
        self.judge = JudgeAgent()

        self.state = self._load_state()
        self._running_procs: Dict[str, subprocess.Popen] = {}
        self._research_context: Optional[Dict] = None
        self._judge_direction: Optional[Dict] = None

    def _load_state(self) -> LoopState:
        if os.path.exists(self.config.state_path):
            with open(self.config.state_path) as f:
                return LoopState.from_dict(json.load(f))
        return LoopState()

    def _save_state(self):
        os.makedirs(os.path.dirname(self.config.state_path), exist_ok=True)
        with open(self.config.state_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def initialize(self):
        """One-time setup: import existing results and bootstrap memory."""
        print("Importing existing experiments...")
        n = self.db.import_existing_results(self.config.results_base_dir)
        print(f"  Imported {n} experiments")

        print("Bootstrapping reflective memory...")
        self.memory.bootstrap()
        print(f"  {len(self.memory.insights)} insights loaded")

        best = self.db.get_best_experiment()
        if best:
            print(f"  Best result: {best.exp_name} = {best.avg_score_mae:.4f}")
        print(f"  Total experiments in DB: {self.db.get_total_count()}")

        self._cleanup_stale_runs()

    def _cleanup_stale_runs(self):
        running = self.db.get_running_experiments()
        for exp in running:
            if exp.pid and not self.runner.check_process_alive(exp.pid):
                print(f"  Cleaning stale experiment: {exp.exp_name} (pid={exp.pid})")
                results = self.runner.collect_results(exp)
                if results is None:
                    self.db.update_status(exp.id, "failed",
                                          completed_at=datetime.now().isoformat())
                if exp.gpu_id is not None:
                    self.gpu_manager.release_gpu(exp.gpu_id)

    def run(self, max_iterations: int = 50):
        """Main 3-agent research loop."""
        if self.state.iteration == 0 and self.db.get_total_count() == 0:
            self.initialize()

        for iteration in range(self.state.iteration, max_iterations):
            self.state.iteration = iteration
            self.state.status = "running"
            self._save_state()

            print(f"\n{'=' * 70}")
            print(f"ANNA v2 — ITERATION {iteration}")
            print(f"{'=' * 70}")

            # Budget check
            remaining_gpu = self._compute_remaining_gpu_budget()
            if remaining_gpu < 10.0:
                print("GPU budget exhausted. Stopping.")
                self.state.status = "budget_exhausted"
                self._save_state()
                return

            print(f"GPU budget remaining: {remaining_gpu:.1f} hours")

            best = self.db.get_best_experiment()
            current_best_mae = best.avg_score_mae if best else 999.0

            # ═══════════════════════════════════════════════
            # STEP 1: RESEARCHER (every N iterations or on judge request)
            # ═══════════════════════════════════════════════
            need_research = (
                iteration % self.config.researcher_interval == 0
                or (self._judge_direction and
                    self._judge_direction.get("next_direction", {}).get("researcher_guidance"))
            )

            if need_research:
                print(f"\n--- STEP 1: RESEARCHER ---")
                weak_actions = self._get_weak_actions(best)
                judge_guidance = None
                if self._judge_direction:
                    judge_guidance = self._judge_direction.get("next_direction", {}).get("researcher_guidance")

                self._research_context = self.researcher.research(
                    current_best_mae=current_best_mae,
                    weak_actions=weak_actions,
                    judge_guidance=judge_guidance,
                    existing_insights=self.memory.format_for_llm(),
                    completed_techniques=self.memory.get_completed_techniques(),
                )
                if self._research_context:
                    techniques = self._research_context.get("techniques", [])
                    print(f"  Found {len(techniques)} techniques:")
                    for t in techniques:
                        print(f"    - {t.get('name', 'unnamed')}: {t.get('core_idea', '')[:80]}")

            # ═══════════════════════════════════════════════
            # STEP 2: ENGINEER — Propose
            # ═══════════════════════════════════════════════
            print(f"\n--- STEP 2: ENGINEER (Propose) ---")
            top_experiments = self.db.format_experiments_for_prompt(self.db.get_top_k(15))
            engineer_guidance = None
            if self._judge_direction:
                engineer_guidance = self._judge_direction.get("next_direction", {}).get("engineer_guidance")

            proposals = self.engineer.propose(
                top_experiments=top_experiments,
                research_context=self._research_context,
                judge_guidance=engineer_guidance,
                existing_insights=self.memory.format_for_llm(),
                current_best_mae=current_best_mae,
                iteration=iteration,
                num_proposals=2,
            )

            if not proposals:
                print("  No valid proposals generated. Skipping iteration.")
                continue

            for p in proposals:
                print(f"  Proposed: {p.get('exp_name')} ({p.get('code_change_type', 'unknown')})")
                print(f"    Rationale: {p.get('rationale', '')[:100]}")

            # ═══════════════════════════════════════════════
            # STEP 3: ENGINEER — Implement + Validate + Launch
            # ═══════════════════════════════════════════════
            print(f"\n--- STEP 3: ENGINEER (Implement & Launch) ---")
            launched = []

            for proposal in proposals:
                exp_name = proposal.get("exp_name", f"exp_{iteration:03d}_unnamed")

                # Create sandbox
                exp_dir = self.sandbox.create_sandbox(exp_name)
                code_dir = os.path.join(exp_dir, "code")

                # Create experiment record
                record = ExperimentRecord(
                    exp_name=exp_name,
                    status="proposed",
                    priority=0.0,
                    expected_improvement=proposal.get("expected_improvement", 0.01),
                    confidence=proposal.get("confidence", 0.5),
                    config=proposal.get("config", {}),
                    experiment_dir=exp_dir,
                    proposed_by="engineer",
                    code_change_type=proposal.get("code_change_type", "hyperparameter"),
                    research_context=json.dumps(self._research_context) if self._research_context else None,
                    proposal_rationale=proposal.get("rationale", ""),
                    iteration=iteration,
                )

                # Implement code changes (if not pure hyperparameter)
                if proposal.get("code_change_type") != "hyperparameter" and proposal.get("changes"):
                    success = self.engineer.implement(proposal, code_dir)
                    if not success:
                        print(f"  Implementation failed for {exp_name}")
                        record.status = "failed"
                        self.db.add_experiment(record)
                        continue

                    # Restore read-only files
                    self.sandbox.restore_readonly_files(exp_dir)

                    # Validate sandbox
                    is_valid, errors = self.sandbox.validate_sandbox(exp_dir)
                    if not is_valid:
                        print(f"  Validation failed for {exp_name}:")
                        for err in errors:
                            print(f"    - {err}")
                        record.status = "failed"
                        record.proposal_rationale += f" | Validation errors: {'; '.join(errors)}"
                        self.db.add_experiment(record)
                        continue

                    # Save diff
                    diff = self.sandbox.save_diff(exp_dir)
                    record.code_diff = diff
                    record.status = "validated"
                else:
                    record.status = "validated"

                # Save proposal.json
                proposal_path = os.path.join(exp_dir, "proposal.json")
                with open(proposal_path, "w") as f:
                    json.dump(proposal, f, indent=2, ensure_ascii=False)

                # Score and add to DB
                cost = self.runner.estimate_cost_hours(record)
                ei = max(record.expected_improvement, 0.001)
                conf = max(record.confidence, 0.1)
                record.priority = (ei * conf) / math.sqrt(max(cost, 0.1))
                self.db.add_experiment(record)

                # Launch on GPU
                available_gpus = self.gpu_manager.get_available_gpus(
                    min_free_memory_mb=self.config.min_free_memory_mb,
                    max_utilization_pct=self.config.max_gpu_utilization_pct,
                )
                if not available_gpus:
                    print("  Waiting for GPU...")
                    gpu_id = self.gpu_manager.wait_for_gpu(
                        min_free_memory_mb=self.config.min_free_memory_mb,
                        max_utilization_pct=self.config.max_gpu_utilization_pct,
                        timeout_seconds=self.config.gpu_wait_timeout,
                        poll_interval=self.config.gpu_poll_interval,
                    )
                    if gpu_id is None:
                        print("  GPU timeout. Skipping remaining.")
                        break
                    available_gpus = [gpu_id]

                gpu_id = available_gpus[0]
                try:
                    proc = self.runner.launch_experiment(record, gpu_id)
                    self._running_procs[record.id] = proc
                    launched.append(record)
                    print(f"  Launched {exp_name} on GPU {gpu_id} (pid={proc.pid})")
                except Exception as e:
                    print(f"  Failed to launch {exp_name}: {e}")
                    self.db.update_status(record.id, "failed")

            if not launched:
                print("  No experiments launched this iteration.")
                continue

            # ═══════════════════════════════════════════════
            # STEP 4: WAIT for completion
            # ═══════════════════════════════════════════════
            print(f"\n--- STEP 4: WAITING ---")
            self._wait_for_completion(launched)

            # ═══════════════════════════════════════════════
            # STEP 5: COLLECT results
            # ═══════════════════════════════════════════════
            print(f"\n--- STEP 5: COLLECTING RESULTS ---")
            latest_results = []
            for exp in launched:
                exp = self.db.get_experiment(exp.id)
                if exp.status == "failed":
                    print(f"  {exp.exp_name}: FAILED")
                    continue

                results = self.runner.collect_results(exp)
                if results is None:
                    self.db.update_status(exp.id, "failed",
                                          completed_at=datetime.now().isoformat())
                    print(f"  {exp.exp_name}: FAILED (no results)")
                    continue

                exp = self.db.get_experiment(exp.id)
                print(f"  {exp.exp_name}: avg_score_mae = {exp.avg_score_mae:.4f}")

                if exp.per_action_results:
                    for k, v in sorted(exp.per_action_results.items()):
                        if "_score_mae" in k:
                            action = k.replace("test_", "").replace("_score_mae", "")
                            print(f"    {action}: {v:.4f}")

                # Record in tree
                improvement = (current_best_mae - (exp.avg_score_mae or 999))
                self.tree.record_experiment(
                    category=exp.code_change_type,
                    experiment_id=exp.id,
                    mae_improvement=improvement,
                    result_mae=exp.avg_score_mae or 999,
                )

                if exp.avg_score_mae and exp.avg_score_mae < current_best_mae:
                    print(f"  *** NEW BEST! Improved by {current_best_mae - exp.avg_score_mae:.4f} ***")

                latest_results.append(exp)

            # ═══════════════════════════════════════════════
            # STEP 6: JUDGE
            # ═══════════════════════════════════════════════
            print(f"\n--- STEP 6: JUDGE ---")
            latest_text = self.db.format_experiments_for_prompt(latest_results) if latest_results else "No results this iteration"
            top_text = self.db.format_experiments_for_prompt(self.db.get_top_k(15))

            updated_best = self.db.get_best_experiment()
            updated_best_mae = updated_best.avg_score_mae if updated_best else current_best_mae

            judge_result = self.judge.judge(
                completed_experiments=top_text,
                latest_experiments=latest_text,
                existing_insights=self.memory.format_for_llm(),
                research_context=self._research_context,
                current_best_mae=updated_best_mae,
                iteration=iteration,
            )

            if judge_result:
                # Print analysis
                analysis = judge_result.get("analysis", "")
                if analysis:
                    print(f"\n  Analysis: {analysis[:300]}...")

                # Add new insights
                new_insights = judge_result.get("new_insights", [])
                if new_insights:
                    self.memory.add_insights_from_judge(new_insights)
                    print(f"  Added {len(new_insights)} new insights")

                # Store direction for next iteration
                self._judge_direction = judge_result
                direction = judge_result.get("next_direction", {})
                print(f"  Strategy: {direction.get('strategy', 'unknown')}")
                if direction.get("engineer_guidance"):
                    print(f"  Engineer guidance: {direction['engineer_guidance'][:150]}")
                if direction.get("researcher_guidance"):
                    print(f"  Researcher guidance: {direction['researcher_guidance'][:150]}")

            # ═══════════════════════════════════════════════
            # STEP 7: CHECKPOINT
            # ═══════════════════════════════════════════════
            self.state.iteration = iteration + 1
            self._save_state()

            # Print summary
            print(f"\n{self.tree.get_direction_stats()}")
            best = self.db.get_best_experiment()
            if best:
                print(f"\nCurrent best: {best.exp_name} = {best.avg_score_mae:.4f}")

            # Human review checkpoint
            if (iteration + 1) % self.config.human_review_interval == 0:
                self.state.status = "awaiting_review"
                self._save_state()
                print(f"\n*** HUMAN REVIEW CHECKPOINT ***")
                print(f"Iteration {iteration + 1} completed.")
                print(f"Best: {best.avg_score_mae:.4f}" if best else "No results yet")
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

    def _wait_for_completion(self, experiments: List[ExperimentRecord]):
        pending = set(exp.id for exp in experiments if exp.id in self._running_procs)
        timeout = self.config.experiment_timeout_hours * 3600
        start_time = time.time()

        while pending:
            if time.time() - start_time > timeout:
                print("  Experiment timeout reached. Marking remaining as failed.")
                for exp_id in pending:
                    self.db.update_status(exp_id, "failed",
                                          completed_at=datetime.now().isoformat())
                    proc = self._running_procs.pop(exp_id, None)
                    if proc:
                        proc.terminate()
                    exp = self.db.get_experiment(exp_id)
                    if exp and exp.gpu_id is not None:
                        self.gpu_manager.release_gpu(exp.gpu_id)
                break

            done = set()
            for exp_id in pending:
                proc = self._running_procs.get(exp_id)
                if proc is None or proc.poll() is not None:
                    done.add(exp_id)
                    if proc and proc.returncode != 0:
                        self.db.update_status(exp_id, "failed",
                                              completed_at=datetime.now().isoformat())
                    exp = self.db.get_experiment(exp_id)
                    if exp and exp.gpu_id is not None:
                        self.gpu_manager.release_gpu(exp.gpu_id)
                    self._running_procs.pop(exp_id, None)

            pending -= done

            if pending:
                for exp_id in pending:
                    exp = self.db.get_experiment(exp_id)
                    progress = self.runner.get_training_progress(exp)
                    if progress:
                        print(f"  [{exp.exp_name}] {progress}")
                time.sleep(self.config.experiment_poll_interval)

    def _get_weak_actions(self, best_exp: Optional[ExperimentRecord]) -> List[str]:
        if not best_exp or not best_exp.per_action_results:
            return ["입_오", "입_이"]
        action_maes = {}
        for k, v in best_exp.per_action_results.items():
            if "_score_mae" in k:
                action = k.replace("test_", "").replace("_score_mae", "")
                action_maes[action] = v
        sorted_actions = sorted(action_maes.items(), key=lambda x: x[1], reverse=True)
        return [a for a, _ in sorted_actions[:3]]

    def _compute_remaining_gpu_budget(self) -> float:
        all_completed = self.db.get_all_completed()
        running = self.db.get_running_experiments()
        used = 0.0
        for exp in all_completed + running:
            if exp.proposed_by == "bootstrap":
                continue
            used += self.runner.estimate_cost_hours(exp)
        return max(0, self.config.budget_gpu_hours - used)

    def resume(self):
        state = self._load_state()
        if state.status == "awaiting_review":
            print(f"Resuming from iteration {state.iteration}")
            self.state = state
            self.state.status = "running"
            self._save_state()
            self._cleanup_stale_runs()
            self.run(max_iterations=50)
        elif state.status == "idle":
            self.run(max_iterations=50)
        else:
            print(f"Cannot resume: status is '{state.status}'")

    def show_status(self):
        state = self._load_state()
        best = self.db.get_best_experiment()
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

        # Tree stats
        print(f"\n{self.tree.get_direction_stats()}")

        if running:
            print(f"\nRunning ({len(running)}):")
            for exp in running:
                progress = self.runner.get_training_progress(exp)
                print(f"  {exp.exp_name} on GPU {exp.gpu_id}: {progress or 'no progress info'}")

        print(f"\n{self.gpu_manager.format_status()}")

    def show_report(self):
        """Generate comprehensive report."""
        all_completed = self.db.get_all_completed()
        best = self.db.get_best_experiment()
        total = self.db.get_total_count()

        print("=" * 70)
        print("ANNA v2 RESEARCH REPORT")
        print("=" * 70)
        print(f"Total experiments: {total}")
        print(f"Completed: {len(all_completed)}")
        print(f"Best Avg Score MAE: {best.avg_score_mae:.4f}" if best else "No completed experiments")
        print(f"Target: {self.config.target_avg_score_mae}")
        print()

        # Top 10
        top10 = self.db.get_top_k(10)
        if top10:
            print("--- Top 10 Experiments ---")
            for i, exp in enumerate(top10, 1):
                print(f"  {i}. {exp.exp_name}: {exp.avg_score_mae:.4f} "
                      f"[{exp.code_change_type}]")
                if exp.per_action_results:
                    for k, v in sorted(exp.per_action_results.items()):
                        if "_score_mae" in k:
                            action = k.replace("test_", "").replace("_score_mae", "")
                            print(f"       {action}: {v:.4f}")
            print()

        # By change type
        for ct in ["architecture", "loss", "augmentation", "training", "hyperparameter"]:
            exps = self.db.get_experiments_by_change_type(ct)
            if exps:
                best_ct = exps[0]
                print(f"  Best {ct}: {best_ct.exp_name} = {best_ct.avg_score_mae:.4f} ({len(exps)} total)")

        # Tree stats
        print(f"\n{self.tree.get_direction_stats()}")

        # Insights
        print(f"\n--- Research Insights ({len(self.memory.insights)}) ---")
        for insight in self.memory.get_insights(min_confidence=0.5):
            print(f"  [{insight.confidence:.2f}] {insight.observation}")
            print(f"         -> {insight.recommendation}")

        print("=" * 70)
