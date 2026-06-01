"""IterationRunner: one pass of the 3-agent loop.

The orchestrator owns long-lived state (DB, GPU manager, reflective memory,
agent handles); each iteration is delegated here. We deliberately do NOT
hold any state inside ``IterationRunner`` — every per-iteration value lives
on the ``IterationContext`` dataclass that is rebuilt each call. This lets a
future test or replay tool drive a single iteration in isolation.

Behavioral contract (module #8 refactor, 2026-06-01):
    The body of ``run_one()`` is a verbatim sequence of the old
    ``Orchestrator.run()`` per-iteration block — Researcher → Engineer
    propose → Engineer implement/launch → Wait → Collect → Judge →
    Checkpoint. Step modules under ``loop.steps`` were each cut out of the
    same method; their return contracts mirror what the old code stored on
    ``self`` (``_research_context``, ``_judge_direction``, etc.). Nothing
    here adds or removes I/O — the only difference is that step bodies are
    importable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .state import LoopState

if TYPE_CHECKING:
    from ..config import SystemConfig
    from ..experiment_db import ExperimentDB, ExperimentRecord
    from ..experiment_runner import ExperimentRunner
    from ..experiment_sandbox import ExperimentSandbox
    from ..experiment_tree import ExperimentTree
    from ..gpu_manager import GPUManager
    from ..reflective_memory import ReflectiveMemory
    from ..agents.researcher import ResearcherAgent
    from ..agents.engineer import EngineerAgent
    from ..agents.judge import JudgeAgent


@dataclass
class IterationContext:
    """Per-iteration handle bundle.

    Carries everything a step function needs to do its job. We pass this
    instead of ``self`` so step modules don't have to import Orchestrator
    (which would create a cycle: loop → orchestrator → loop).

    Cross-iteration carry-over state (``research_context``,
    ``judge_direction``, ``running_procs``) is owned by the orchestrator;
    we expose it through the context as a mutable handle so step modules
    can update it without each owning their own copy.
    """

    config: "SystemConfig"
    db: "ExperimentDB"
    gpu_manager: "GPUManager"
    runner: "ExperimentRunner"
    sandbox: "ExperimentSandbox"
    memory: "ReflectiveMemory"
    tree: "ExperimentTree"
    researcher: "ResearcherAgent"
    engineer: "EngineerAgent"
    judge: "JudgeAgent"

    state: LoopState
    iteration: int

    # Mutable carry-overs (orchestrator owns the underlying objects; the
    # context just exposes them so steps can read/write).
    research_context: Optional[Dict[str, Any]] = None
    judge_direction: Optional[Dict[str, Any]] = None
    running_procs: Dict[str, Any] = field(default_factory=dict)

    # Orchestrator helpers — bound methods passed in to avoid coupling
    # step modules to the Orchestrator class. Each is one of the small
    # query helpers that survived on the Orchestrator (best/top-k/etc.).
    get_best_valid_experiment: Any = None  # () -> Optional[ExperimentRecord]
    get_top_valid_k: Any = None            # (int) -> List[ExperimentRecord]
    get_parent_experiment: Any = None      # () -> Optional[ExperimentRecord]
    get_weak_actions: Any = None           # (Optional[ExperimentRecord]) -> List[str]
    get_vlm_baseline_context: Any = None   # () -> Optional[str]
    compute_stagnation_diagnosis: Any = None  # () -> str
    save_state: Any = None                 # () -> None


class IterationRunner:
    """Execute one Researcher → … → Judge pass.

    Stateless: every call to ``run_one`` operates only on the supplied
    ``ctx``. The split into a class (vs. a free function) is purely for
    parity with how the rest of the package is organized.
    """

    def run_one(self, ctx: IterationContext) -> bool:
        """Run one iteration. Returns False if the caller should `continue`.

        The boolean is the same signal the old in-line code used: if no
        proposals could be generated or no experiments launched, the
        outer while-loop should advance without trying to wait/collect.
        ``True`` means a normal iteration completed (results collected,
        Judge run, state advanced).
        """
        # Local imports to keep the module's top-level import cheap and to
        # avoid the (already-tight) import order in the package's
        # ``__init__``.
        from .steps import (
            research as _research,
            propose as _propose,
            implement as _implement,
            wait as _wait,
            collect as _collect,
            judge as _judge,
            checkpoint as _checkpoint,
        )

        best = ctx.get_best_valid_experiment()
        current_best_mae = best.avg_score_mae if best else 999.0

        # STEP 1: Researcher (conditional)
        _research.run_researcher(ctx, best=best, current_best_mae=current_best_mae)

        # STEP 2: Engineer — Propose
        proposals, parent_info, parent_exp, parent_code_dir = _propose.run_engineer_propose(
            ctx,
            current_best_mae=current_best_mae,
        )
        if not proposals:
            print("  No valid proposals generated. Skipping iteration.")
            return False

        # STEP 3: Engineer — Implement + Validate + Launch
        launched: List["ExperimentRecord"] = _implement.run_engineer_implement(
            ctx,
            proposals=proposals,
            parent_exp=parent_exp,
            parent_code_dir=parent_code_dir,
        )
        if not launched:
            print("  No experiments launched this iteration.")
            return False

        # STEP 4: Wait
        print(f"\n--- STEP 4: WAITING ---")
        _wait.wait_for_completion(ctx, experiments=launched)

        # STEP 5: Collect
        print(f"\n--- STEP 5: COLLECTING RESULTS ---")
        latest_results = _collect.collect_results(
            ctx, launched=launched, current_best_mae=current_best_mae,
        )

        # STEP 6: Judge
        print(f"\n--- STEP 6: JUDGE ---")
        _judge.run_judge(ctx, latest_results=latest_results,
                         current_best_mae=current_best_mae)

        # STEP 7: Checkpoint
        _checkpoint.save_checkpoint(ctx)

        # Tail summary (kept inline — it touches ctx.tree directly).
        print(f"\n{ctx.tree.get_direction_stats()}")
        best = ctx.get_best_valid_experiment()
        if best:
            print(f"\nCurrent best: {best.exp_name} = {best.avg_score_mae:.4f}")

        return True
