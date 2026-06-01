"""Pure-read query helpers used by the iteration steps.

Extracted from ``orchestrator.py`` (module #8, 2026-06-01). These functions
were originally Orchestrator methods (``_get_best_valid_experiment`` etc.)
that took only ``self.db`` / ``self.config`` and returned values — no
mutation, no agent calls, no GPU side-effects. Moving them out keeps
``orchestrator.py`` narrow to its true responsibilities (composition root,
lifecycle, signal handling).

The Orchestrator class re-exposes them as thin bound methods so external
callers (cli.py, watchdog) that imported orchestrator-level helpers keep
working without churn. Step modules call them through ``ctx.get_*``
bindings, which is just a function reference — no class membership needed.
"""

from __future__ import annotations

import json as _json
import math
import os
from collections import Counter
from datetime import datetime as _dt
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..config import SystemConfig
    from ..experiment_db import ExperimentDB, ExperimentRecord


_ACTIONS = [
    "눈_살짝감기", "눈_질끈감기", "이마_주름",
    "입_이", "입_우", "안면_무표정",
]


# ------------------------------------------------------------------ sandbox
def has_valid_sandbox_code(
    config: "SystemConfig",
    exp: Optional["ExperimentRecord"],
) -> bool:
    """True if an experiment's code dir contains every sandbox file.

    ``create_sandbox()`` only copies ``config.sandbox_files``, so an
    experiment missing any of them cannot be inherited cleanly. This check
    also separates anna_v2-era experiments from the pre-anna_v2 ones
    (train_cloc_v2.py layout), which were scored on a different,
    non-comparable test set (MAE ~0.27 vs ~0.60 now) — so they must NOT
    count toward "current best", current_best_mae, or the target check.
    """
    if not exp or not getattr(exp, "experiment_dir", None):
        return False
    cd = os.path.join(exp.experiment_dir, "code")
    if not os.path.isdir(cd):
        return False
    return all(
        os.path.exists(os.path.join(cd, f))
        for f in config.sandbox_files
    )


# ----------------------------------------------------------------- top-k
def get_best_valid_experiment(
    db: "ExperimentDB",
    config: "SystemConfig",
) -> Optional["ExperimentRecord"]:
    """Best experiment restricted to anna_v2-compatible runs.

    db.get_best_experiment() ranks over the WHOLE table, dominated by
    pre-anna_v2 experiments scored on an old, easier test set. Using that
    as 'current best' falsely trips the target check (exp_286 = 0.2724 on
    the old test set <= target 0.49). get_top_k is MAE-ascending, so the
    first valid-sandbox experiment is the best one.
    """
    for exp in db.get_top_k(300):
        if has_valid_sandbox_code(config, exp):
            return exp
    return None


def get_top_valid_k(
    db: "ExperimentDB",
    config: "SystemConfig",
    k: int,
) -> List["ExperimentRecord"]:
    """Top-k experiments restricted to anna_v2-compatible runs.

    Used for Engineer/Judge prompt context. Without this filter the
    prompt's 'top 15' was dominated by pre-anna_v2 experiments
    (exp_286 = 0.2724 on the old test set), making the agents imitate
    an unreachable target and propose strategies that don't transfer.
    """
    out: List["ExperimentRecord"] = []
    for exp in db.get_top_k(300):
        if has_valid_sandbox_code(config, exp):
            out.append(exp)
            if len(out) >= k:
                break
    return out


# ---------------------------------------------------------------- parent
def get_parent_experiment(
    db: "ExperimentDB",
    config: "SystemConfig",
) -> Optional["ExperimentRecord"]:
    """Select a parent for next iteration with UCB1-style exploration.

    Plateau diagnosis 2026-05-13: original implementation always returned
    the first valid Top-5 candidate, causing the same parent to be reused
    10+ times in 50 iterations. Replaced with:
      1) Wider candidate pool (Top-20).
      2) Reuse cap: any parent already used >= 3 times in the last 20
         iterations is excluded unless nothing else is valid.
      3) UCB1 score: ``-MAE + c * sqrt(log(N) / (1 + uses))`` so that
         rarely-used candidates are mixed in with the current best.
    Falls back to None (baseline) only if no valid parent exists.

    Plateau diagnosis 2026-05-18: get_top_k(20) is globally dominated by
    pre-anna_v2 experiments (train_cloc_v2.py layout, MAE ~0.27 on the OLD
    test set) which are NOT valid sandbox parents. All 20 fail
    has_valid_code() -> "No valid parent" -> every iteration restarts from
    baseline -> no lineage. Pull a wide pool and filter to valid code
    FIRST, then keep the best 20 valid candidates for UCB1.
    """
    candidates = db.get_top_k(300)
    if not candidates:
        return None

    # Recent parent reuse counts (last 20 child experiments)
    recent_children = (
        db.get_recent_experiments(limit=20)
        if hasattr(db, "get_recent_experiments") else []
    )
    if not recent_children:
        with db._conn() as conn:
            cur = conn.execute(
                "SELECT parent_experiment_id FROM experiments "
                "WHERE parent_experiment_id IS NOT NULL "
                "ORDER BY created_at DESC LIMIT 20"
            )
            recent_parents = [r[0] for r in cur.fetchall()]
    else:
        recent_parents = [
            e.parent_experiment_id for e in recent_children
            if getattr(e, "parent_experiment_id", None)
        ]
    use_counts = Counter(recent_parents)
    REUSE_CAP = 3

    valid = [e for e in candidates if has_valid_sandbox_code(config, e)]
    if not valid:
        return None
    valid = sorted(valid, key=lambda e: e.avg_score_mae)[:20]

    eligible = [e for e in valid if use_counts.get(e.id, 0) < REUSE_CAP]
    pool = eligible if eligible else valid

    # UCB1 scoring: reward = -MAE (lower is better), bonus = exploration
    c_explore = 0.05  # tuned so a 3-uses-vs-0-uses gap ≈ 0.02 MAE bonus
    N = max(1, sum(use_counts.values()))
    scored = []
    best_mae = min(e.avg_score_mae for e in pool)
    for e in pool:
        uses = use_counts.get(e.id, 0)
        exploit = -(e.avg_score_mae - best_mae)
        explore = c_explore * math.sqrt(math.log(N + 1) / (1 + uses))
        scored.append((exploit + explore, e, uses))
    scored.sort(key=lambda t: t[0], reverse=True)
    chosen_score, chosen, chosen_uses = scored[0]
    print(
        f"  [parent] selected {chosen.exp_name[:60]} "
        f"(MAE={chosen.avg_score_mae:.4f}, "
        f"recent-uses={chosen_uses}/{REUSE_CAP}, "
        f"ucb1={chosen_score:+.4f})"
    )
    return chosen


# ----------------------------------------------------------- weak actions
def get_weak_actions(
    db: "ExperimentDB",
    best_exp: Optional["ExperimentRecord"],
) -> List[str]:
    """Return actions to prioritize next.

    Originally returned the 3 worst actions of the current best. After the
    2026-05-13 stagnation diagnosis, we additionally compute the per-action
    ORACLE gap (current_best minus the all-time best per action) and rotate
    in actions that have a large oracle gap but are NOT currently the most
    attacked one. This breaks the "kill the worst-action of the best model
    in every iteration" loop that produced the 입_우 68% concentration.
    """
    if not best_exp or not best_exp.per_action_results:
        return ["입_우", "입_이"]

    cur_action_mae = {}
    for k, v in best_exp.per_action_results.items():
        # Match ONLY the exact per-action MAE key 'test_<action>_score_mae'.
        # Bug fix 2026-05-15: a substring check ("_score_mae" in k) also
        # matched CI keys like 'test_입_이_score_mae_ci_upper', producing
        # phantom actions ('입_이_ci_upper') in the weak-action focus list.
        if k.startswith("test_") and k.endswith("_score_mae"):
            a = k[len("test_"):-len("_score_mae")]
            if a in _ACTIONS:
                cur_action_mae[a] = v

    # Recent worst-action concentration (last 30 experiments)
    recent_worst: Counter = Counter()
    try:
        with db._conn() as conn:
            rows = conn.execute(
                "SELECT per_action_results_json FROM experiments "
                "WHERE status IN ('completed','validated','early_stopped') "
                "  AND per_action_results_json IS NOT NULL "
                "ORDER BY completed_at DESC LIMIT 30"
            ).fetchall()
        for r in rows:
            par = _json.loads(r[0])
            m = {a: par.get(f"test_{a}_score_mae") for a in _ACTIONS}
            m = {k: v for k, v in m.items() if v is not None}
            if m:
                recent_worst[max(m, key=m.get)] += 1
    except Exception:
        recent_worst = Counter()

    # Per-action oracle gap = current_best minus best-ever per action
    oracle = {a: float("inf") for a in _ACTIONS}
    try:
        with db._conn() as conn:
            all_rows = conn.execute(
                "SELECT per_action_results_json FROM experiments "
                "WHERE status IN ('completed','validated','early_stopped') "
                "  AND per_action_results_json IS NOT NULL"
            ).fetchall()
        for r in all_rows:
            par = _json.loads(r[0])
            for a in _ACTIONS:
                v = par.get(f"test_{a}_score_mae")
                if v is not None and v < oracle[a]:
                    oracle[a] = v
    except Exception:
        pass

    # Score each action: (current MAE) + (oracle gap = potential gain)
    #                    − (recency penalty if it's been attacked too much)
    scored = []
    total_recent = max(1, sum(recent_worst.values()))
    for a, mae in cur_action_mae.items():
        gap = mae - oracle.get(a, mae)
        attack_share = recent_worst.get(a, 0) / total_recent
        score = mae + gap - 0.15 * attack_share
        scored.append((score, a, mae, gap, attack_share))
    scored.sort(key=lambda t: t[0], reverse=True)
    chosen = [a for _, a, *_ in scored[:3]]
    details = ", ".join(
        f"{a}(mae={m:.3f},gap={g:+.3f},attack={s*100:.0f}%)"
        for _, a, m, g, s in scored
    )
    print(f"  [weak-actions] {details}  →  next focus: {chosen}")
    return chosen


# ---------------------------------------------------------- VLM baseline
def get_vlm_baseline_context(db: "ExperimentDB") -> Optional[str]:
    """VLM baseline results as a context string for Judge/Engineer."""
    all_exps = db.get_all_experiments()
    vlm_exps = [
        e for e in all_exps
        if e.code_change_type == "vlm_baseline" and e.status == "completed"
    ]
    if not vlm_exps:
        return None
    best_vlm = min(vlm_exps, key=lambda e: e.avg_score_mae or 999)
    lines = [
        f"VLM Baseline ({best_vlm.config.get('vlm_model', 'unknown')}): "
        f"avg_score_mae={best_vlm.avg_score_mae:.4f}"
    ]
    if best_vlm.per_action_results:
        for k, v in sorted(best_vlm.per_action_results.items()):
            if (k.startswith("test_") and k.endswith("_score_mae")
                    and isinstance(v, (int, float))):
                action = k[len("test_"):-len("_score_mae")]
                lines.append(f"  {action}: {v:.4f}")
        w1 = best_vlm.per_action_results.get("vlm_within_1", 0)
        if w1:
            lines.append(f"  Within-1 accuracy: {w1:.1%}")
    return "\n".join(lines)


# ------------------------------------------------------ stagnation diagnosis
def compute_stagnation_diagnosis(
    db: "ExperimentDB",
    config: "SystemConfig",
) -> str:
    """Diagnose recent stagnation patterns so Judge can break local search.

    Added 2026-05-13 after diagnosis showed:
      - best MAE stuck at 0.2724 for 19+ hours
      - one parent (exp_198) reused 10x in 50 iterations
      - worst-action = 입_우 in 68% of recent experiments
      - 96% of recent experiments fail to beat current best

    The text is passed to Judge.judge() as the separate
    ``stagnation_diagnosis`` argument (never concatenated into the
    research_context dict — Judge calls .get on that).
    """
    try:
        with db._conn() as conn:
            rows = conn.execute(
                "SELECT exp_name, avg_score_mae, per_action_results_json, "
                "       completed_at, parent_experiment_id "
                "FROM experiments "
                "WHERE status IN ('completed','validated','early_stopped') "
                "  AND avg_score_mae IS NOT NULL "
                "  AND per_action_results_json IS NOT NULL "
                "ORDER BY completed_at DESC LIMIT 50"
            ).fetchall()
    except Exception:
        return ""
    if len(rows) < 10:
        return ""

    worst_cnt: Counter = Counter()
    for r in rows:
        try:
            par = _json.loads(r[2])
        except Exception:
            continue
        m = {a: par.get(f"test_{a}_score_mae") for a in _ACTIONS}
        m = {k: v for k, v in m.items() if v is not None}
        if m:
            worst_cnt[max(m, key=m.get)] += 1

    parent_cnt = Counter(r[4] for r in rows if r[4])
    top_parent = parent_cnt.most_common(1)
    cur_best = get_best_valid_experiment(db, config)
    cur_best_mae = cur_best.avg_score_mae if cur_best else float("inf")
    recent_maes = [r[1] for r in rows]
    beats = sum(1 for v in recent_maes if v < cur_best_mae)

    hours_since = None
    if cur_best and cur_best.completed_at:
        try:
            t = _dt.fromisoformat(cur_best.completed_at[:19])
            hours_since = (_dt.now() - t).total_seconds() / 3600
        except Exception:
            pass

    lines = ["=== STAGNATION DIAGNOSIS (auto-generated) ==="]
    lines.append(f"Current best MAE: {cur_best_mae:.4f}")
    if hours_since is not None:
        lines.append(f"Hours since last best improvement: {hours_since:.1f}h")
    lines.append(
        f"Last 50 experiments: {beats}/{len(rows)} "
        f"({100*beats/len(rows):.0f}%) beat current best — "
        f"system is in local-search loop."
    )
    if worst_cnt:
        top_worst, top_n = worst_cnt.most_common(1)[0]
        lines.append(
            f"Worst-action concentration: {top_worst} appears as worst "
            f"in {top_n}/{sum(worst_cnt.values())} recent experiments "
            f"({100*top_n/sum(worst_cnt.values()):.0f}%). "
            f"Other worst-actions: " +
            ", ".join(f"{a}={n}" for a, n in worst_cnt.most_common()[1:5])
        )
    if top_parent and top_parent[0][1] >= 4:
        parent = db.get_experiment(top_parent[0][0])
        pn = parent.exp_name if parent else top_parent[0][0]
        lines.append(
            f"Parent reuse: {pn[:60]} used as parent "
            f"{top_parent[0][1]} times in the last 50. "
            f"Parent diversity is too low."
        )

    # Per-action best (oracle gap insight)
    try:
        with db._conn() as conn:
            all_rows = conn.execute(
                "SELECT per_action_results_json FROM experiments "
                "WHERE status IN ('completed','validated','early_stopped') "
                "  AND per_action_results_json IS NOT NULL"
            ).fetchall()
        per_best = {a: float("inf") for a in _ACTIONS}
        for r in all_rows:
            par = _json.loads(r[0])
            for a in _ACTIONS:
                v = par.get(f"test_{a}_score_mae")
                # Filter stub zeros (e.g. baseline_6action_merged left
                # an unfilled 입_우 = 0.0).
                if v is not None and v > 0.05 and v < per_best[a]:
                    per_best[a] = v
        if all(v < float("inf") for v in per_best.values()):
            oracle = sum(per_best.values()) / 6
            lines.append(
                f"Per-action oracle combination: "
                f"avg(best_per_action) = {oracle:.4f} vs current best "
                f"{cur_best_mae:.4f} "
                f"(oracle gap {cur_best_mae - oracle:+.4f}). "
                f"Per-action bests: " +
                ", ".join(f"{a}={v:.4f}" for a, v in per_best.items())
            )
    except Exception:
        pass

    lines.append("")
    lines.append("HINTS for next iteration:")
    lines.append(
        "  - Avoid generating yet another variant of the most-reused parent."
    )
    if worst_cnt:
        top_worst = worst_cnt.most_common(1)[0][0]
        lines.append(
            f"  - Stop attacking {top_worst} exclusively; "
            "rotate to other under-attended actions."
        )
    lines.append(
        "  - Prefer techniques that target the meta-learner / "
        "ensemble combiner (current stacked-ridge cannot close the oracle gap)."
    )
    lines.append(
        "  - Consider parent diversification: build on a different "
        "branch of the experiment tree, not just on the current top-1."
    )
    lines.append("=== END STAGNATION DIAGNOSIS ===")
    return "\n".join(lines)
