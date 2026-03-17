"""
SQLite-based experiment database for ANNA v2.
Extended from v1 with code_diff, code_change_type, experiment_dir columns.
"""

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExperimentRecord:
    id: str = ""
    exp_name: str = ""
    status: str = "proposed"  # proposed, validated, running, completed, failed
    priority: float = 0.0
    expected_improvement: float = 0.0
    confidence: float = 0.5
    config: Dict[str, Any] = field(default_factory=dict)
    gpu_id: Optional[int] = None
    pid: Optional[int] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    avg_score_mae: Optional[float] = None
    per_action_results: Optional[Dict[str, float]] = None
    results_dir: Optional[str] = None
    experiment_dir: Optional[str] = None  # v2: isolated experiment folder
    proposed_by: str = "manual"  # bootstrap, researcher, engineer, judge, manual
    parent_experiment_id: Optional[str] = None
    proposal_rationale: str = ""
    code_change_type: str = "hyperparameter"  # hyperparameter, architecture, loss, augmentation, training
    code_diff: Optional[str] = None  # v2: unified diff of code changes
    research_context: Optional[str] = None  # v2: papers/techniques that inspired this
    iteration: int = 0
    created_at: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class ExperimentDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    exp_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'proposed',
                    priority REAL DEFAULT 0.0,
                    expected_improvement REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.5,
                    config_json TEXT NOT NULL,
                    gpu_id INTEGER,
                    pid INTEGER,
                    started_at TEXT,
                    completed_at TEXT,
                    avg_score_mae REAL,
                    per_action_results_json TEXT,
                    results_dir TEXT,
                    experiment_dir TEXT,
                    proposed_by TEXT DEFAULT 'manual',
                    parent_experiment_id TEXT,
                    proposal_rationale TEXT DEFAULT '',
                    code_change_type TEXT DEFAULT 'hyperparameter',
                    code_diff TEXT,
                    research_context TEXT,
                    iteration INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON experiments(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_avg_score_mae ON experiments(avg_score_mae)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_iteration ON experiments(iteration)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _row_to_record(self, row: tuple, columns: List[str]) -> ExperimentRecord:
        d = dict(zip(columns, row))
        return ExperimentRecord(
            id=d["id"],
            exp_name=d["exp_name"],
            status=d["status"],
            priority=d.get("priority", 0.0),
            expected_improvement=d.get("expected_improvement", 0.0),
            confidence=d.get("confidence", 0.5),
            config=json.loads(d["config_json"]) if d.get("config_json") else {},
            gpu_id=d.get("gpu_id"),
            pid=d.get("pid"),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            avg_score_mae=d.get("avg_score_mae"),
            per_action_results=json.loads(d["per_action_results_json"]) if d.get("per_action_results_json") else None,
            results_dir=d.get("results_dir"),
            experiment_dir=d.get("experiment_dir"),
            proposed_by=d.get("proposed_by", "manual"),
            parent_experiment_id=d.get("parent_experiment_id"),
            proposal_rationale=d.get("proposal_rationale", ""),
            code_change_type=d.get("code_change_type", "hyperparameter"),
            code_diff=d.get("code_diff"),
            research_context=d.get("research_context"),
            iteration=d.get("iteration", 0),
            created_at=d.get("created_at", ""),
        )

    def add_experiment(self, exp: ExperimentRecord) -> str:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments
                (id, exp_name, status, priority, expected_improvement,
                 confidence, config_json, gpu_id, pid, started_at, completed_at,
                 avg_score_mae, per_action_results_json, results_dir, experiment_dir,
                 proposed_by, parent_experiment_id, proposal_rationale,
                 code_change_type, code_diff, research_context,
                 iteration, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp.id, exp.exp_name, exp.status,
                exp.priority, exp.expected_improvement, exp.confidence,
                json.dumps(exp.config, ensure_ascii=False),
                exp.gpu_id, exp.pid, exp.started_at, exp.completed_at,
                exp.avg_score_mae,
                json.dumps(exp.per_action_results, ensure_ascii=False) if exp.per_action_results else None,
                exp.results_dir, exp.experiment_dir, exp.proposed_by,
                exp.parent_experiment_id, exp.proposal_rationale,
                exp.code_change_type, exp.code_diff, exp.research_context,
                exp.iteration, exp.created_at,
            ))
        return exp.id

    def update_status(self, exp_id: str, status: str, **kwargs):
        sets = ["status = ?"]
        vals = [status]
        for key, val in kwargs.items():
            sets.append(f"{key} = ?")
            vals.append(val)
        vals.append(exp_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE experiments SET {', '.join(sets)} WHERE id = ?",
                vals
            )

    def update_results(self, exp_id: str, avg_score_mae: float,
                       per_action_results: dict, results_dir: str):
        with self._conn() as conn:
            conn.execute("""
                UPDATE experiments
                SET avg_score_mae = ?, per_action_results_json = ?,
                    results_dir = ?, status = 'completed',
                    completed_at = ?
                WHERE id = ?
            """, (
                avg_score_mae,
                json.dumps(per_action_results, ensure_ascii=False),
                results_dir,
                datetime.now().isoformat(),
                exp_id,
            ))

    def get_experiment(self, exp_id: str) -> Optional[ExperimentRecord]:
        with self._conn() as conn:
            cur = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
            columns = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_record(row, columns)

    def get_by_name(self, exp_name: str) -> Optional[ExperimentRecord]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM experiments WHERE exp_name = ? ORDER BY created_at DESC LIMIT 1",
                (exp_name,)
            )
            columns = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_record(row, columns)

    def get_all_completed(self) -> List[ExperimentRecord]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM experiments WHERE status = 'completed' ORDER BY avg_score_mae ASC"
            )
            columns = [desc[0] for desc in cur.description]
            return [self._row_to_record(row, columns) for row in cur.fetchall()]

    def get_best_experiment(self) -> Optional[ExperimentRecord]:
        with self._conn() as conn:
            cur = conn.execute("""
                SELECT * FROM experiments
                WHERE status = 'completed' AND avg_score_mae IS NOT NULL
                ORDER BY avg_score_mae ASC LIMIT 1
            """)
            columns = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_record(row, columns)

    def get_top_k(self, k: int = 10) -> List[ExperimentRecord]:
        with self._conn() as conn:
            cur = conn.execute("""
                SELECT * FROM experiments
                WHERE status = 'completed' AND avg_score_mae IS NOT NULL
                ORDER BY avg_score_mae ASC LIMIT ?
            """, (k,))
            columns = [desc[0] for desc in cur.description]
            return [self._row_to_record(row, columns) for row in cur.fetchall()]

    def get_running_experiments(self) -> List[ExperimentRecord]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM experiments WHERE status = 'running'"
            )
            columns = [desc[0] for desc in cur.description]
            return [self._row_to_record(row, columns) for row in cur.fetchall()]

    def get_experiments_by_iteration(self, iteration: int) -> List[ExperimentRecord]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM experiments WHERE iteration >= ? ORDER BY created_at DESC",
                (iteration,)
            )
            columns = [desc[0] for desc in cur.description]
            return [self._row_to_record(row, columns) for row in cur.fetchall()]

    def get_total_count(self) -> int:
        with self._conn() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM experiments")
            return cur.fetchone()[0]

    def get_experiments_by_change_type(self, change_type: str) -> List[ExperimentRecord]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM experiments WHERE code_change_type = ? AND status = 'completed' ORDER BY avg_score_mae ASC",
                (change_type,)
            )
            columns = [desc[0] for desc in cur.description]
            return [self._row_to_record(row, columns) for row in cur.fetchall()]

    def import_existing_results(self, results_base_dir: str) -> int:
        """Import existing experiment results from results_cloc_v2/ directory."""
        imported = 0
        if not os.path.exists(results_base_dir):
            return 0

        for exp_name in os.listdir(results_base_dir):
            exp_dir = os.path.join(results_base_dir, exp_name)
            if not os.path.isdir(exp_dir):
                continue

            for timestamp_dir in sorted(os.listdir(exp_dir)):
                full_dir = os.path.join(exp_dir, timestamp_dir)
                if not os.path.isdir(full_dir):
                    continue

                config_path = os.path.join(full_dir, "config.json")
                metrics_path = os.path.join(full_dir, "test_metrics.json")

                if not os.path.exists(config_path):
                    continue

                existing = self.get_by_name(f"{exp_name}/{timestamp_dir}")
                if existing:
                    continue

                with open(config_path) as f:
                    config = json.load(f)

                per_action_results = None
                avg_score_mae = None
                status = "completed"

                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        per_action_results = json.load(f)
                    score_mae_values = [
                        v for k, v in per_action_results.items()
                        if k.endswith("_score_mae")
                    ]
                    if score_mae_values:
                        avg_score_mae = sum(score_mae_values) / len(score_mae_values)
                else:
                    status = "failed"

                record = ExperimentRecord(
                    exp_name=f"{exp_name}/{timestamp_dir}",
                    status=status,
                    config=config,
                    avg_score_mae=avg_score_mae,
                    per_action_results=per_action_results,
                    results_dir=full_dir,
                    proposed_by="bootstrap",
                    code_change_type="hyperparameter",
                    created_at=timestamp_dir[:15] if len(timestamp_dir) >= 15 else datetime.now().isoformat(),
                )
                self.add_experiment(record)
                imported += 1

        return imported

    def format_experiments_for_prompt(self, experiments: List[ExperimentRecord]) -> str:
        lines = []
        for exp in experiments:
            if exp.avg_score_mae is None:
                continue
            cfg = exp.config
            line = (
                f"- {exp.exp_name}: avg_score_mae={exp.avg_score_mae:.4f} | "
                f"change_type={exp.code_change_type}, "
                f"model={cfg.get('model_type')}, loss={cfg.get('severity_loss_fn')}, "
                f"cloc={cfg.get('use_cloc')}, cloc_w={cfg.get('cloc_weight')}, "
                f"margins={cfg.get('initial_margins')}, "
                f"lr={cfg.get('lr')}, bs={cfg.get('batch_size')}"
            )
            lines.append(line)
            if exp.per_action_results:
                for k, v in sorted(exp.per_action_results.items()):
                    if "_score_mae" in k:
                        action = k.replace("test_", "").replace("_score_mae", "")
                        lines.append(f"    {action}: {v:.4f}")
            if exp.code_diff:
                # Show summary of code changes
                diff_lines = exp.code_diff.split('\n')
                added = sum(1 for l in diff_lines if l.startswith('+') and not l.startswith('+++'))
                removed = sum(1 for l in diff_lines if l.startswith('-') and not l.startswith('---'))
                lines.append(f"    code_changes: +{added}/-{removed} lines")
        return "\n".join(lines) if lines else "None available"
