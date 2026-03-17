"""
MCTS-inspired experiment tree for tracking exploration/exploitation.
Inspired by AI-Scientist-v2's tree search with UCB1 selection.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TreeNode:
    """A node in the experiment tree.

    Each node represents a research direction or technique category.
    Children represent specific experiments within that direction.
    """
    id: str
    name: str  # e.g., "attention_mechanism", "ordinal_loss", "augmentation"
    category: str  # architecture, loss, augmentation, training, hyperparameter
    parent_id: Optional[str] = None
    experiment_ids: List[str] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0  # Sum of improvements (negative MAE changes)
    best_mae: Optional[float] = None

    @property
    def avg_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb1(self, total_visits: int, exploration_weight: float = 1.414) -> float:
        """Upper Confidence Bound for balancing explore/exploit."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.avg_reward
        exploration = exploration_weight * math.sqrt(
            math.log(max(total_visits, 1)) / self.visits
        )
        return exploitation + exploration


class ExperimentTree:
    """Tracks the tree of research directions and experiments.

    Root children are technique categories. Leaves are experiments.
    Uses UCB1 to balance exploring new directions vs exploiting known good ones.
    """

    def __init__(self):
        self.nodes: Dict[str, TreeNode] = {}
        self._init_root_nodes()

    def _init_root_nodes(self):
        """Initialize with broad research direction categories."""
        categories = [
            ("architecture", "Architecture changes (heads, attention, feature aggregation)"),
            ("loss", "Loss function changes (new losses, combinations, weighting)"),
            ("augmentation", "Data augmentation strategies"),
            ("training", "Training strategy (scheduler, optimizer, curriculum)"),
            ("hyperparameter", "Hyperparameter tuning (lr, batch_size, margins)"),
            ("backbone", "Backbone modifications (freezing, adapters)"),
        ]
        for cat_id, name in categories:
            self.nodes[cat_id] = TreeNode(
                id=cat_id, name=name, category=cat_id,
            )

    @property
    def total_visits(self) -> int:
        return sum(n.visits for n in self.nodes.values() if n.parent_id is None)

    def select_direction(self) -> str:
        """Use UCB1 to select which research direction to explore next."""
        root_nodes = [n for n in self.nodes.values() if n.parent_id is None]
        total = self.total_visits
        best_node = max(root_nodes, key=lambda n: n.ucb1(total))
        return best_node.id

    def record_experiment(self, category: str, experiment_id: str,
                          mae_improvement: float, result_mae: float):
        """Record an experiment result in the tree."""
        if category not in self.nodes:
            self.nodes[category] = TreeNode(
                id=category, name=category, category=category,
            )

        node = self.nodes[category]
        node.visits += 1
        node.total_reward += mae_improvement  # Positive = improvement
        node.experiment_ids.append(experiment_id)
        if node.best_mae is None or result_mae < node.best_mae:
            node.best_mae = result_mae

    def get_direction_stats(self) -> str:
        """Format tree statistics for prompts."""
        lines = ["Research Direction Statistics:"]
        total = self.total_visits
        root_nodes = sorted(
            [n for n in self.nodes.values() if n.parent_id is None],
            key=lambda n: n.ucb1(total), reverse=True,
        )
        for node in root_nodes:
            ucb = node.ucb1(total)
            best_str = f"{node.best_mae:.4f}" if node.best_mae is not None else "N/A"
            ucb_str = f"{ucb:.4f}" if ucb != float('inf') else "inf"
            lines.append(
                f"  {node.id}: visits={node.visits}, "
                f"avg_reward={node.avg_reward:+.4f}, "
                f"best_mae={best_str}, "
                f"UCB1={ucb_str}"
            )
        return "\n".join(lines)

    def suggest_strategy(self, current_best_mae: float) -> str:
        """Suggest exploit vs explore based on tree state."""
        total = self.total_visits
        if total < 6:
            return "explore"  # Not enough data yet

        # Find best direction
        root_nodes = [n for n in self.nodes.values() if n.parent_id is None]
        best_direction = max(root_nodes, key=lambda n: n.avg_reward if n.visits > 0 else -999)

        # If best direction has < 3 visits, explore more
        if best_direction.visits < 3:
            return "explore"

        # If gap to target is large, explore
        gap = current_best_mae - 0.49
        if gap > 0.1:
            return "explore"

        # If recent improvements are diminishing, explore
        if best_direction.avg_reward < 0.005:
            return "explore"

        return "exploit"
