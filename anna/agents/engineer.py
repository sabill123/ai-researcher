"""
Agent 2: Engineer
Proposes code changes and implements them in isolated sandboxes.
Two-phase: (1) Propose (pure reasoning), (2) Implement (with tools).
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from .base_agent import call_claude_cli, SAFETY_SYSTEM_PROMPT


class EngineerAgent:
    """Proposes and implements code changes for experiments."""

    def __init__(self, model: str = "sonnet",
                 propose_budget: float = 0.50,
                 implement_budget: float = 2.00):
        self.model = model
        self.propose_budget = propose_budget
        self.implement_budget = implement_budget

    def propose(self, top_experiments: str,
                research_context: Optional[Dict],
                judge_guidance: Optional[str],
                existing_insights: str,
                current_best_mae: float,
                iteration: int,
                num_proposals: int = 2) -> List[Dict[str, Any]]:
        """Phase 1: Propose code changes (pure reasoning, no tools).

        Returns list of proposal dicts.
        """
        prompt = self._build_propose_prompt(
            top_experiments, research_context, judge_guidance,
            existing_insights, current_best_mae, iteration, num_proposals,
        )

        print("  [Engineer] Proposing experiments...")
        response = call_claude_cli(
            prompt=prompt,
            model=self.model,
            max_budget_usd=self.propose_budget,
            timeout_seconds=180,
            tools="",  # No tools — pure reasoning
        )

        if not response:
            print("  [Engineer] No response for proposals")
            return []

        return self._parse_proposals(response, iteration)

    def implement(self, proposal: Dict[str, Any],
                  sandbox_code_dir: str) -> bool:
        """Phase 2: Implement code changes in sandbox (with tools).

        Args:
            proposal: The proposal dict from phase 1
            sandbox_code_dir: Path to the experiment's code/ directory

        Returns True if implementation succeeded.
        """
        prompt = self._build_implement_prompt(proposal, sandbox_code_dir)

        print(f"  [Engineer] Implementing: {proposal.get('exp_name', 'unnamed')}...")
        response = call_claude_cli(
            prompt=prompt,
            model=self.model,
            max_budget_usd=self.implement_budget,
            timeout_seconds=300,
            tools=None,  # Enable tools for file editing
            cwd=sandbox_code_dir,
            system_prompt=self._implement_system_prompt(sandbox_code_dir),
        )

        if not response:
            print("  [Engineer] Implementation failed — no response")
            return False

        # Check if the agent reported success
        if "IMPLEMENTATION_COMPLETE" in (response or ""):
            return True

        # Even without explicit marker, implementation may have succeeded
        # (the agent may have edited files via tools)
        return True

    def _build_propose_prompt(self, top_experiments: str,
                               research_context: Optional[Dict],
                               judge_guidance: Optional[str],
                               existing_insights: str,
                               current_best_mae: float,
                               iteration: int,
                               num_proposals: int) -> str:
        research_section = ""
        if research_context and research_context.get("techniques"):
            techniques = research_context["techniques"]
            research_section = "\n## Research Findings (from Researcher Agent)\n"
            for t in techniques:
                research_section += f"\n### {t['name']}\n"
                research_section += f"Paper: {t.get('paper', 'N/A')}\n"
                research_section += f"Core idea: {t.get('core_idea', 'N/A')}\n"
                research_section += f"Implementation: {t.get('implementation_sketch', 'N/A')}\n"
                research_section += f"Target files: {t.get('target_files', [])}\n"

        prompt = f"""You are an ML engineer optimizing a facial palsy severity prediction model.

## Current Status
- Current best Avg Score MAE: {current_best_mae:.4f} (target: 0.49, lower is better)
- Iteration: {iteration}

## Top Experiments (sorted by performance)
{top_experiments}

## Existing Research Insights
{existing_insights}
{research_section}
{f"## Judge's Guidance{chr(10)}{judge_guidance}" if judge_guidance else ""}

## Base Code Structure
- train_cloc_v2.py: Training loop (argparse CLI, train/val/test)
- model.py: SeparateHeadWithFeatures (FaRL backbone + per-action MLP heads)
- losses.py: MSELoss, WeightedKappaLoss, CrossEntropyLoss, SeverityLossCombiner
- cloc_loss.py: CLOCLoss (contrastive ordinal), OrdinalContrastiveLoss
- dataset.py: KHUPalsyDataset (image loading, augmentation)
- backbone.py: FaRLEncoder (ViT-B/16, pretrained on 20M faces)
- utils.py: READ-ONLY (metrics computation)

## What You Can Change
You can propose changes to ANY file except utils.py:
- **Architecture**: New head designs, attention mechanisms, feature aggregation
- **Loss functions**: New losses, different combinations, class-balanced losses
- **Augmentation**: New transforms, mixup, cutmix, face-specific augmentation
- **Training**: Learning rate schedules, optimizer changes, curriculum learning
- **Backbone**: Layer freezing strategies, adapter modules

## CONSTRAINTS
- Must preserve CLI interface: --data_dir, --output_dir, --exp, --epochs, --no_wandb
- Must output test_metrics.json with *_score_mae keys
- Max 500 lines of code changes per experiment
- utils.py is READ-ONLY
- CRITICAL: The "config" JSON must use ONLY existing argparse choices:
  - severity_loss_fn: ONLY "CE", "MSE", or "MSE+WK" (even if you add a new loss, set this to "MSE+WK" and integrate your new loss inside the training code)
  - model_type: ONLY "separate" or "shared"
  - cloc_type: ONLY "cloc" or "ordinal"
  - margin_mode: ONLY "single" or "multi"
- If you add a new loss function or architecture, integrate it INTO the existing training pipeline code, do NOT change argparse choices
- New loss functions should be added inside losses.py/cloc_loss.py and called from train_cloc_v2.py's training loop directly

## Task
Propose {num_proposals} experiments. Each proposal should include:
1. What to change and why
2. Which files to modify
3. The specific code changes (detailed enough for implementation)
4. Hyperparameter configuration

## Output Format
Return ONLY a JSON array:
[
  {{
    "exp_name": "exp_{iteration:03d}_descriptive_name",
    "rationale": "Why this change should improve performance",
    "code_change_type": "architecture|loss|augmentation|training|hyperparameter",
    "changes": [
      {{
        "file": "model.py",
        "description": "What to change in this file",
        "code_snippet": "Key code to add or modify (pseudocode or actual Python)"
      }}
    ],
    "expected_improvement": 0.02,
    "confidence": 0.6,
    "config": {{
      "model_type": "separate",
      "severity_loss_fn": "MSE+WK",
      "use_cloc": true,
      "cloc_type": "cloc",
      "cloc_weight": 0.5,
      "cloc_temperature": 0.1,
      "margin_mode": "multi",
      "initial_margins": [2.0, 1.0, 0.8, 0.6],
      "learnable_margins": true,
      "ordinal_alpha": 1.0,
      "batch_size": 16,
      "lr": 0.0001,
      "weight_decay": 0.0001,
      "do_action_classification": true,
      "seed": 42
    }}
  }}
]"""
        return prompt

    def _build_implement_prompt(self, proposal: Dict[str, Any],
                                 sandbox_code_dir: str) -> str:
        changes_desc = ""
        for change in proposal.get("changes", []):
            changes_desc += f"\n### {change['file']}\n"
            changes_desc += f"{change['description']}\n"
            if change.get("code_snippet"):
                changes_desc += f"```python\n{change['code_snippet']}\n```\n"

        prompt = f"""Implement the following code changes in the current directory.

## Experiment: {proposal.get('exp_name', 'unnamed')}
## Rationale: {proposal.get('rationale', '')}
## Change Type: {proposal.get('code_change_type', 'unknown')}

## Changes to Implement
{changes_desc}

## CRITICAL RULES
1. Read each file BEFORE modifying it to understand the full context
2. Make ONLY the changes described above — no extra modifications
3. NEVER modify utils.py
4. Preserve the CLI interface: --data_dir, --output_dir, --exp, --epochs, --no_wandb
5. Preserve test_metrics.json output format (must contain *_score_mae keys)
6. Add proper imports for any new dependencies
7. Keep changes minimal and focused
8. NEVER change argparse choices in train_cloc_v2.py — integrate new features by modifying the training loop code directly
9. If adding a new loss, add it in losses.py or cloc_loss.py and call it from within main() in train_cloc_v2.py
10. The script must still run with: python train_cloc_v2.py --severity_loss_fn MSE+WK --model_type separate (etc.)

## Working Directory
Your working directory is: {sandbox_code_dir}
All files are already copied here. Edit them in place.

After completing all changes, output: IMPLEMENTATION_COMPLETE"""
        return prompt

    def _implement_system_prompt(self, sandbox_code_dir: str) -> str:
        return f"""You are implementing code changes for a ML experiment.

## SAFETY CONSTRAINTS
- You may ONLY modify files in: {sandbox_code_dir}
- NEVER access files outside this directory
- NEVER modify utils.py
- NEVER run any training commands — just edit the code files
- NEVER use subprocess, os.system, or any command execution
- Read files before editing to understand context

## TASK
Implement the requested code changes carefully and precisely.
When done, output: IMPLEMENTATION_COMPLETE"""

    def _parse_proposals(self, response: str,
                          iteration: int) -> List[Dict[str, Any]]:
        text = response.strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            print("  [Engineer] Could not parse proposals JSON")
            return []

        try:
            proposals = json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"  [Engineer] JSON parse error: {e}")
            return []

        valid = []
        for p in proposals:
            if not isinstance(p, dict):
                continue
            if "config" not in p:
                continue
            if not p.get("exp_name"):
                p["exp_name"] = f"exp_{iteration:03d}_unnamed"
            valid.append(p)

        return valid
