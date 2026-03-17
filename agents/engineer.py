"""
Agent 2: Engineer
Proposes code changes and implements them in isolated sandboxes.
Two-phase: (1) Propose (pure reasoning), (2) Implement (with tools).
"""

import filecmp
import json
import os
import re
from typing import Any, Dict, List, Optional

from .base_agent import call_claude_cli, SAFETY_SYSTEM_PROMPT


class EngineerAgent:
    """Proposes and implements code changes for experiments."""

    def __init__(self, model: str = "sonnet",
                 propose_budget: float = 0.50,
                 implement_budget: float = 5.00):
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
                  sandbox_code_dir: str,
                  baseline_code_dir: str = "") -> bool:
        """Phase 2: Implement code changes in sandbox (with tools).

        Args:
            proposal: The proposal dict from phase 1
            sandbox_code_dir: Path to the experiment's code/ directory
            baseline_code_dir: Path to the original baseline code for diff verification

        Returns True if implementation succeeded AND code was actually modified.
        """
        prompt = self._build_implement_prompt(proposal, sandbox_code_dir)
        exp_name = proposal.get('exp_name', 'unnamed')

        print(f"  [Engineer] Implementing: {exp_name}...")
        response = call_claude_cli(
            prompt=prompt,
            model=self.model,
            max_budget_usd=self.implement_budget,
            timeout_seconds=600,
            tools=None,  # Enable default tools
            cwd=sandbox_code_dir,
            system_prompt=self._implement_system_prompt(sandbox_code_dir),
            allowed_tools="Edit,Read,Write,Glob,Grep",
        )

        if not response:
            print(f"  [Engineer] Implementation failed for {exp_name} — no response")
            return False

        # Verify that at least one file was actually modified
        if baseline_code_dir and os.path.isdir(baseline_code_dir):
            changed = self._count_changed_files(baseline_code_dir, sandbox_code_dir)
            if changed == 0:
                print(f"  [Engineer] Implementation FAILED for {exp_name} — no files were modified!")
                return False
            print(f"  [Engineer] Implementation OK for {exp_name} — {changed} file(s) modified")

        return True

    def _count_changed_files(self, original_dir: str, modified_dir: str) -> int:
        """Count how many .py files differ between original and modified dirs."""
        changed = 0
        for filename in os.listdir(modified_dir):
            if not filename.endswith(".py"):
                continue
            orig = os.path.join(original_dir, filename)
            mod = os.path.join(modified_dir, filename)
            if os.path.exists(orig) and os.path.exists(mod):
                if not filecmp.cmp(orig, mod, shallow=False):
                    changed += 1
        return changed

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

        prompt = f"""You are an ML research engineer developing a facial palsy severity prediction model.
This is MODEL RESEARCH & DEVELOPMENT — not just hyperparameter tuning.

## Current Status
- Current best Avg Score MAE: {current_best_mae:.4f} (target: 0.49, lower is better)
- Iteration: {iteration}

## Top Experiments (sorted by performance)
{top_experiments}

## Existing Research Insights
{existing_insights}
{research_section}
{f"## Judge's Guidance{chr(10)}{judge_guidance}" if judge_guidance else ""}

## Base Code Structure (THESE ARE THE ONLY FILES IN THE SANDBOX)
The sandbox contains EXACTLY these 6 files. NO other files exist:
- train.py: Training loop (argparse CLI, train/test, 100 epochs, AdamW + CosineAnnealing)
- model.py: BaselinePalsyClassifier (FaRL backbone + separate per-action MLP severity heads + action classifier)
- losses.py: MSELoss, WeightedKappaLoss, XEntropyLoss(hard/soft/SORD), SeverityLossCombiner
- dataset.py: KHUPalsyDataset (6 actions: 눈_살짝감기, 눈_질끈감기, 이마_주름, 입_이, 입_우, 안면_무표정)
- backbone.py: FaRLEncoder (ViT-B/16, pretrained on 20M faces, intermediate layers [3,5,7,11])
  - MultiLayerFeatureCombiner: takes [B, L, N, D] → CLS + mean/max pool → weighted sum → linear
  - feature_dim=768, 1025 tokens (1024 patches + CLS) for 512x512 input
  - 12 transformer blocks total, hooks on layers [3,5,7,11]
- utils.py: READ-ONLY (metrics computation — compute_mae, compute_score_mae, AverageMeter)

IMPORTANT: There is NO cloc_loss.py, NO train_cloc_v2.py, NO __init__.py in this codebase.
Do NOT create new .py files. ALL changes must be made within the existing 5 editable files above.
If you want to add new loss functions, add them to losses.py.
If you want to add new model architectures, add them to model.py.

## PRIORITY: BACKBONE & ARCHITECTURE IMPROVEMENTS (MANDATORY)
This is a model development project. At least {max(1, num_proposals - 1)} of {num_proposals} proposals MUST modify
backbone.py or model.py with meaningful architectural changes. Pure loss-only or hyperparameter-only changes are NOT sufficient.

### High-Impact Architecture Directions to Explore:
1. **Backbone Adaptation** (backbone.py):
   - LoRA / Adapter layers injected into ViT transformer blocks (self.net.transformer.resblocks)
   - Learnable layer selection weights or attention over intermediate layers
   - Visual prompt tuning: prepend learnable tokens to ViT input sequence
   - Spatial attention pooling instead of simple mean/max aggregation in MultiLayerFeatureCombiner

2. **Model Architecture** (model.py):
   - Cross-attention between facial region patches and action-specific query tokens
   - Transformer decoder heads instead of simple MLP severity heads
   - Action-conditioned feature extraction: different attention masks for eye vs mouth vs forehead actions
   - Multi-scale feature pyramid from different ViT layers with FPN-style aggregation
   - Spatial attention maps guided by facial action regions (eyes=upper patches, mouth=lower patches)

3. **Feature Aggregation** (backbone.py + model.py):
   - Attention-weighted feature combination across layers (replace simple weighted sum in MultiLayerFeatureCombiner)
   - Separate CLS-based and spatial-based feature paths fused before classification
   - Progressive feature refinement: cross-layer attention between ViT intermediate outputs

### Lower Priority (at most 1 of {num_proposals} proposals can be loss-only or augmentation-only):
4. **Loss functions**: Only if combined with architecture changes
5. **Training strategy**: Discriminative learning rates (lower lr for backbone, higher for heads)
6. **Augmentation**: Face-specific augmentation, ordinal-aware mixup

## CONSTRAINTS
- Must preserve CLI interface: --data_dir, --output_dir, --exp, --epochs, --no_wandb
- Must output test_metrics.json with *_score_mae keys
- Max 500 lines of code changes per experiment
- utils.py is READ-ONLY
- CRITICAL: The "config" JSON must use ONLY existing argparse choices:
  - severity_loss_fn: ONLY "CE", "SORD", "MSE", "MSE+WK" (if you add a new loss, set this to "MSE+WK" and integrate your new loss inside the training code)
  - model_type: ONLY "baseline", "baseline_shared_head", or "simple_baseline"
  - head_type: MUST be "feature_fusion" (NEVER use "linear" — feature_fusion performs significantly better)
- If you add a new loss function or architecture, integrate it INTO the existing training pipeline code, do NOT change argparse choices
- New loss functions should be added inside losses.py and called from train.py's training loop directly
- batch_size MUST be 12 (not 16, to avoid CUDA OOM on 24GB GPU)
- Ensure numerical stability: avoid NaN/Inf from log(0), division by zero, or unbounded loss terms. Always add epsilon (1e-8) where needed. Clamp loss values if they could explode. Test that your loss stays in a reasonable range (~0.5-5.0) at initialization.

## Task
Propose {num_proposals} experiments focused on MODEL DEVELOPMENT. Each proposal MUST include:
1. What architectural change to backbone.py or model.py and why
2. Which files to modify (MUST include backbone.py or model.py for architecture proposals)
3. The specific code changes (actual Python classes/methods, not pseudocode)
4. Hyperparameter configuration

## Output Format
Return ONLY a JSON array:
[
  {{
    "exp_name": "exp_{iteration:03d}_descriptive_name",
    "rationale": "Why this architectural change should improve performance",
    "code_change_type": "architecture",
    "changes": [
      {{
        "file": "backbone.py or model.py",
        "description": "What architectural change to make",
        "code_snippet": "Actual Python code for the new module/class"
      }}
    ],
    "expected_improvement": 0.02,
    "confidence": 0.6,
    "config": {{
      "model_type": "baseline",
      "severity_loss_fn": "MSE+WK",
      "head_type": "feature_fusion",
      "batch_size": 12,
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
8. NEVER change argparse choices in train.py — integrate new features by modifying the training loop code directly
9. If adding a new loss, add it in losses.py and call it from within main() in train.py
10. The script must still run with: python train.py --severity_loss_fn MSE+WK --model_type baseline --head_type feature_fusion (etc.)
11. DO NOT CREATE NEW .py FILES. The sandbox has ONLY: train.py, model.py, losses.py, dataset.py, backbone.py, utils.py
12. There is NO cloc_loss.py, NO train_cloc_v2.py — do not import from them or create them

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
