"""
Agent 2: Engineer
Proposes code changes and implements them in isolated sandboxes.
Two-phase: (1) Propose (pure reasoning), (2) Implement (with tools).
"""

import filecmp
import os
from typing import Any, Dict, List, Optional

from ..utils.logging_setup import get_logger
from .base_agent import SAFETY_SYSTEM_PROMPT, call_claude_cli, parse_json_response

logger = get_logger(__name__)


class EngineerAgent:
    """Proposes and implements code changes for experiments."""

    def __init__(self, model: str = "sonnet",
                 propose_budget: float = 2.00,
                 implement_budget: float = 5.00):
        self.model = model
        # propose_budget raised 0.50→2.00 on 2026-05-16: with num_proposals=4
        # and the richer prompt (retrospectives, parent history), 4 detailed
        # proposals need more budget/time. The old 180 s timeout caused every
        # propose() to time out → 0 proposals → 100+ empty iterations.
        self.propose_budget = propose_budget
        self.implement_budget = implement_budget

    def propose(self, top_experiments: str,
                research_context: Optional[Dict],
                judge_guidance: Optional[str],
                existing_insights: str,
                current_best_mae: float,
                iteration: int,
                num_proposals: int = 2,
                parent_experiment: Optional[Dict] = None,
                parent_retrospective_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """Phase 1: Propose code changes (pure reasoning, no tools).

        Returns list of proposal dicts.
        """
        prompt = self._build_propose_prompt(
            top_experiments, research_context, judge_guidance,
            existing_insights, current_best_mae, iteration, num_proposals,
            parent_experiment,
            parent_retrospective_text=parent_retrospective_text,
        )

        logger.info("[Engineer] Proposing experiments...")
        response = call_claude_cli(
            prompt=prompt,
            model=self.model,
            max_budget_usd=self.propose_budget,
            # 2026-05-31: 600s timeout caused 509 'No proposals' over 500+ iters
            # (Engineer Claude CLI was timing out due to API latency variation).
            # Raised to 1800 s. With 4 detailed proposals + retrospectives the
            # call sometimes needs 10+ min of LLM thinking.
            timeout_seconds=1800,
            tools="",  # No tools — pure reasoning
            agent="engineer.propose",
        )

        if not response:
            logger.warning("[Engineer] No response for proposals")
            return []

        return self._parse_proposals(str(response), iteration)

    def implement(self, proposal: Dict[str, Any],
                  sandbox_code_dir: str,
                  baseline_code_dir: str = "") -> bool:
        """Phase 2: Implement code changes in sandbox (with tools).

        Args:
            proposal: The proposal dict from phase 1
            sandbox_code_dir: Path to the experiment's code/ directory
            baseline_code_dir: Path to the original baseline code for diff verification

        Returns True if implementation succeeded AND code was actually modified.
        On the first call producing zero file diffs, retry once with an
        amended prompt — Claude occasionally responds with prose only and a
        fresh attempt usually fixes it.
        """
        exp_name = proposal.get('exp_name', 'unnamed')

        def _invoke(prompt: str, attempt_label: str) -> bool:
            logger.info("[Engineer] Implementing: %s (%s)...", exp_name, attempt_label)
            response = call_claude_cli(
                prompt=prompt,
                model=self.model,
                max_budget_usd=self.implement_budget,
                timeout_seconds=1200,  # 2026-05-31: same as propose, agent timeouts blocked progress
                tools=None,  # Enable default tools
                cwd=sandbox_code_dir,
                system_prompt=self._implement_system_prompt(sandbox_code_dir),
                allowed_tools="Edit,Read,Write,Glob,Grep",
                agent=f"engineer.implement.{attempt_label}",
            )

            if not response:
                logger.warning(
                    "[Engineer] Implementation %s for %s — no response",
                    attempt_label, exp_name,
                )
                return False

            if baseline_code_dir and os.path.isdir(baseline_code_dir):
                changed = self._count_changed_files(baseline_code_dir, sandbox_code_dir)
                if changed == 0:
                    logger.warning(
                        "[Engineer] Implementation %s for %s produced 0 file changes",
                        attempt_label, exp_name,
                    )
                    return False
                logger.info(
                    "[Engineer] Implementation OK for %s — %d file(s) modified (%s)",
                    exp_name, changed, attempt_label,
                )
            return True

        base_prompt = self._build_implement_prompt(proposal, sandbox_code_dir)
        if _invoke(base_prompt, "attempt1"):
            return True

        # Retry once with explicit emphasis on actually editing files.
        retry_prompt = (
            base_prompt
            + "\n\n## RETRY NOTICE\n"
            "Your previous attempt produced NO file modifications. You MUST use the "
            "Edit/Write tools to actually change at least one of the existing .py "
            "files in this sandbox. Read the file first, then make the concrete code "
            "edits described above. Do not respond with prose only.\n"
        )
        return _invoke(retry_prompt, "retry")

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
                               num_proposals: int,
                               parent_experiment: Optional[Dict] = None,
                               parent_retrospective_text: Optional[str] = None) -> str:
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

        # Build parent experiment section
        parent_section = ""
        if parent_experiment:
            per_action_text = ""
            if parent_experiment.get("per_action_results"):
                for k, v in sorted(parent_experiment["per_action_results"].items()):
                    if "_score_mae" in k and isinstance(v, (int, float)):
                        action = k.replace("test_", "").replace("_score_mae", "")
                        per_action_text += f"  - {action}: {v:.4f}\n"
            parent_section = f"""
## PARENT EXPERIMENT (your code starts from this, NOT the original baseline)
Your code sandbox will be initialized from this experiment's code, not the original baseline.
All code changes from the parent are already present in the files you will modify.
- Name: {parent_experiment.get('exp_name', 'N/A')}
- MAE: {parent_experiment.get('avg_score_mae', 'N/A')}
- Change type: {parent_experiment.get('code_change_type', 'N/A')}
- Code diff from baseline: {parent_experiment.get('code_diff_summary', 'N/A')}
- Rationale: {(parent_experiment.get('proposal_rationale', '') or '')[:200]}
{f'- Per-action results:{chr(10)}{per_action_text}' if per_action_text else ''}
IMPORTANT: You are proposing INCREMENTAL improvements on top of this parent's code.
Do NOT re-implement changes that the parent already has. Focus on what to change NEXT.
Analyze the parent's per-action results to identify which actions need the most improvement.
"""
            # NEW (2026-05-14): if the Judge has written retrospectives about
            # this parent (or its recent siblings/children), surface them here
            # so the Engineer's proposal explicitly takes prior failure modes
            # into account.
            if parent_retrospective_text:
                parent_section += (
                    "\n### Retrospectives on this parent and recent runs (Judge-written)\n"
                    f"{parent_retrospective_text}\n"
                    "Take these into account: do not repeat a documented failure mode "
                    "of this parent. Each proposal MUST address at least one specific "
                    "learning above (cite it briefly in the rationale field).\n"
                )

        prompt = f"""You are an ML research engineer developing a facial palsy severity prediction model.
This is MODEL RESEARCH & DEVELOPMENT — not just hyperparameter tuning.

## Current Status
- Current best Avg Score MAE: {current_best_mae:.4f} (target: 0.49, lower is better)
- Iteration: {iteration}
{parent_section}
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

## CODE INHERITANCE NOTICE
The files in your sandbox may already contain modifications from a parent experiment.
Do NOT revert them to the original baseline. Build ON TOP of what's already there.
Read each file FIRST to understand what changes are already present.

## CRITICAL RULES
1. Read each file BEFORE modifying it to understand the full context (including any parent changes)
2. Make ONLY the changes described above — no extra modifications
3. NEVER modify utils.py
4. Preserve the CLI interface: --data_dir, --output_dir, --exp, --epochs, --no_wandb
5. Preserve test_metrics.json output format (must contain *_score_mae keys)
6. Add proper imports for any new dependencies
7. Keep changes minimal and focused
8. NEVER change argparse choices in train.py — integrate new features by modifying the training loop code directly
9. If adding a new loss, add it in losses.py and call it from within main() in train.py
10. The script must still run with: python train.py --severity_loss_fn MSE+WK --model_type baseline --head_type feature_fusion (etc.)
11. DO NOT CREATE NEW .py FILES. The sandbox has ONLY: train.py, model.py, losses.py, dataset.py, backbone.py, utils.py. Any other .py file will be deleted and the experiment will fail validation.
12. There is NO cloc_loss.py, NO train_cloc_v2.py — do not import from them or create them

## KNOWN FAILURE PATTERNS (MUST AVOID — THESE HAVE ALL CAUSED REAL CRASHES)
- **LoRA: FAILED 8 CONSECUTIVE TIMES (iter 3-9). DO NOT attempt LoRA unless you use the EXACT pattern below.**
  - 5 crashes: Replaced attn.forward → TypeError: `need_weights` not accepted
  - 3 crashes: LoRALinear missing `weight` attribute → AttributeError in PyTorch MHA internals
  - If you implement LoRA, you MUST copy the EXACT LoRALinear class below. ANY deviation WILL crash.
- **New files**: NEVER create verify_syntax.py, test_*.py, or any helper scripts. Only the 6 allowed files exist.
- **ARFP / Region-Aware Feature Pyramid**: CATASTROPHICALLY FAILED 3 times (0.84→0.87→1.03). NEVER implement.
- **head_type=linear**: NEVER use. Always use head_type=feature_fusion.
- **batch_size>12**: Causes OOM on 24GB GPU. Always use batch_size≤12.
- **Calling methods that don't exist**: NEVER call `encoder.forward_patch_features()` or any method not in the original backbone.py. If you need a new method, ADD it to backbone.py first.
- **Python float in torch.exp()**: Always convert to tensor first. `torch.exp(torch.tensor(x))` not `torch.exp(x)` when x is a Python float.

## CORRECT LoRA IMPLEMENTATION PATTERN (for backbone.py)
The FaRL backbone uses CLIP's ResidualAttentionBlock with nn.MultiheadAttention.
CLIP calls: `self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)`
PyTorch MHA internally accesses `self.out_proj.weight` and `self.out_proj.bias`.

**ABSOLUTE RULES — VIOLATING ANY ONE WILL CRASH:**
1. NEVER modify `attn.forward` — CLIP's calling convention passes `need_weights=False`
2. NEVER split `attn.in_proj_weight` into separate q/k/v projections
3. NEVER create a custom forward that doesn't accept `**kwargs`
4. ONLY wrap `attn.out_proj` (a plain nn.Linear) — the ONLY safe injection point
5. LoRALinear MUST expose `.weight` and `.bias` properties (PyTorch accesses them internally)

```python
class LoRALinear(nn.Module):
    \"\"\"Drop-in replacement for nn.Linear with LoRA. Exposes .weight/.bias for PyTorch MHA compatibility.\"\"\"
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        in_f, out_f = original_linear.in_features, original_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(in_f, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        self.scale = alpha / rank
        original_linear.weight.requires_grad_(False)
        if original_linear.bias is not None:
            original_linear.bias.requires_grad_(False)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    @property
    def in_features(self):
        return self.original.in_features

    @property
    def out_features(self):
        return self.original.out_features

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scale

# ONLY safe injection — wrap out_proj in late blocks:
for i in range(8, 12):
    block = self.net.transformer.resblocks[i]
    block.attn.out_proj = LoRALinear(block.attn.out_proj, rank=8, alpha=16)
# That's it. Do NOT touch in_proj_weight, attn.forward, or create separate q/k/v projections.
```

## PER-ACTION LOSS STRATEGY (for train.py + losses.py)
Different actions respond best to different losses (proven by experiments):
- 이마_주름: RFLOS focal loss reduces MAE from 0.93→0.74 (biggest improvement)
- 눈_질끈감기, 입_우: VOR uncertainty modeling reduces MAE by ~0.10
- 눈_살짝감기, 입_이: MSE+WK already effective
Consider implementing ACTION-CONDITIONAL loss that applies different loss weights or loss types per action_idx.

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
        proposals = parse_json_response(response, expect_array=True)
        if proposals is None:
            logger.warning("[Engineer] Could not parse proposals JSON")
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
