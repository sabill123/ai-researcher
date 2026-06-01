"""
Agent 1: Researcher
Searches for papers and techniques, extracts implementable ideas.
"""

from typing import Any, Dict, List, Optional

from ..utils.logging_setup import get_logger
from .base_agent import call_claude_cli, parse_json_response

logger = get_logger(__name__)


class ResearcherAgent:
    """Uses Claude CLI with web search to find relevant techniques."""

    def __init__(self, model: str = "sonnet", max_budget_usd: float = 0.50):
        self.model = model
        self.max_budget_usd = max_budget_usd

    def research(self, current_best_mae: float,
                 weak_actions: List[str],
                 judge_guidance: Optional[str],
                 existing_insights: str,
                 completed_techniques: List[str]) -> Optional[Dict[str, Any]]:
        """Search for new techniques to try.

        Returns dict with:
            techniques: [{name, paper, core_idea, implementation_sketch, target_files, expected_impact}]
            recommended_priority: str
        """
        prompt = self._build_prompt(
            current_best_mae, weak_actions, judge_guidance,
            existing_insights, completed_techniques,
        )

        logger.info("[Researcher] Searching for techniques...")
        response = call_claude_cli(
            prompt=prompt,
            model=self.model,
            max_budget_usd=self.max_budget_usd,
            # 2026-05-31: 300s timeout caused ~400 timeouts over recent
            # ~500-iter run. Researcher does web search + LLM analysis which
            # can take 5-10 min. Raised to 900s.
            timeout_seconds=900,
            tools=None,  # Enable all tools (web search)
            system_prompt=self._system_prompt(),
            agent="researcher",
        )

        if not response:
            logger.warning("[Researcher] No response")
            return None

        return self._parse_response(str(response))

    def _system_prompt(self) -> str:
        return """You are a research assistant for facial palsy severity prediction using deep learning.
Your task is to search for papers and techniques that could improve the model.

## SAFETY CONSTRAINTS
- You are on a SHARED GPU SERVER.
- ONLY output research findings as JSON.
- NEVER suggest system commands, file deletions, or process management.
- ALLOWED PATH: /home/jaeseokhan/2025-02/khu/** ONLY

## OUTPUT FORMAT
Return ONLY a JSON object with the structure specified in the prompt.
No other text before or after the JSON."""

    def _build_prompt(self, current_best_mae: float,
                      weak_actions: List[str],
                      judge_guidance: Optional[str],
                      existing_insights: str,
                      completed_techniques: List[str]) -> str:
        techniques_tried = "\n".join(f"  - {t}" for t in completed_techniques) if completed_techniques else "  None yet"

        prompt = f"""## Current Status
- Task: Facial palsy severity prediction (ordinal regression, 0-5 scale)
- Model: FaRL ViT-B/16 (face-specific CLIP encoder) + per-action classification heads
- Loss: MSE + Weighted Kappa (SeverityLossCombiner)
- Current best Avg Score MAE: {current_best_mae:.4f} (target: 0.49)
- Weak actions (high MAE): {', '.join(weak_actions)}
- Dataset: ~8000+ facial palsy patient images, 6 facial actions (눈_살짝감기, 눈_질끈감기, 이마_주름, 입_이, 입_우, 안면_무표정), severity 0-4

## Base Architecture
- backbone.py: FaRL ViT-B/16 encoder (pretrained on 20M faces, intermediate layers [3,5,7,11])
- model.py: BaselinePalsyClassifier — FaRL backbone + separate per-action MLP severity heads + action classifier
- losses.py: MSELoss, WeightedKappaLoss, XEntropyLoss (hard/soft SORD), SeverityLossCombiner
- dataset.py: KHUPalsyDataset (image loading, augmentation, severity labels for 6 actions)
- train.py: Training loop (100 epochs, AdamW, CosineAnnealing)

## Files That Can Be Modified
- model.py (architecture changes)
- losses.py (new loss functions)
- dataset.py (augmentation, preprocessing)
- backbone.py (backbone modifications)
- train.py (training strategy)
- utils.py is READ-ONLY (metric computation)

## Existing Research Insights
{existing_insights}

## Techniques Already Tried
{techniques_tried}

{f"## Judge's Guidance for This Round{chr(10)}{judge_guidance}" if judge_guidance else ""}

## Task
Search for 2-3 techniques from recent papers (2023-2026) that could help improve the model.
This is a MODEL DEVELOPMENT project — focus on ARCHITECTURE and BACKBONE improvements, not just loss/augmentation changes.

Focus on (in priority order):
1. **ViT backbone adaptation for medical/face images**: LoRA, adapter modules, visual prompt tuning, side-tuning for pretrained ViT models
2. **Spatial attention / region-aware architectures for face analysis**: attention mechanisms that focus on specific facial regions (eyes, mouth, forehead) based on action type
3. **Multi-scale feature extraction from Vision Transformers**: feature pyramid networks on ViT intermediate layers, progressive feature aggregation across transformer blocks
4. **Action-conditioned / task-conditioned architectures**: models that use different processing paths or attention patterns for different facial actions
5. **Advanced classification heads**: transformer decoder heads for ordinal regression, cross-attention heads, graph-based heads for spatial reasoning
6. **Fine-tuning strategies for pretrained ViT**: discriminative learning rates, gradual unfreezing, intermediate layer distillation

Lower priority (only 1 technique can be non-architecture):
7. Ordinal regression loss improvements
8. Face-specific data augmentation

## Output Format
Return ONLY this JSON:
{{
  "techniques": [
    {{
      "name": "Short technique name",
      "paper": "Paper title and year",
      "core_idea": "1-2 sentence description of the core idea",
      "implementation_sketch": "How to implement this in the existing codebase (which files to modify, what to add)",
      "target_files": ["model.py", "losses.py"],
      "expected_impact": "Why this might help and estimated MAE improvement"
    }}
  ],
  "recommended_priority": "Which technique to try first and why"
}}"""
        return prompt

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        data = parse_json_response(response, expected_keys=("techniques",))
        if data is None:
            logger.warning("[Researcher] Could not parse JSON from response")
            return None
        return data
