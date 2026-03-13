"""
Agent 1: Researcher
Searches for papers and techniques, extracts implementable ideas.
"""

import json
import re
from typing import Any, Dict, List, Optional

from .base_agent import call_claude_cli


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

        print("  [Researcher] Searching for techniques...")
        response = call_claude_cli(
            prompt=prompt,
            model=self.model,
            max_budget_usd=self.max_budget_usd,
            timeout_seconds=300,
            tools=None,  # Enable all tools (web search)
            system_prompt=self._system_prompt(),
        )

        if not response:
            print("  [Researcher] No response")
            return None

        return self._parse_response(response)

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
- Loss: MSE + Weighted Kappa + CLOC contrastive ordinal loss
- Current best Avg Score MAE: {current_best_mae:.4f} (target: 0.49)
- Weak actions (high MAE): {', '.join(weak_actions)}
- Dataset: ~600 facial palsy patient images, 6 facial actions, severity 0-5

## Base Architecture
- backbone.py: FaRL ViT-B/16 encoder (frozen or fine-tuned last 2 layers)
- model.py: SeparateHeadWithFeatures — independent MLP heads per action
- losses.py: MSE + WeightedKappa + optional CrossEntropy
- cloc_loss.py: CLOC (Contrastive Learning for Ordinal Classification, CVPR 2025)
- dataset.py: Image loading, augmentation, severity labels
- train_cloc_v2.py: Training loop with validation

## Files That Can Be Modified
- model.py (architecture changes)
- losses.py (new loss functions)
- cloc_loss.py (ordinal loss improvements)
- dataset.py (augmentation, preprocessing)
- backbone.py (backbone modifications)
- train_cloc_v2.py (training strategy)
- utils.py is READ-ONLY (metric computation)

## Existing Research Insights
{existing_insights}

## Techniques Already Tried
{techniques_tried}

{f"## Judge's Guidance for This Round{chr(10)}{judge_guidance}" if judge_guidance else ""}

## Task
Search for 2-3 techniques from recent papers (2023-2026) that could help improve the model.
Focus on:
1. Ordinal regression improvements
2. Class imbalance / hard sample mining for medical images
3. Attention mechanisms for facial action unit analysis
4. Better loss functions for ordinal severity prediction
5. Data augmentation strategies for small medical datasets

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
        text = response.strip()
        # Try to find JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            print("  [Researcher] Could not parse JSON from response")
            return None

        try:
            data = json.loads(match.group())
            if "techniques" in data:
                return data
        except json.JSONDecodeError:
            pass

        print("  [Researcher] Invalid JSON in response")
        return None
