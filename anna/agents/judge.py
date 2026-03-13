"""
Agent 3: Judge
Analyzes completed experiments, generates insights, and sets direction.
"""

import json
import re
from typing import Any, Dict, List, Optional

from .base_agent import call_claude_cli


class JudgeAgent:
    """Analyzes results and decides next research direction."""

    def __init__(self, model: str = "sonnet", max_budget_usd: float = 0.75):
        self.model = model
        self.max_budget_usd = max_budget_usd

    def judge(self, completed_experiments: str,
              latest_experiments: str,
              existing_insights: str,
              research_context: Optional[Dict],
              current_best_mae: float,
              iteration: int) -> Optional[Dict[str, Any]]:
        """Analyze results and decide next direction.

        Returns dict with:
            analysis: str (detailed analysis)
            new_insights: [{observation, category, parameter_affected, recommendation, confidence}]
            next_direction: {
                strategy: "exploit" | "explore",
                researcher_guidance: str | null,
                engineer_guidance: str,
                focus_areas: [str],
            }
        """
        prompt = self._build_prompt(
            completed_experiments, latest_experiments,
            existing_insights, research_context,
            current_best_mae, iteration,
        )

        print("  [Judge] Analyzing results...")
        response = call_claude_cli(
            prompt=prompt,
            model=self.model,
            max_budget_usd=self.max_budget_usd,
            timeout_seconds=180,
            tools="",  # No tools — pure reasoning
        )

        if not response:
            print("  [Judge] No response")
            return None

        return self._parse_response(response)

    def _build_prompt(self, completed_experiments: str,
                       latest_experiments: str,
                       existing_insights: str,
                       research_context: Optional[Dict],
                       current_best_mae: float,
                       iteration: int) -> str:
        research_section = ""
        if research_context and research_context.get("techniques"):
            research_section = "\n## Available Research Techniques\n"
            for t in research_context["techniques"]:
                research_section += f"- {t['name']}: {t.get('core_idea', 'N/A')}\n"

        prompt = f"""You are the lead researcher analyzing experiment results for facial palsy severity prediction.

## Current Status
- Iteration: {iteration}
- Current best Avg Score MAE: {current_best_mae:.4f} (target: 0.49, lower is better)
- Gap to target: {current_best_mae - 0.49:.4f}

## Top 15 Completed Experiments (by performance)
{completed_experiments}

## Latest Iteration Results
{latest_experiments}

## Accumulated Research Insights
{existing_insights}
{research_section}

## Task
1. **Analyze**: What patterns do you see? What's working, what's not?
2. **Generate Insights**: What new insights can we extract from the latest results?
3. **Decide Direction**: Should we exploit (refine what works) or explore (try new approaches)?

Consider:
- Which actions have the highest MAE? Why might that be?
- Are there diminishing returns on hyperparameter tuning?
- Should we try code-level changes (architecture, loss, augmentation)?
- What haven't we tried yet that might help?

## Output Format
Return ONLY this JSON:
{{
  "analysis": "Detailed analysis of current state and latest results (2-3 paragraphs)",
  "new_insights": [
    {{
      "observation": "What we learned",
      "category": "architecture|loss|augmentation|training|hyperparameter|general",
      "parameter_affected": "specific param or null",
      "recommendation": "What to do based on this",
      "confidence": 0.7
    }}
  ],
  "next_direction": {{
    "strategy": "exploit or explore",
    "researcher_guidance": "What the researcher should search for (null if not needed)",
    "engineer_guidance": "Specific instructions for the engineer's next experiments",
    "focus_areas": ["specific areas to focus on"]
  }}
}}"""
        return prompt

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        text = response.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            print("  [Judge] Could not parse JSON from response")
            return None

        try:
            data = json.loads(match.group())
            if "analysis" in data or "next_direction" in data:
                return data
        except json.JSONDecodeError:
            pass

        print("  [Judge] Invalid JSON in response")
        return None
