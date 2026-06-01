"""
Agent 3: Judge
Analyzes completed experiments, generates insights, and sets direction.
"""

from typing import Any, Dict, List, Optional

from ..utils.logging_setup import get_logger
from .base_agent import call_claude_cli, parse_json_response

logger = get_logger(__name__)


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
              iteration: int,
              vlm_baseline: Optional[str] = None,
              parent_comparisons: Optional[str] = None,
              stagnation_diagnosis: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
            current_best_mae, iteration, vlm_baseline,
            parent_comparisons, stagnation_diagnosis,
        )

        logger.info("[Judge] Analyzing results...")
        response = call_claude_cli(
            prompt=prompt,
            model=self.model,
            max_budget_usd=self.max_budget_usd,
            timeout_seconds=600,
            tools="",  # No tools — pure reasoning
            agent="judge",
        )

        if not response:
            logger.warning("[Judge] No response")
            return None

        return self._parse_response(str(response))

    def _build_prompt(self, completed_experiments: str,
                       latest_experiments: str,
                       existing_insights: str,
                       research_context: Optional[Dict],
                       current_best_mae: float,
                       iteration: int,
                       vlm_baseline: Optional[str] = None,
                       parent_comparisons: Optional[str] = None,
                       stagnation_diagnosis: Optional[str] = None) -> str:
        research_section = ""
        # research_context is expected to be a dict; tolerate a stray str/None.
        if isinstance(research_context, dict) and research_context.get("techniques"):
            research_section = "\n## Available Research Techniques\n"
            for t in research_context["techniques"]:
                research_section += f"- {t['name']}: {t.get('core_idea', 'N/A')}\n"
        elif isinstance(research_context, str) and research_context.strip():
            research_section = "\n## Available Research Techniques\n" + research_context + "\n"

        stagnation_section = ""
        if stagnation_diagnosis and stagnation_diagnosis.strip():
            stagnation_section = (
                "\n## Stagnation Diagnosis (auto-computed — read this FIRST)\n"
                + stagnation_diagnosis + "\n"
            )

        parent_section = ""
        if parent_comparisons:
            parent_section = (
                "\n## Parent-vs-Child Comparison (Iterative Refinement Tracking)\n"
                "Each experiment was built incrementally on a parent experiment code.\n"
                "Here is how the incremental changes affected performance:\n"
                + parent_comparisons + "\n\n"
                "KEY QUESTION: Did the incremental changes help or hurt?\n"
                "What specific code modifications caused improvement or regression?\n"
                "Use this to guide the next iteration direction.\n"
            )

        prompt = f"""You are the lead researcher analyzing experiment results for facial palsy severity prediction.

## Current Status
- Iteration: {iteration}
- Current best Avg Score MAE: {current_best_mae:.4f} (target: 0.49, lower is better)
- Gap to target: {current_best_mae - 0.49:.4f}
{stagnation_section}
## Top 15 Completed Experiments (by performance)
{completed_experiments}

## Latest Iteration Results
{latest_experiments}
{parent_section}
## Accumulated Research Insights
{existing_insights}
{research_section}
## VLM Auto-Labeling Baseline
{vlm_baseline or "Not yet evaluated"}
(VLM = Vision-Language Model that directly predicts severity from images via API. Useful as reference.)

## Task — perform these FOUR steps in order

### Step 1. Per-experiment retrospective (NEW — required before anything else)
For every experiment in "Latest Iteration Results" above, write a short retrospective:
why it succeeded or failed, what specific code change drove the outcome, and what
this teaches us about the search direction. Be concrete (cite numbers, action MAEs,
parent comparison). Do NOT skip experiments — every latest experiment must have one
entry, even if just one sentence.

### Step 2. Insight re-examination (NEW — required before generating new insights)
Read the "Accumulated Research Insights" block above. For the current iteration:
  (a) Pick the 3–5 PAST insights that are most relevant to today's results.
      Quote each insight briefly and explain why it matters NOW.
  (b) For each picked insight, state whether the latest results CONFIRM it,
      CONTRADICT it, or REFINE it. If contradicted, propose a replacement.
  (c) If any past insight was clearly wrong or outdated, mark it for retirement.

Self-critique: the system has previously documented insights but failed to act on
them. The point of this step is to make insight use VISIBLE in the decision —
the next_direction below MUST reference at least one specific picked insight.

### Step 3. Generate new insights
What additional patterns emerge from the latest results that are not already
covered by the picked-and-confirmed insights? Only add genuinely new ones.

### Step 4. Decide direction (must cite picked insights from Step 2)
Choose exploit vs explore and give engineer/researcher guidance. In the
`engineer_guidance` field you MUST reference at least one specific picked
insight ID by quoting its key phrase, so the chain of reasoning is auditable.

## Things to consider
- Which actions have the highest MAE? Why might that be?
- Are there diminishing returns on hyperparameter tuning?
- Should we try code-level changes (architecture, loss, augmentation)?
- What haven't we tried yet that might help?
- Has the same parent been reused too often? Is the system stuck on one bottleneck action?

## Output Format
Return ONLY this JSON (note the new `retrospectives` and `insight_reexamination` fields):
{{
  "retrospectives": [
    {{
      "exp_name": "exp_X_name",
      "outcome": "improved | regressed | neutral",
      "delta_mae_vs_parent": -0.0123,
      "why": "1-3 sentence concrete cause analysis",
      "learning": "what this teaches about the search direction"
    }}
  ],
  "insight_reexamination": [
    {{
      "insight_quote": "short quote of the past insight body",
      "status": "confirmed | contradicted | refined | retire",
      "applies_now_because": "why it matters for the current iteration",
      "implication": "what should change in the next experiment as a result"
    }}
  ],
  "analysis": "Synthesis paragraph (2-3 short paragraphs) — what is the system actually learning?",
  "new_insights": [
    {{
      "observation": "What we learned that was NOT already in the picked insights",
      "category": "architecture|loss|augmentation|training|hyperparameter|general",
      "parameter_affected": "specific param or null",
      "recommendation": "What to do based on this",
      "confidence": 0.7
    }}
  ],
  "next_direction": {{
    "strategy": "exploit | explore",
    "researcher_guidance": "What the researcher should search for (null if not needed)",
    "engineer_guidance": "Specific instructions for the next experiment, REFERENCING at least one picked-insight quote from Step 2",
    "focus_areas": ["specific areas to focus on"],
    "cited_insights": ["short quotes of insights driving this direction"]
  }}
}}"""
        return prompt

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        data = parse_json_response(response)
        if data is None:
            logger.warning("[Judge] Could not parse JSON from response")
            return None
        if "analysis" not in data and "next_direction" not in data:
            logger.warning("[Judge] JSON missing required keys (analysis/next_direction)")
            return None
        return data
