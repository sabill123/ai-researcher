"""
Reflective memory: accumulates insights from experiment comparisons.
Extended from v1 with code-change insight categories.
Inspired by MARS paper's Comparative Reflective Memory.

2026-06-01 hardening (module fix #2):
* Persistence routes through utils.atomic_write (tmp+fsync+rename) and
  safe_load_json (corrupt files quarantined to <path>.corrupted.<ts>
  before booting an empty memory) so a partial write or torn JSON can
  never wedge the whole orchestrator on restart.
* prune() runs at most once per Judge batch, with a weighted score
  (cited_count*2 + confidence) instead of raw confidence, and grants a
  30-minute grace window to brand-new insights so a single noisy batch
  cannot evict a Judge's own freshly-emitted lesson.
* add_insight() now auto-fills evidence with the caller's exp_name when
  the Judge omits it, which keeps the provenance chain usable for the
  re-examination step.
* apply_insight_reexamination() matches each Judge quote against the
  single insight whose observation contains the longest substring of the
  quote, instead of fanning the update out to every insight that happens
  to contain the substring. Stops noisy double-counting of cited_count.
* Retrospective cap raised to 200; before truncation we always keep one
  entry per Top-20 most-frequent parent_exp_name so the Engineer never
  loses the retrospective of a still-live candidate parent.
* All print() calls replaced with module-scoped logger from
  utils.logging_setup (still goes to console + rotating file handler).

Public API (ReflectiveMemory.{bootstrap, add_insight,
add_insights_from_judge, add_retrospectives_from_judge,
apply_insight_reexamination, get_retrospective, prune, get_insights,
format_for_llm, format_retrospectives_for_llm,
get_completed_techniques}) and the Insight / Retrospective dataclasses
are unchanged so orchestrator.py needs no edits.
"""

from __future__ import annotations

import os
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Optional

from .utils.atomic_write import atomic_write_json, safe_load_json
from .utils.logging_setup import get_logger

logger = get_logger(__name__)

# Grace window: insights created within this many seconds are exempt
# from prune() eviction. Stops the common case where the Judge emits a
# fresh insight that prune() immediately drops because cited_count=0.
_PRUNE_GRACE_SECONDS = 30 * 60

# Retrospective storage cap (raised from 100 → 200 in this refit so we
# can keep both recent-by-time and parent-frequency-preserved entries).
_RETRO_CAP = 200

# How many of the most-frequent parent_exp_names must always survive a
# retrospective truncation. Engineer reads parent retrospectives, so
# losing them silently is a regression risk worse than file bloat.
_RETRO_PARENT_TOPK = 20


@dataclass
class Insight:
    id: str = ""
    observation: str = ""
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5
    category: str = "general"  # architecture, hyperparameter, loss, training, augmentation, code_change
    parameter_affected: Optional[str] = None
    recommendation: str = ""
    created_at: str = ""
    last_updated: str = ""
    # Added 2026-05-14 — track how often Judge has explicitly cited this insight
    # in subsequent iterations and how many times it has been confirmed /
    # contradicted by later results. Makes the "documented but ignored" problem
    # visible: if cited_count stays 0 over many iterations the insight is
    # functionally dead and prune() will drop it first.
    cited_count: int = 0
    confirmed_count: int = 0
    contradicted_count: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.last_updated:
            self.last_updated = self.created_at


@dataclass
class Retrospective:
    """Per-experiment retrospective written by Judge after each iteration.

    Kept separate from Insight: a retrospective is anchored to ONE experiment
    and describes why it succeeded or failed, while an Insight is a
    generalised lesson. The Engineer reads the retrospective of a candidate
    parent before proposing a child, so the parent's known failure modes
    can inform the next step.
    """
    id: str = ""
    exp_name: str = ""
    outcome: str = "neutral"   # improved | regressed | neutral
    delta_mae_vs_parent: Optional[float] = None
    why: str = ""
    learning: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


BOOTSTRAP_INSIGHTS = [
    Insight(
        observation="Separate heads significantly outperform shared heads (0.656 vs 0.80+)",
        evidence=["separate_cloc_high_margin", "margin_2.0_1.5_1.0_0.5"],
        confidence=0.95,
        category="architecture",
        parameter_affected="model_type",
        recommendation="Always use model_type='separate'",
    ),
    Insight(
        observation="CLOC contrastive loss improves performance over baseline (0.656 vs 0.67)",
        evidence=["separate_cloc_high_margin", "separate_baseline"],
        confidence=0.8,
        category="loss",
        parameter_affected="use_cloc",
        recommendation="Keep use_cloc=True",
    ),
    Insight(
        observation="MSE+WK loss outperforms CE loss (0.656 vs 0.722)",
        evidence=["separate_cloc_high_margin", "separate_cloc_CE"],
        confidence=0.85,
        category="loss",
        parameter_affected="severity_loss_fn",
        recommendation="Use MSE+WK, not CE",
    ),
    Insight(
        observation="CLOC weight 0.3 and 0.5 perform equally well (both ~0.656)",
        evidence=["separate_cloc_high_margin", "separate_cloc_w0.3"],
        confidence=0.7,
        category="hyperparameter",
        parameter_affected="cloc_weight",
        recommendation="Explore values around 0.3-0.7",
    ),
    Insight(
        observation="Margins [2.0, 1.0, 0.8, 0.6] are best tested preset",
        evidence=["separate_cloc_high_margin"],
        confidence=0.6,
        category="hyperparameter",
        parameter_affected="initial_margins",
        recommendation="Explore variations around [2.0, 1.0, 0.8, 0.6]",
    ),
    Insight(
        observation="입_오 (mouth O) consistently has worst per-action MAE (~1.0)",
        evidence=["separate_cloc_high_margin", "separate_baseline"],
        confidence=0.9,
        category="general",
        recommendation="Focus optimization on improving 입_오 performance",
    ),
    Insight(
        observation="Hyperparameter tuning alone has plateaued around 0.63 MAE — code-level changes needed",
        evidence=["bootstrap_analysis"],
        confidence=0.7,
        category="general",
        recommendation="Try architecture changes, new loss functions, or augmentation strategies",
    ),
]


def _parse_iso(ts: str) -> Optional[datetime]:
    """Best-effort ISO-8601 parse. Returns None on failure (insight stays
    eligible for prune in that case, which is the safe fallback)."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _longest_common_substring_len(a: str, b: str) -> int:
    """Length of the longest substring shared by ``a`` and ``b``.

    Used by apply_insight_reexamination to pick the SINGLE insight that
    best matches a Judge quote, rather than fanning the update out to
    every insight whose observation happens to contain the quote. O(n*m)
    DP — fine for the short strings we deal with (insight observations
    are typically <300 chars and we only run this once per Judge batch).
    """
    if not a or not b:
        return 0
    a = a.lower()
    b = b.lower()
    # Rolling rows: only previous row needed.
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best


class ReflectiveMemory:
    def __init__(self, memory_path: str):
        self.memory_path = memory_path
        self.insights: List[Insight] = []
        self.retrospectives: List[Retrospective] = []
        # Persistent per-exp_name appearance counter. We dedup
        # retrospectives by exp_name, so an in-list Counter would always
        # show frequency=1; preserving Top-K "by frequency" instead means
        # preserving the exp_names that show up MOST OFTEN as parents
        # over the whole orchestrator history. Persisted next to the
        # retrospectives file as a sidecar.
        self._exp_frequency: Counter = Counter()
        self._load()

    def _retrospective_path(self) -> str:
        # Stored next to memory_path, e.g. reflective_memory.json → retrospectives.json
        return os.path.join(os.path.dirname(self.memory_path), "retrospectives.json")

    def _exp_frequency_path(self) -> str:
        return os.path.join(
            os.path.dirname(self.memory_path), "retro_exp_frequency.json"
        )

    def _load(self):
        """Crash-safe load. Corrupt JSON is quarantined to
        ``<path>.corrupted.<unix_ts>`` and we boot with an empty list so
        the orchestrator can keep running instead of crash-looping."""
        raw_insights = safe_load_json(self.memory_path, default=[])
        self.insights = []
        for d in raw_insights:
            if not isinstance(d, dict):
                continue
            # Tolerate old-format entries that lack the new tracking fields
            d.setdefault("cited_count", 0)
            d.setdefault("confirmed_count", 0)
            d.setdefault("contradicted_count", 0)
            try:
                self.insights.append(Insight(**d))
            except TypeError as exc:
                # Unknown/extra keys on an Insight record — skip this row
                # rather than poison the whole load. Log so we notice if
                # the schema actually drifted.
                logger.warning(
                    "Skipping malformed insight row in %s: %s",
                    self.memory_path, exc,
                )

        rp = self._retrospective_path()
        raw_retros = safe_load_json(rp, default=[])
        self.retrospectives = []
        for d in raw_retros:
            if not isinstance(d, dict):
                continue
            try:
                self.retrospectives.append(Retrospective(**d))
            except TypeError as exc:
                logger.warning(
                    "Skipping malformed retrospective row in %s: %s",
                    rp, exc,
                )

        # Persistent exp_name frequency for retrospective preservation.
        # Defaults to an empty dict; if missing or corrupt we just lose
        # historical frequency until new retros arrive (acceptable
        # degradation — the retrospectives themselves stay intact).
        raw_freq = safe_load_json(self._exp_frequency_path(), default={})
        if isinstance(raw_freq, dict):
            try:
                self._exp_frequency = Counter(
                    {str(k): int(v) for k, v in raw_freq.items()}
                )
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping malformed exp_frequency file: %s", exc,
                )
                self._exp_frequency = Counter()

    def _save(self):
        """Persist both insight and retrospective files atomically.

        atomic_write_json writes to ``<path>.tmp`` then fsync+rename, so a
        crash mid-write leaves either the old version or the new version
        on disk — never a half-written file that safe_load_json would
        have to quarantine on next boot."""
        atomic_write_json(
            self.memory_path,
            [asdict(i) for i in self.insights],
        )
        atomic_write_json(
            self._retrospective_path(),
            [asdict(r) for r in self.retrospectives],
        )
        atomic_write_json(
            self._exp_frequency_path(),
            dict(self._exp_frequency),
        )

    def bootstrap(self):
        if self.insights:
            return
        for insight in BOOTSTRAP_INSIGHTS:
            self.add_insight(insight)

    def add_insight(
        self,
        insight: Insight,
        *,
        caller_exp_name: Optional[str] = None,
    ):
        """Add or merge an insight.

        ``caller_exp_name`` is the experiment that motivated this insight.
        If the caller didn't pre-populate ``insight.evidence`` we use
        ``[caller_exp_name]`` so the provenance chain stays intact for
        downstream re-examination. If both are absent we leave evidence
        empty (existing behaviour).
        """
        if not insight.evidence and caller_exp_name:
            insight.evidence = [caller_exp_name]

        existing = self._find_similar(insight)
        if existing:
            for eid in insight.evidence:
                if eid not in existing.evidence:
                    existing.evidence.append(eid)
            existing.confidence = min(0.95, existing.confidence + 0.05)
            existing.last_updated = datetime.now().isoformat()
        else:
            self.insights.append(insight)
        self._save()

    def add_insights_from_judge(
        self,
        judge_insights: list,
        *,
        caller_exp_name: Optional[str] = None,
    ):
        """Add insights generated by the Judge agent.

        Pruning runs ONCE at the end of the batch (used to fire after each
        add_insight via the per-call path, which meant a 5-insight batch
        could evict its own earlier entries before the Judge finished
        emitting). The single trailing prune call uses the new weighted
        score and the 30-minute grace window so a batch's own insights
        cannot disappear before they get a chance to be cited.
        """
        added = 0
        for ji in judge_insights:
            if not isinstance(ji, dict):
                continue
            evidence = ji.get("evidence")
            if not isinstance(evidence, list):
                evidence = []
            insight = Insight(
                observation=ji.get("observation", ""),
                evidence=list(evidence),
                confidence=ji.get("confidence", 0.5),
                category=ji.get("category", "general"),
                parameter_affected=ji.get("parameter_affected"),
                recommendation=ji.get("recommendation", ""),
            )
            if insight.observation:
                self.add_insight(insight, caller_exp_name=caller_exp_name)
                added += 1
        if added:
            logger.info(
                "Memory: added %d insights from Judge (total=%d)",
                added, len(self.insights),
            )
        # Single prune pass at end of batch — see docstring for rationale.
        self.prune()

    def add_retrospectives_from_judge(self, judge_retrospectives: list):
        """Persist per-experiment retrospectives written by Judge.

        Called after each iteration. Each retrospective is anchored to one
        exp_name and contains why/learning text that the Engineer will read
        when that experiment becomes a candidate parent.
        """
        added = 0
        for r in judge_retrospectives:
            if not isinstance(r, dict):
                continue
            exp_name = r.get("exp_name", "").strip()
            if not exp_name:
                continue
            # Replace any prior retrospective for the same exp (Judge revises)
            self.retrospectives = [rr for rr in self.retrospectives
                                   if rr.exp_name != exp_name]
            self.retrospectives.append(Retrospective(
                exp_name=exp_name,
                outcome=r.get("outcome", "neutral"),
                delta_mae_vs_parent=r.get("delta_mae_vs_parent"),
                why=r.get("why", "")[:1000],
                learning=r.get("learning", "")[:600],
            ))
            # Bump the persistent appearance counter — this is what
            # drives Top-K preservation when we later truncate.
            self._exp_frequency[exp_name] += 1
            added += 1
        # Cap stored retrospectives, but always preserve at least one entry
        # for each of the Top-K most-frequent parent_exp_names so the
        # Engineer never loses the retrospective of a still-live parent.
        self._truncate_retrospectives()
        self._save()
        if added:
            logger.info(
                "Memory: saved %d per-experiment retrospectives "
                "(total stored: %d)",
                added, len(self.retrospectives),
            )

    def _truncate_retrospectives(self) -> None:
        """Keep up to ``_RETRO_CAP`` retrospectives. When pruning, retain
        the most recent ``_RETRO_CAP`` BUT also force-keep one entry per
        Top-``_RETRO_PARENT_TOPK`` ``exp_name`` ranked by HISTORICAL
        appearance frequency (``self._exp_frequency``). Frequency is the
        count of how many times the Judge has emitted a retrospective
        for that exp — a high count means the experiment is still a
        live candidate parent that the Engineer will want to inspect,
        so dropping its retrospective on cap overflow would silently
        regress proposal quality."""
        if len(self.retrospectives) <= _RETRO_CAP:
            return

        # Recent-by-position window we'd keep with naive truncation.
        recent = self.retrospectives[-_RETRO_CAP:]
        recent_names = {r.exp_name for r in recent}

        # Top-K by HISTORICAL frequency across the whole run, not by
        # in-list dedup frequency (which is always 1 since we replace
        # same-named entries on insert).
        top_names = [
            name for name, _ in self._exp_frequency.most_common(_RETRO_PARENT_TOPK)
        ]

        # Must-keep = Top-K names whose retrospective is not already in
        # `recent`. We look them up by exp_name in the full list so we
        # grab the freshest revision rather than an arbitrary one.
        by_name = {}
        for r in self.retrospectives:
            by_name[r.exp_name] = r  # later overwrites — keep newest

        must_keep: list = []
        for name in top_names:
            if name in recent_names:
                continue
            r = by_name.get(name)
            if r is not None:
                must_keep.append(r)

        if not must_keep:
            self.retrospectives = recent
            return

        # Merge: drop oldest recent entries to make room for must_keep,
        # so total length stays at _RETRO_CAP.
        keep_count = max(0, _RETRO_CAP - len(must_keep))
        merged = list(must_keep) + recent[-keep_count:]
        # Sort by created_at so chronological order is preserved on disk.
        merged.sort(key=lambda r: r.created_at or "")
        self.retrospectives = merged

    def apply_insight_reexamination(self, judge_reexam: list):
        """Update cited/confirmed/contradicted counts based on Judge's
        explicit re-examination of past insights.

        This is what turns insights from inert documentation into a feedback
        signal: an insight that is repeatedly cited gets prioritised, and one
        that keeps being contradicted gets retired by prune().

        Matching policy (changed 2026-06-01): for each Judge quote, pick the
        SINGLE insight whose observation shares the longest substring with
        the quote (ties broken by current confidence). Previously any
        observation that contained the quote as a substring would get
        updated, which double-counted cited_count whenever two insights
        overlapped on common phrasing like "MSE+WK".
        """
        if not judge_reexam:
            return
        for entry in judge_reexam:
            if not isinstance(entry, dict):
                continue
            quote = (entry.get("insight_quote") or "").strip()
            status = (entry.get("status") or "").lower()
            if not quote:
                continue

            # Score every insight by longest-common-substring length, keep
            # the single best match (>=4 chars to avoid noise matches).
            best_ins: Optional[Insight] = None
            best_len = 0
            for ins in self.insights:
                cur = _longest_common_substring_len(quote, ins.observation)
                if cur > best_len or (
                    cur == best_len and best_ins is not None
                    and ins.confidence > best_ins.confidence
                ):
                    best_len = cur
                    best_ins = ins
            if best_ins is None or best_len < 4:
                logger.debug(
                    "Re-exam quote %r had no usable insight match (best_len=%d)",
                    quote[:60], best_len,
                )
                continue

            best_ins.cited_count += 1
            if status == "confirmed":
                best_ins.confirmed_count += 1
                best_ins.confidence = min(0.99, best_ins.confidence + 0.02)
            elif status == "contradicted":
                best_ins.contradicted_count += 1
                best_ins.confidence = max(0.05, best_ins.confidence - 0.15)
            elif status == "retire":
                best_ins.confidence = max(0.05, best_ins.confidence - 0.25)
            best_ins.last_updated = datetime.now().isoformat()
        self._save()

    def get_retrospective(self, exp_name: str) -> Optional["Retrospective"]:
        """Look up the most recent retrospective for a given experiment."""
        for r in reversed(self.retrospectives):
            if r.exp_name == exp_name:
                return r
        return None

    def prune(self, max_insights: int = 30):
        """Keep only the top N insights, removing low-value entries.

        Score = cited_count * 2 + confidence. Cited counts are weighted
        more heavily than raw Judge-assigned confidence because cites are
        observable downstream behaviour ("the Judge actually used this")
        whereas confidence is the Judge's own self-report. Tie-break on
        evidence count to prefer better-attested entries.

        Insights created within the last ``_PRUNE_GRACE_SECONDS`` are
        exempt from eviction so a fresh Judge-emitted batch is not wiped
        out before it has had a chance to be cited.
        """
        if len(self.insights) <= max_insights:
            return

        cutoff = datetime.now() - timedelta(seconds=_PRUNE_GRACE_SECONDS)
        protected: list = []
        evictable: list = []
        for ins in self.insights:
            created = _parse_iso(ins.created_at)
            if created is not None and created >= cutoff:
                protected.append(ins)
            else:
                evictable.append(ins)

        evictable.sort(
            key=lambda i: (i.cited_count * 2 + i.confidence, len(i.evidence)),
            reverse=True,
        )

        slots = max_insights - len(protected)
        if slots <= 0:
            # Grace window alone exceeds the cap. Trust the grace policy
            # (these are all fresh) and bail without dropping anything;
            # the next prune cycle (once grace expires) will catch up.
            kept = protected
            removed = 0
            self.insights = kept
        else:
            kept_evictable = evictable[:slots]
            removed = len(evictable) - len(kept_evictable)
            self.insights = protected + kept_evictable

        if removed:
            self._save()
            logger.info(
                "Memory: pruned %d low-priority insights "
                "(kept %d, %d in grace)",
                removed, len(self.insights), len(protected),
            )

    def get_insights(self, category: Optional[str] = None,
                     min_confidence: float = 0.0) -> List[Insight]:
        result = self.insights
        if category:
            result = [i for i in result if i.category == category]
        result = [i for i in result if i.confidence >= min_confidence]
        return sorted(result, key=lambda x: x.confidence, reverse=True)

    def format_for_llm(self) -> str:
        lines = []
        # Sort so insights that the Judge has actually cited come first —
        # this makes the live-vs-dead distinction visible to the next agent.
        ranked = sorted(
            self.insights,
            key=lambda x: (x.cited_count, x.confidence),
            reverse=True,
        )
        for insight in ranked:
            if insight.confidence >= 0.8:
                marker = "[HIGH]"
            elif insight.confidence >= 0.5:
                marker = "[MED]"
            else:
                marker = "[LOW]"
            usage = (
                f"cited={insight.cited_count}, "
                f"confirmed={insight.confirmed_count}, "
                f"contradicted={insight.contradicted_count}"
            )
            lines.append(
                f"{marker} (conf={insight.confidence:.2f}, {usage}) "
                f"{insight.observation} -> {insight.recommendation}"
            )
        if not lines:
            return "No insights accumulated yet."
        return "\n".join(lines)

    def format_retrospectives_for_llm(self, parent_exp_names: Optional[List[str]] = None,
                                      limit: int = 8) -> str:
        """Format the most relevant retrospectives for downstream agents.

        If `parent_exp_names` is given, prioritise retrospectives of those
        experiments (the Engineer cares most about its candidate parent's
        history). Otherwise return the most recent ones.
        """
        if not self.retrospectives:
            return "No experiment retrospectives recorded yet."
        chosen = []
        if parent_exp_names:
            for name in parent_exp_names:
                r = self.get_retrospective(name)
                if r:
                    chosen.append(r)
        # Fill remaining slots with most recent retrospectives
        for r in reversed(self.retrospectives):
            if r in chosen:
                continue
            chosen.append(r)
            if len(chosen) >= limit:
                break
        lines = []
        for r in chosen[:limit]:
            delta = f"  Δ vs parent: {r.delta_mae_vs_parent:+.4f}" if r.delta_mae_vs_parent is not None else ""
            lines.append(
                f"- [{r.outcome}] {r.exp_name}{delta}\n"
                f"    why: {r.why}\n"
                f"    learning: {r.learning}"
            )
        return "\n".join(lines)

    def get_completed_techniques(self) -> List[str]:
        """Get list of techniques/approaches that have been tried."""
        techniques = []
        for insight in self.insights:
            if insight.category in ("architecture", "loss", "augmentation", "code_change"):
                techniques.append(insight.observation)
        return techniques

    def _find_similar(self, new_insight: Insight) -> Optional[Insight]:
        if not new_insight.parameter_affected:
            return None
        for insight in self.insights:
            if insight.parameter_affected == new_insight.parameter_affected:
                if insight.recommendation and new_insight.recommendation:
                    if insight.recommendation == new_insight.recommendation:
                        return insight
        return None
