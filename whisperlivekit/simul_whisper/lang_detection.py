"""
Language detection utilities with soft voting and gap analysis.

This module provides improved language detection algorithms that address
issues with the original quadratic weighting approach.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LangDecisionReason(Enum):
    """Reason for language detection decision."""
    HIGH_CONFIDENCE = "high_confidence"
    CONSENSUS = "consensus"
    AMBIGUOUS = "ambiguous"
    FALLBACK = "fallback"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class LangCandidate:
    """Represents a language candidate with its scores."""
    lang: str
    raw_votes: int = 0
    weighted_score: float = 0.0
    mean_prob: float = 0.0
    max_prob: float = 0.0
    prob_sum: float = 0.0

    def finalize(self):
        """Calculate mean probability after all votes are accumulated."""
        if self.raw_votes > 0:
            self.mean_prob = self.prob_sum / self.raw_votes


@dataclass
class LangDecisionResult:
    """Result of language detection with full metadata."""
    selected_lang: str
    confidence: float
    reason: LangDecisionReason
    candidates: List[LangCandidate]
    gap_to_runner_up: float
    consistency_score: float
    is_ambiguous: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for frontend consumption."""
        return {
            "selected_lang": self.selected_lang,
            "confidence": round(self.confidence, 4),
            "reason": self.reason.value,
            "candidates": [
                {
                    "lang": c.lang,
                    "votes": c.raw_votes,
                    "weighted_score": round(c.weighted_score, 4),
                    "mean_prob": round(c.mean_prob, 4),
                }
                for c in self.candidates[:5]  # Top 5 candidates
            ],
            "gap_to_runner_up": round(self.gap_to_runner_up, 4),
            "consistency": round(self.consistency_score, 4),
            "is_ambiguous": self.is_ambiguous,
        }


class SoftVotingAggregator:
    """
    Soft voting aggregator with linear weighting and gap analysis.

    Key improvements over quadratic weighting:
    - Linear weighting: prob instead of prob^2
    - Gap analysis: detects ambiguous cases where top candidates are close
    - Ambiguity detection: marks results when gap < threshold

    Example:
        >>> aggregator = SoftVotingAggregator()
        >>> predictions = [("en", 0.9), ("en", 0.85), ("es", 0.6)]
        >>> result = aggregator.aggregate(predictions)
        >>> print(result.selected_lang)  # "en"
    """

    def __init__(
        self,
        min_gap_threshold: float = 0.05,
        ambiguity_threshold: float = 0.10,
        min_consensus_ratio: float = 0.5,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the aggregator.

        Args:
            min_gap_threshold: Minimum gap required between top-1 and top-2.
                              If gap < this, result is marked AMBIGUOUS.
            ambiguity_threshold: Threshold for is_ambiguous flag.
                                 If gap < this, is_ambiguous=True.
            min_consensus_ratio: Minimum fraction of votes for winner required
                                 for HIGH_CONFIDENCE decision.
            confidence_threshold: Minimum mean probability for HIGH_CONFIDENCE.
        """
        self.min_gap_threshold = min_gap_threshold
        self.ambiguity_threshold = ambiguity_threshold
        self.min_consensus_ratio = min_consensus_ratio
        self.confidence_threshold = confidence_threshold

    def aggregate(
        self,
        predictions: List[Tuple[str, float]]
    ) -> LangDecisionResult:
        """
        Aggregate predictions using soft voting with linear weights.

        Args:
            predictions: List of (language_code, probability) tuples

        Returns:
            LangDecisionResult with full metadata including candidates,
            gap analysis, and decision reason.
        """
        if not predictions:
            return self._empty_result()

        # Step 1: Accumulate scores with LINEAR weighting (not quadratic!)
        candidates: Dict[str, LangCandidate] = {}

        for lang, prob in predictions:
            if lang not in candidates:
                candidates[lang] = LangCandidate(lang=lang)

            c = candidates[lang]
            c.raw_votes += 1
            c.weighted_score += prob  # LINEAR weighting
            c.prob_sum += prob
            c.max_prob = max(c.max_prob, prob)

        # Step 2: Finalize candidates (calculate mean_prob)
        for c in candidates.values():
            c.finalize()

        # Step 3: Normalize weighted scores to sum to 1.0
        total_score = sum(c.weighted_score for c in candidates.values())
        if total_score > 0:
            for c in candidates.values():
                c.weighted_score = c.weighted_score / total_score

        # Step 4: Sort by weighted score (descending)
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda c: c.weighted_score,
            reverse=True
        )

        best = sorted_candidates[0]
        runner_up = sorted_candidates[1] if len(sorted_candidates) > 1 else None

        # Step 5: Calculate gap and consistency
        gap = (best.weighted_score - runner_up.weighted_score) if runner_up else 1.0
        total_votes = sum(c.raw_votes for c in candidates.values())
        consistency = best.raw_votes / total_votes if total_votes > 0 else 0.0

        # Step 6: Determine decision reason and ambiguity
        is_ambiguous = gap < self.ambiguity_threshold

        if gap < self.min_gap_threshold:
            reason = LangDecisionReason.AMBIGUOUS
        elif (consistency >= self.min_consensus_ratio and
              best.mean_prob >= self.confidence_threshold):
            reason = LangDecisionReason.HIGH_CONFIDENCE
        elif consistency >= self.min_consensus_ratio:
            reason = LangDecisionReason.CONSENSUS
        else:
            reason = LangDecisionReason.LOW_CONFIDENCE

        # Log decision details
        self._log_decision(
            best, runner_up, gap, consistency, reason, is_ambiguous, sorted_candidates
        )

        return LangDecisionResult(
            selected_lang=best.lang,
            confidence=best.mean_prob,
            reason=reason,
            candidates=sorted_candidates,
            gap_to_runner_up=gap,
            consistency_score=consistency,
            is_ambiguous=is_ambiguous,
        )

    def _empty_result(self) -> LangDecisionResult:
        """Return empty result when no predictions available."""
        return LangDecisionResult(
            selected_lang="",
            confidence=0.0,
            reason=LangDecisionReason.FALLBACK,
            candidates=[],
            gap_to_runner_up=0.0,
            consistency_score=0.0,
            is_ambiguous=True,
        )

    def _log_decision(
        self,
        best: LangCandidate,
        runner_up: Optional[LangCandidate],
        gap: float,
        consistency: float,
        reason: LangDecisionReason,
        is_ambiguous: bool,
        all_candidates: List[LangCandidate],
    ):
        """Log detailed decision information."""
        logger.info(f"[LangID] Soft voting result: {best.lang}")
        logger.info(
            f"[LangID] Confidence: {best.mean_prob:.4f}, "
            f"Gap: {gap:.4f}, Consistency: {consistency:.2%}"
        )
        logger.info(
            f"[LangID] Reason: {reason.value}, Ambiguous: {is_ambiguous}"
        )

        # Log top candidates
        for i, c in enumerate(all_candidates[:3]):
            marker = ">>>" if i == 0 else "   "
            runner_up_marker = "(runner-up)" if i == 1 else ""
            logger.info(
                f"[LangID] {marker} {c.lang}: votes={c.raw_votes}, "
                f"score={c.weighted_score:.4f}, mean_prob={c.mean_prob:.4f} {runner_up_marker}"
            )
