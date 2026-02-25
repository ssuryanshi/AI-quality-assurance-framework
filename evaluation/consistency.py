"""
Consistency Measurement Module
==============================
Measures how consistently an LLM responds across rephrased versions
of the same question. Inconsistent responses indicate unreliable behavior.

Usage:
    from evaluation.consistency import measure_consistency, detect_contradictions

    responses = [
        "The speed of light is approximately 3 x 10^8 m/s.",
        "Light travels at about 300,000 km per second.",
        "The velocity of light in vacuum is 299,792,458 m/s."
    ]
    score = measure_consistency(responses)
    print(f"Consistency: {score:.2%}")
"""

import re
import logging
from typing import Any, Dict, List
from difflib import SequenceMatcher
from itertools import combinations

logger = logging.getLogger(__name__)


def measure_consistency(responses: List[str]) -> float:
    """
    Measure the consistency of multiple responses to the same question.
    Computes pairwise similarity across all response pairs.

    Args:
        responses: List of response strings (from same question, different phrasings).

    Returns:
        Average pairwise similarity score (0.0 to 1.0).
        Higher = more consistent.

    Example:
        >>> measure_consistency(["Paris", "paris", "Paris, France"])
        0.85  # approximately
    """
    if len(responses) < 2:
        return 1.0  # Single response is trivially consistent

    # Normalize responses
    normalized = [_normalize_for_comparison(r) for r in responses]

    # Calculate pairwise similarities
    similarities = []
    for r1, r2 in combinations(normalized, 2):
        sim = _compute_similarity(r1, r2)
        similarities.append(sim)

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    return round(avg_similarity, 4)


def detect_contradictions(responses: List[str]) -> List[Dict[str, Any]]:
    """
    Find contradictory pairs among a set of responses.
    Two responses contradict if they have low similarity but share key terms.

    Args:
        responses: List of response strings to an equivalent question.

    Returns:
        List of dicts, each describing a detected contradiction:
            - pair: Tuple of (index1, index2)
            - response_1: First response text
            - response_2: Second response text
            - similarity: Similarity score between the pair
            - explanation: Why this was flagged
    """
    if len(responses) < 2:
        return []

    contradictions = []
    normalized = [_normalize_for_comparison(r) for r in responses]

    for (i, r1), (j, r2) in combinations(enumerate(normalized), 2):
        similarity = _compute_similarity(r1, r2)

        # Low similarity indicates potential contradiction
        if similarity < 0.5:
            # Check if they share key terms (which makes low similarity suspicious)
            shared_terms = _get_shared_terms(r1, r2)

            if shared_terms or similarity < 0.3:
                explanation = _build_contradiction_explanation(
                    responses[i], responses[j], similarity, shared_terms
                )
                contradictions.append({
                    "pair": (i, j),
                    "response_1": responses[i],
                    "response_2": responses[j],
                    "similarity": round(similarity, 4),
                    "shared_terms": list(shared_terms),
                    "explanation": explanation,
                })

    return contradictions


def consistency_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary report from consistency evaluation results.

    Args:
        results: List of dicts, each with:
            - topic: The question topic
            - consistency_score: Float 0-1
            - contradictions: List from detect_contradictions()

    Returns:
        Summary dictionary with overall stats.
    """
    if not results:
        return {
            "total_topics": 0,
            "avg_consistency": 0.0,
            "min_consistency": 0.0,
            "max_consistency": 0.0,
            "topics_with_contradictions": 0,
            "total_contradictions": 0,
        }

    scores = [r.get("consistency_score", 0.0) for r in results]
    contradiction_counts = [len(r.get("contradictions", [])) for r in results]

    return {
        "total_topics": len(results),
        "avg_consistency": round(sum(scores) / len(scores), 4),
        "min_consistency": round(min(scores), 4),
        "max_consistency": round(max(scores), 4),
        "topics_with_contradictions": sum(1 for c in contradiction_counts if c > 0),
        "total_contradictions": sum(contradiction_counts),
    }


# ── Private Helper Functions ──


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _compute_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity between two normalized texts.
    Uses a combination of SequenceMatcher and word-level Jaccard similarity.
    """
    if not text1 or not text2:
        return 0.0

    # Character-level similarity
    char_sim = SequenceMatcher(None, text1, text2).ratio()

    # Word-level Jaccard similarity
    words1 = set(text1.split())
    words2 = set(text2.split())
    if words1 or words2:
        jaccard = len(words1 & words2) / len(words1 | words2)
    else:
        jaccard = 0.0

    # Weighted average (word-level gets more weight for semantic comparison)
    return 0.4 * char_sim + 0.6 * jaccard


def _get_shared_terms(text1: str, text2: str) -> set:
    """Extract meaningful shared terms (words with 4+ characters)."""
    words1 = set(re.findall(r'\b\w{4,}\b', text1))
    words2 = set(re.findall(r'\b\w{4,}\b', text2))
    return words1 & words2


def _build_contradiction_explanation(
    resp1: str, resp2: str, similarity: float, shared_terms: set
) -> str:
    """Build a human-readable explanation for a detected contradiction."""
    parts = [f"Low similarity ({similarity:.2f})"]

    if shared_terms:
        terms_str = ", ".join(list(shared_terms)[:5])
        parts.append(f"despite sharing terms: {terms_str}")

    if similarity < 0.2:
        parts.append("Responses appear fundamentally different")
    elif similarity < 0.4:
        parts.append("Responses show significant divergence")

    return ". ".join(parts) + "."
