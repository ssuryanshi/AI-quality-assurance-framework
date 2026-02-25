"""
Hallucination Detection Module
===============================
Detects hallucinations in LLM responses using multiple strategies:
    1. Contradiction detection: Does the response contradict the expected answer?
    2. Fabrication detection: Does the response contain claims not in the context?
    3. Confidence scoring: Assigns a hallucination probability (0.0 - 1.0)

Usage:
    from evaluation.hallucination import detect_hallucination, calculate_hallucination_rate

    result = detect_hallucination(
        response="The capital of France is Berlin.",
        context="France is a country in Western Europe.",
        expected="Paris"
    )
    print(result["hallucination_score"])  # 0.85
"""

import re
import logging
from typing import Any, Dict, List, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# ── Indicators of uncertainty (good — model knows it doesn't know) ──
HEDGING_PHRASES = [
    "i'm not sure", "i don't know", "i cannot", "i can't",
    "i am not sure", "i do not know", "uncertain", "unclear",
    "i don't have", "i do not have", "not certain", "may not be",
    "might not", "cannot confirm", "unable to verify",
    "no reliable information", "no verified", "not a real",
    "does not exist", "is not a recognized", "is fictional",
    "is not a real", "not a known", "fabricated",
]

# ── Indicators of confident fabrication (bad — model is making things up) ──
CONFIDENT_FABRICATION_MARKERS = [
    "is located in", "was founded in", "the answer is",
    "the capital is", "was born in", "died in",
    "published in", "according to", "studies show",
    "research confirms", "it is well known", "definitely",
    "the exact", "precisely",
]


def detect_hallucination(
    response: str,
    context: str,
    expected: str,
    hallucination_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a model response for potential hallucinations.

    Combines multiple detection strategies to produce a hallucination score.

    Args:
        response: The model's generated response.
        context: Reference context/facts provided to the model.
        expected: The expected/correct answer.
        hallucination_type: Optional known hallucination category for scoring.

    Returns:
        Dictionary with:
            - hallucination_score: Float 0.0 (no hallucination) to 1.0 (definite hallucination)
            - is_hallucination: Boolean based on score > 0.5
            - contradiction_detected: Whether response contradicts expected answer
            - fabrication_detected: Whether response fabricates information
            - hedging_detected: Whether model expressed appropriate uncertainty
            - details: Human-readable explanation
    """
    response_lower = response.lower().strip()
    expected_lower = expected.lower().strip()
    context_lower = context.lower().strip() if context else ""

    scores = []
    details = []

    # ── Strategy 1: Contradiction Detection ──
    contradiction_score = _detect_contradiction(response_lower, expected_lower)
    scores.append(("contradiction", contradiction_score, 0.35))
    if contradiction_score > 0.5:
        details.append(f"Contradiction detected (score: {contradiction_score:.2f})")

    # ── Strategy 2: Fabrication Detection ──
    fabrication_score = _detect_fabrication(response_lower, context_lower, expected_lower)
    scores.append(("fabrication", fabrication_score, 0.35))
    if fabrication_score > 0.5:
        details.append(f"Potential fabrication detected (score: {fabrication_score:.2f})")

    # ── Strategy 3: Hedging Analysis ──
    hedging_score = _analyze_hedging(response_lower, expected_lower)
    scores.append(("hedging", hedging_score, 0.15))

    # ── Strategy 4: Response-Expected Similarity ──
    similarity_score = _check_answer_relevance(response_lower, expected_lower)
    scores.append(("relevance", 1.0 - similarity_score, 0.15))
    if similarity_score < 0.3:
        details.append(f"Low relevance to expected answer (similarity: {similarity_score:.2f})")

    # ── Calculate weighted hallucination score ──
    hallucination_score = sum(score * weight for _, score, weight in scores)
    hallucination_score = round(min(max(hallucination_score, 0.0), 1.0), 4)

    # Adjust if model appropriately hedged
    if hedging_score < 0.3:  # Model expressed uncertainty appropriately
        hallucination_score = max(0.0, hallucination_score - 0.2)
        details.append("Model appropriately expressed uncertainty")

    is_hallucination = hallucination_score > 0.5

    return {
        "hallucination_score": hallucination_score,
        "is_hallucination": is_hallucination,
        "contradiction_detected": contradiction_score > 0.5,
        "fabrication_detected": fabrication_score > 0.5,
        "hedging_detected": hedging_score < 0.3,
        "response_relevance": round(similarity_score, 4),
        "details": "; ".join(details) if details else "No hallucination indicators found",
    }


def _detect_contradiction(response: str, expected: str) -> float:
    """
    Detect if the response directly contradicts the expected answer.

    Uses keyword overlap and negation pattern analysis.

    Returns:
        Score from 0.0 (no contradiction) to 1.0 (clear contradiction).
    """
    # Extract key words from expected answer
    expected_words = set(re.findall(r'\b\w{3,}\b', expected))
    response_words = set(re.findall(r'\b\w{3,}\b', response))

    if not expected_words:
        return 0.0

    # Check word overlap
    overlap = expected_words & response_words
    overlap_ratio = len(overlap) / len(expected_words) if expected_words else 0

    # Check for negation patterns
    negation_patterns = [
        r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bnone\b',
        r"\bisn't\b", r"\bwasn't\b", r"\bdoesn't\b", r"\bdon't\b",
        r"\bcannot\b", r"\bcan't\b",
    ]

    response_has_negation = any(re.search(p, response) for p in negation_patterns)
    expected_has_negation = any(re.search(p, expected) for p in negation_patterns)

    # Contradiction: response negates what expected affirms (or vice versa)
    if response_has_negation != expected_has_negation and overlap_ratio > 0.3:
        return 0.7

    # Low overlap suggests the response is about something different
    if overlap_ratio < 0.2:
        return 0.6

    # High overlap with matching sentiment — likely not a contradiction
    if overlap_ratio > 0.5:
        return 0.1

    return 0.3


def _detect_fabrication(response: str, context: str, expected: str) -> float:
    """
    Detect if the response fabricates information not found in context or expected answer.

    Looks for confident claims that introduce new, unverifiable information.

    Returns:
        Score from 0.0 (no fabrication) to 1.0 (definite fabrication).
    """
    reference_text = f"{context} {expected}".lower()
    reference_words = set(re.findall(r'\b\w{4,}\b', reference_text))

    # Check for confident fabrication markers
    fabrication_marker_count = sum(
        1 for marker in CONFIDENT_FABRICATION_MARKERS
        if marker in response
    )

    # Check for specific claims (numbers, dates, names) not in reference
    specific_patterns = re.findall(
        r'\b(?:\d{4}|\d+(?:\.\d+)?%?|[A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b',
        response,
    )

    novel_claims = 0
    for claim in specific_patterns:
        if claim.lower() not in reference_text:
            novel_claims += 1

    # Calculate fabrication score
    score = 0.0

    if fabrication_marker_count >= 3:
        score += 0.4
    elif fabrication_marker_count >= 1:
        score += 0.2

    if novel_claims >= 3:
        score += 0.4
    elif novel_claims >= 1:
        score += 0.2

    # If response is very long compared to expected, more room for fabrication
    if len(response) > len(expected) * 5 and len(expected) > 10:
        score += 0.1

    return min(score, 1.0)


def _analyze_hedging(response: str, expected: str) -> float:
    """
    Analyze whether the model appropriately hedges its response.
    Lower score = more hedging (good when model should be uncertain).

    Returns:
        Score from 0.0 (heavy hedging) to 1.0 (fully confident).
    """
    hedging_count = sum(1 for phrase in HEDGING_PHRASES if phrase in response)

    if hedging_count >= 3:
        return 0.1
    elif hedging_count >= 2:
        return 0.2
    elif hedging_count >= 1:
        return 0.4
    return 1.0


def _check_answer_relevance(response: str, expected: str) -> float:
    """
    Check how relevant the response is to the expected answer.

    Returns:
        Similarity score 0.0 to 1.0.
    """
    if not response or not expected:
        return 0.0

    # Use SequenceMatcher for basic similarity
    ratio = SequenceMatcher(None, response[:500], expected).ratio()

    # Check if expected answer is contained in response
    if expected in response:
        return max(ratio, 0.9)

    # Check key word containment
    expected_words = set(re.findall(r'\b\w{3,}\b', expected))
    if expected_words:
        response_words = set(re.findall(r'\b\w{3,}\b', response))
        word_overlap = len(expected_words & response_words) / len(expected_words)
        return max(ratio, word_overlap)

    return ratio


def calculate_hallucination_rate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate aggregate hallucination metrics from a list of detection results.

    Args:
        results: List of dicts, each from detect_hallucination().

    Returns:
        Dictionary with:
            - total: Total items evaluated
            - hallucination_count: Number flagged as hallucinations
            - hallucination_rate: Percentage of hallucinations
            - avg_score: Average hallucination score
            - contradiction_count: Number with contradictions
            - fabrication_count: Number with fabrications
    """
    if not results:
        return {
            "total": 0,
            "hallucination_count": 0,
            "hallucination_rate": 0.0,
            "avg_score": 0.0,
            "contradiction_count": 0,
            "fabrication_count": 0,
        }

    total = len(results)
    hallucination_count = sum(1 for r in results if r.get("is_hallucination", False))
    contradiction_count = sum(1 for r in results if r.get("contradiction_detected", False))
    fabrication_count = sum(1 for r in results if r.get("fabrication_detected", False))
    avg_score = sum(r.get("hallucination_score", 0.0) for r in results) / total

    return {
        "total": total,
        "hallucination_count": hallucination_count,
        "hallucination_rate": round((hallucination_count / total) * 100, 2),
        "avg_score": round(avg_score, 4),
        "contradiction_count": contradiction_count,
        "fabrication_count": fabrication_count,
    }
