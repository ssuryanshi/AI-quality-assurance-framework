"""
Accuracy Evaluation Module
==========================
Computes accuracy metrics for LLM responses against expected answers.

Methods:
    - exact_match: Case-insensitive exact string comparison
    - fuzzy_match: Approximate matching using SequenceMatcher
    - semantic_similarity: TF-IDF cosine similarity for meaning comparison
    - calculate_accuracy: Aggregate accuracy from a list of results

Usage:
    from evaluation.accuracy import exact_match, fuzzy_match, calculate_accuracy

    score = fuzzy_match("Albert Einstein", "einstein, albert")
    results = [{"predicted": "Paris", "expected": "Paris"}, ...]
    acc = calculate_accuracy(results)
"""

import re
import difflib
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    Lowercases, strips whitespace, removes extra spaces and punctuation.

    Args:
        text: Raw text string.

    Returns:
        Normalized string.
    """
    text = text.lower().strip()
    # Remove common punctuation that doesn't affect meaning
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def exact_match(predicted: str, expected: str) -> bool:
    """
    Check if the predicted answer exactly matches the expected answer.
    Comparison is case-insensitive and whitespace-normalized.

    Args:
        predicted: The model's response.
        expected: The ground-truth answer.

    Returns:
        True if the normalized texts match exactly.

    Example:
        >>> exact_match("Paris", "paris")
        True
        >>> exact_match("New York City", "New York")
        False
    """
    return _normalize_text(predicted) == _normalize_text(expected)


def fuzzy_match(predicted: str, expected: str, threshold: float = 0.80) -> float:
    """
    Compute a fuzzy similarity score between predicted and expected answers.
    Uses Python's difflib.SequenceMatcher for approximate string matching.

    Args:
        predicted: The model's response.
        expected: The ground-truth answer.
        threshold: Not used in scoring, but available for pass/fail logic.

    Returns:
        A similarity score between 0.0 (no match) and 1.0 (identical).

    Example:
        >>> fuzzy_match("Albert Einstein", "einstein albert")
        0.8  # approximately
    """
    norm_predicted = _normalize_text(predicted)
    norm_expected = _normalize_text(expected)

    if not norm_predicted or not norm_expected:
        return 0.0

    # Primary: SequenceMatcher ratio
    ratio = difflib.SequenceMatcher(None, norm_predicted, norm_expected).ratio()

    # Bonus: check if expected answer is contained within predicted
    if norm_expected in norm_predicted:
        containment_bonus = len(norm_expected) / len(norm_predicted)
        ratio = max(ratio, containment_bonus)

    return round(ratio, 4)


def semantic_similarity(predicted: str, expected: str) -> float:
    """
    Compute semantic similarity using TF-IDF vectorization and cosine similarity.
    More robust than fuzzy matching for paraphrased responses.

    Args:
        predicted: The model's response.
        expected: The ground-truth answer.

    Returns:
        Cosine similarity score between 0.0 and 1.0.

    Example:
        >>> semantic_similarity(
        ...     "The capital of France is Paris",
        ...     "Paris is the capital city of France"
        ... )
        0.85  # approximately
    """
    if not predicted.strip() or not expected.strip():
        return 0.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([predicted, expected])
        similarity = cos_sim(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity), 4)

    except ImportError:
        logger.warning("scikit-learn not available, falling back to fuzzy match")
        return fuzzy_match(predicted, expected)

    except ValueError:
        # Handles edge case where TF-IDF produces empty vocabulary
        return fuzzy_match(predicted, expected)


def calculate_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate aggregate accuracy metrics from a list of evaluation results.

    Each result should have 'predicted' and 'expected' keys.

    Args:
        results: List of dicts with 'predicted' and 'expected' answer strings.

    Returns:
        Dictionary with:
            - total: Total number of items
            - exact_matches: Count of exact matches
            - exact_match_rate: Percentage of exact matches
            - avg_fuzzy_score: Average fuzzy match score
            - avg_semantic_score: Average semantic similarity
            - per_item: List of individual scores

    Example:
        >>> results = [
        ...     {"predicted": "Paris", "expected": "Paris"},
        ...     {"predicted": "London", "expected": "Berlin"},
        ... ]
        >>> acc = calculate_accuracy(results)
        >>> print(acc["exact_match_rate"])
        50.0
    """
    if not results:
        return {
            "total": 0,
            "exact_matches": 0,
            "exact_match_rate": 0.0,
            "avg_fuzzy_score": 0.0,
            "avg_semantic_score": 0.0,
            "per_item": [],
        }

    per_item = []
    exact_count = 0
    fuzzy_scores = []
    semantic_scores = []

    for item in results:
        predicted = item.get("predicted", "")
        expected = item.get("expected", "")

        is_exact = exact_match(predicted, expected)
        fuzzy = fuzzy_match(predicted, expected)
        semantic = semantic_similarity(predicted, expected)

        if is_exact:
            exact_count += 1

        fuzzy_scores.append(fuzzy)
        semantic_scores.append(semantic)

        per_item.append({
            "predicted": predicted,
            "expected": expected,
            "exact_match": is_exact,
            "fuzzy_score": fuzzy,
            "semantic_score": semantic,
        })

    total = len(results)
    return {
        "total": total,
        "exact_matches": exact_count,
        "exact_match_rate": round((exact_count / total) * 100, 2),
        "avg_fuzzy_score": round(sum(fuzzy_scores) / total, 4),
        "avg_semantic_score": round(sum(semantic_scores) / total, 4),
        "per_item": per_item,
    }
