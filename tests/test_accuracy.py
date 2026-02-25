"""Tests for the accuracy evaluation module."""

import pytest

from evaluation.accuracy import (
    exact_match,
    fuzzy_match,
    semantic_similarity,
    calculate_accuracy,
)


class TestExactMatch:
    """Test exact_match function."""

    def test_identical_strings(self):
        assert exact_match("Paris", "Paris") is True

    def test_case_insensitive(self):
        assert exact_match("paris", "PARIS") is True

    def test_whitespace_handling(self):
        assert exact_match("  Paris  ", "Paris") is True

    def test_different_strings(self):
        assert exact_match("London", "Paris") is False

    def test_partial_match_fails(self):
        assert exact_match("Paris, France", "Paris") is False

    def test_empty_strings(self):
        assert exact_match("", "") is True


class TestFuzzyMatch:
    """Test fuzzy_match function."""

    def test_identical_returns_one(self):
        score = fuzzy_match("Albert Einstein", "Albert Einstein")
        assert score >= 0.95

    def test_similar_strings(self):
        score = fuzzy_match("Albert Einstein", "einstein albert")
        assert score > 0.5

    def test_completely_different(self):
        score = fuzzy_match("apple", "basketball")
        assert score < 0.5

    def test_contained_answer(self):
        """Test that containing the expected answer boosts score."""
        score = fuzzy_match(
            "The answer is Paris, the capital of France", "Paris"
        )
        assert score > 0.0

    def test_empty_strings(self):
        assert fuzzy_match("", "something") == 0.0
        assert fuzzy_match("something", "") == 0.0

    def test_score_range(self):
        """Scores should always be between 0 and 1."""
        score = fuzzy_match("test", "testing")
        assert 0.0 <= score <= 1.0


class TestSemanticSimilarity:
    """Test semantic_similarity function."""

    def test_identical_texts(self):
        score = semantic_similarity("The capital of France is Paris",
                                    "The capital of France is Paris")
        assert score >= 0.99

    def test_paraphrased_texts(self):
        score = semantic_similarity(
            "Paris is the capital of France",
            "The capital of France is Paris"
        )
        assert score > 0.5

    def test_unrelated_texts(self):
        score = semantic_similarity(
            "Python is a programming language",
            "The weather is sunny today"
        )
        assert score < 0.5

    def test_empty_strings(self):
        assert semantic_similarity("", "something") == 0.0

    def test_score_range(self):
        score = semantic_similarity("hello world", "hi there")
        assert 0.0 <= score <= 1.0


class TestCalculateAccuracy:
    """Test calculate_accuracy function."""

    def test_perfect_accuracy(self):
        results = [
            {"predicted": "Paris", "expected": "Paris"},
            {"predicted": "H2O", "expected": "H2O"},
        ]
        acc = calculate_accuracy(results)
        assert acc["total"] == 2
        assert acc["exact_matches"] == 2
        assert acc["exact_match_rate"] == 100.0

    def test_zero_accuracy(self):
        results = [
            {"predicted": "London", "expected": "Paris"},
            {"predicted": "NaCl", "expected": "H2O"},
        ]
        acc = calculate_accuracy(results)
        assert acc["exact_matches"] == 0
        assert acc["exact_match_rate"] == 0.0

    def test_partial_accuracy(self):
        results = [
            {"predicted": "Paris", "expected": "Paris"},
            {"predicted": "London", "expected": "Berlin"},
            {"predicted": "H2O", "expected": "H2O"},
        ]
        acc = calculate_accuracy(results)
        assert acc["total"] == 3
        assert acc["exact_matches"] == 2
        assert abs(acc["exact_match_rate"] - 66.67) < 0.1

    def test_empty_results(self):
        acc = calculate_accuracy([])
        assert acc["total"] == 0
        assert acc["exact_match_rate"] == 0.0

    def test_per_item_details(self):
        results = [{"predicted": "Paris", "expected": "Paris"}]
        acc = calculate_accuracy(results)
        assert len(acc["per_item"]) == 1
        assert acc["per_item"][0]["exact_match"] is True
        assert "fuzzy_score" in acc["per_item"][0]
        assert "semantic_score" in acc["per_item"][0]

    def test_avg_scores_present(self):
        results = [
            {"predicted": "Paris", "expected": "Paris"},
            {"predicted": "London", "expected": "Berlin"},
        ]
        acc = calculate_accuracy(results)
        assert "avg_fuzzy_score" in acc
        assert "avg_semantic_score" in acc
        assert 0.0 <= acc["avg_fuzzy_score"] <= 1.0
