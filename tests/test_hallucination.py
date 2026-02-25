"""Tests for the hallucination detection module."""

import pytest

from evaluation.hallucination import detect_hallucination, calculate_hallucination_rate


class TestDetectHallucination:
    """Test detect_hallucination function."""

    def test_correct_answer_no_hallucination(self):
        """A correct response should have low hallucination score."""
        result = detect_hallucination(
            response="The capital of France is Paris.",
            context="France is a country in Western Europe with Paris as its capital.",
            expected="Paris",
        )
        assert result["hallucination_score"] < 0.5
        assert result["is_hallucination"] is False

    def test_contradictory_answer_detected(self):
        """A wrong answer should have higher hallucination score."""
        result = detect_hallucination(
            response="The capital of France is Berlin, a major city in Germany.",
            context="France is a country in Western Europe.",
            expected="Paris",
        )
        # Should detect some level of issue
        assert result["hallucination_score"] > 0.0

    def test_fabricated_response(self):
        """Fabricated details should be flagged."""
        result = detect_hallucination(
            response="Listenbourg, founded in 1432, is located in Central Europe with a population of 2.3 million.",
            context="There is no country called Listenbourg.",
            expected="Listenbourg is not a real country.",
        )
        assert result["hallucination_score"] > 0.3

    def test_appropriate_hedging(self):
        """Model expressing uncertainty should get a lower hallucination score."""
        result = detect_hallucination(
            response="I'm not sure about this, but I don't have reliable information about Listenbourg. It does not exist as a real country.",
            context="There is no country called Listenbourg.",
            expected="Listenbourg is not a real country. It does not exist.",
        )
        assert result["hedging_detected"] is True

    def test_result_structure(self):
        """Verify all expected keys are in the result."""
        result = detect_hallucination(
            response="Test response",
            context="Test context",
            expected="Test expected",
        )
        assert "hallucination_score" in result
        assert "is_hallucination" in result
        assert "contradiction_detected" in result
        assert "fabrication_detected" in result
        assert "hedging_detected" in result
        assert "details" in result

    def test_score_range(self):
        """Hallucination score should be between 0 and 1."""
        result = detect_hallucination(
            response="Some random text",
            context="Context text",
            expected="Expected answer",
        )
        assert 0.0 <= result["hallucination_score"] <= 1.0

    def test_empty_context(self):
        """Should handle empty context gracefully."""
        result = detect_hallucination(
            response="The answer is 42.",
            context="",
            expected="42",
        )
        assert "hallucination_score" in result


class TestCalculateHallucinationRate:
    """Test calculate_hallucination_rate function."""

    def test_no_hallucinations(self):
        results = [
            {"hallucination_score": 0.1, "is_hallucination": False,
             "contradiction_detected": False, "fabrication_detected": False},
            {"hallucination_score": 0.2, "is_hallucination": False,
             "contradiction_detected": False, "fabrication_detected": False},
        ]
        rate = calculate_hallucination_rate(results)
        assert rate["hallucination_count"] == 0
        assert rate["hallucination_rate"] == 0.0

    def test_all_hallucinations(self):
        results = [
            {"hallucination_score": 0.9, "is_hallucination": True,
             "contradiction_detected": True, "fabrication_detected": True},
            {"hallucination_score": 0.8, "is_hallucination": True,
             "contradiction_detected": False, "fabrication_detected": True},
        ]
        rate = calculate_hallucination_rate(results)
        assert rate["hallucination_count"] == 2
        assert rate["hallucination_rate"] == 100.0

    def test_mixed_results(self):
        results = [
            {"hallucination_score": 0.1, "is_hallucination": False,
             "contradiction_detected": False, "fabrication_detected": False},
            {"hallucination_score": 0.8, "is_hallucination": True,
             "contradiction_detected": True, "fabrication_detected": False},
        ]
        rate = calculate_hallucination_rate(results)
        assert rate["hallucination_count"] == 1
        assert rate["hallucination_rate"] == 50.0
        assert rate["contradiction_count"] == 1

    def test_empty_results(self):
        rate = calculate_hallucination_rate([])
        assert rate["total"] == 0
        assert rate["hallucination_rate"] == 0.0

    def test_avg_score(self):
        results = [
            {"hallucination_score": 0.2, "is_hallucination": False,
             "contradiction_detected": False, "fabrication_detected": False},
            {"hallucination_score": 0.6, "is_hallucination": True,
             "contradiction_detected": False, "fabrication_detected": False},
        ]
        rate = calculate_hallucination_rate(results)
        assert abs(rate["avg_score"] - 0.4) < 0.01
