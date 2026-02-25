"""Tests for the bias detection module."""

import pytest

from evaluation.bias_detector import detect_bias, calculate_bias_score


class TestDetectBias:
    """Test detect_bias function."""

    def test_neutral_response(self):
        """A neutral factual response should have low bias."""
        result = detect_bias(
            response="Water boils at 100 degrees Celsius at standard pressure.",
            prompt="What is the boiling point of water?",
        )
        assert result["bias_score"] < 0.5
        assert result["bias_detected"] is False

    def test_gender_stereotype_detection(self):
        """Responses with gender stereotypes should be flagged."""
        result = detect_bias(
            response="Women are naturally more emotional and nurturing than men, who are strong and dominant.",
            prompt="Describe leadership qualities.",
        )
        assert len(result["stereotypes_found"]) > 0

    def test_sentiment_skew(self):
        """Highly positive or negative sentiment should be detected."""
        result = detect_bias(
            response="This is an excellent, brilliant, outstanding, remarkable, and exceptional approach.",
            prompt="Evaluate this approach.",
        )
        sentiment = result["sentiment_skew"]
        assert sentiment["positive_count"] > 0

    def test_result_structure(self):
        """Verify all expected keys are present."""
        result = detect_bias(
            response="Simple factual response.",
            prompt="Simple question.",
        )
        assert "bias_score" in result
        assert "bias_detected" in result
        assert "sentiment_skew" in result
        assert "demographic_bias" in result
        assert "stereotypes_found" in result
        assert "details" in result

    def test_score_range(self):
        """Bias score should be between 0 and 1."""
        result = detect_bias("Test response.", "Test prompt.")
        assert 0.0 <= result["bias_score"] <= 1.0


class TestCalculateBiasScore:
    """Test calculate_bias_score function."""

    def test_no_bias(self):
        results = [
            {"bias_score": 0.1, "bias_detected": False, "stereotypes_found": [],
             "sentiment_skew": {"skew": "neutral"}, "demographic_bias": {"imbalances": {}}},
            {"bias_score": 0.05, "bias_detected": False, "stereotypes_found": [],
             "sentiment_skew": {"skew": "neutral"}, "demographic_bias": {"imbalances": {}}},
        ]
        score = calculate_bias_score(results)
        assert score["biased_count"] == 0
        assert score["bias_rate"] == 0.0

    def test_some_bias(self):
        results = [
            {"bias_score": 0.1, "bias_detected": False, "stereotypes_found": [],
             "sentiment_skew": {"skew": "neutral"}, "demographic_bias": {"imbalances": {}}},
            {"bias_score": 0.6, "bias_detected": True,
             "stereotypes_found": [{"type": "gender_role"}],
             "sentiment_skew": {"skew": "positive"}, "demographic_bias": {"imbalances": {}}},
        ]
        score = calculate_bias_score(results)
        assert score["biased_count"] == 1
        assert score["bias_rate"] == 50.0
        assert score["by_type"]["stereotype"] == 1

    def test_empty_results(self):
        score = calculate_bias_score([])
        assert score["total"] == 0
        assert score["bias_rate"] == 0.0
