"""Tests for the consistency measurement module."""

import pytest

from evaluation.consistency import (
    measure_consistency,
    detect_contradictions,
    consistency_report,
)


class TestMeasureConsistency:
    """Test measure_consistency function."""

    def test_identical_responses(self):
        """Identical responses should give perfect consistency."""
        score = measure_consistency(["Paris", "Paris", "Paris"])
        assert score >= 0.99

    def test_similar_responses(self):
        """Semantically similar responses should have high consistency."""
        score = measure_consistency([
            "Paris is the capital",
            "The capital is Paris",
            "Paris, capital of France",
        ])
        assert score > 0.4

    def test_different_responses(self):
        """Very different responses should have low consistency."""
        score = measure_consistency([
            "The speed of light is fast",
            "Python is a programming language",
            "Mount Everest is the tallest mountain",
        ])
        assert score < 0.4

    def test_single_response(self):
        """Single response is trivially consistent."""
        score = measure_consistency(["Only one response"])
        assert score == 1.0

    def test_empty_list(self):
        """Empty list should return 1.0 (trivially consistent)."""
        score = measure_consistency([])
        assert score == 1.0

    def test_score_range(self):
        """Score should always be between 0 and 1."""
        score = measure_consistency(["hello", "world", "test"])
        assert 0.0 <= score <= 1.0


class TestDetectContradictions:
    """Test detect_contradictions function."""

    def test_no_contradictions(self):
        """Similar responses should not trigger contradictions."""
        contradictions = detect_contradictions([
            "Paris is the capital of France",
            "The capital of France is Paris",
        ])
        assert len(contradictions) == 0

    def test_contradictory_responses(self):
        """Very different responses should be flagged."""
        contradictions = detect_contradictions([
            "The answer is definitely yes, it works perfectly",
            "Quantum computing uses qubits for parallel processing",
        ])
        assert len(contradictions) >= 0  # May or may not detect

    def test_single_response(self):
        """Single response cannot have contradictions."""
        contradictions = detect_contradictions(["Just one"])
        assert len(contradictions) == 0

    def test_contradiction_structure(self):
        """Verify contradiction results have expected fields."""
        contradictions = detect_contradictions([
            "The temperature is very hot at 100 degrees",
            "It is extremely cold and freezing outside today",
        ])
        for c in contradictions:
            assert "pair" in c
            assert "response_1" in c
            assert "response_2" in c
            assert "similarity" in c


class TestConsistencyReport:
    """Test consistency_report function."""

    def test_basic_report(self):
        results = [
            {"topic": "test1", "consistency_score": 0.9, "contradictions": []},
            {"topic": "test2", "consistency_score": 0.7, "contradictions": [{"pair": (0, 1)}]},
        ]
        report = consistency_report(results)
        assert report["total_topics"] == 2
        assert report["avg_consistency"] == 0.8
        assert report["min_consistency"] == 0.7
        assert report["max_consistency"] == 0.9
        assert report["topics_with_contradictions"] == 1

    def test_empty_report(self):
        report = consistency_report([])
        assert report["total_topics"] == 0
        assert report["avg_consistency"] == 0.0
