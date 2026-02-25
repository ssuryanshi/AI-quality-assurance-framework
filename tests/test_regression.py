"""Tests for the regression testing module."""

import json
import os
import pytest

from regression.baseline_manager import BaselineManager
from regression.regression_runner import RegressionRunner


class TestBaselineManager:
    """Test BaselineManager save/load operations."""

    def test_save_and_load_baseline(self, temp_dir):
        """Test saving and loading a baseline file."""
        manager = BaselineManager(temp_dir)
        metrics = {"accuracy": {"exact_match_rate": 80.0}, "overall_score": 75.0}
        path = manager.save_baseline(metrics, model_name="test-model")

        assert os.path.exists(path)
        loaded = manager.load_baseline(path)
        assert loaded["model_name"] == "test-model"
        assert loaded["metrics"]["accuracy"]["exact_match_rate"] == 80.0

    def test_load_latest_baseline(self, temp_dir):
        """Test loading the most recent baseline."""
        manager = BaselineManager(temp_dir)

        # Save two baselines
        manager.save_baseline({"v": 1}, model_name="test", version="v1")
        manager.save_baseline({"v": 2}, model_name="test", version="v2")

        latest = manager.load_latest_baseline("test")
        assert latest is not None
        assert latest["metrics"]["v"] == 2

    def test_load_latest_no_baselines(self, temp_dir):
        """Test that None is returned when no baselines exist."""
        manager = BaselineManager(temp_dir)
        assert manager.load_latest_baseline("unknown") is None

    def test_list_baselines(self, temp_dir):
        """Test listing available baselines."""
        manager = BaselineManager(temp_dir)
        manager.save_baseline({"a": 1}, model_name="model-a", version="v1")
        manager.save_baseline({"b": 1}, model_name="model-b", version="v1")

        all_baselines = manager.list_baselines()
        assert len(all_baselines) == 2

        model_a = manager.list_baselines("model-a")
        assert len(model_a) == 1

    def test_delete_baseline(self, temp_dir):
        """Test deleting a baseline file."""
        manager = BaselineManager(temp_dir)
        path = manager.save_baseline({"x": 1}, model_name="test")

        assert manager.delete_baseline(path) is True
        assert not os.path.exists(path)
        assert manager.delete_baseline(path) is False


class TestRegressionRunner:
    """Test RegressionRunner comparison logic."""

    def test_no_baseline(self, temp_dir):
        """First run should return NO_BASELINE status."""
        runner = RegressionRunner(baseline_dir=temp_dir)
        result = runner.run_comparison(
            current_metrics={"accuracy": {"exact_match_rate": 80.0}, "overall_score": 75.0},
            model_name="test-model",
        )
        assert result["status"] == "NO_BASELINE"

    def test_stable_metrics(self, temp_dir):
        """Unchanged metrics should PASS."""
        runner = RegressionRunner(baseline_dir=temp_dir)

        metrics = {
            "accuracy": {"exact_match_rate": 80.0, "avg_fuzzy_score": 0.85},
            "hallucination": {"hallucination_rate": 10.0, "avg_score": 0.3},
            "consistency": {"avg_consistency": 0.82},
            "bias": {"bias_rate": 5.0, "avg_bias_score": 0.1},
            "overall_score": 75.0,
        }

        # First run saves baseline
        runner.run_comparison(metrics, model_name="test")

        # Second run with same metrics should pass
        result = runner.run_comparison(metrics, model_name="test")
        assert result["status"] == "PASS"
        assert len(result["regressions"]) == 0

    def test_regression_detected(self, temp_dir):
        """Degraded metrics should FAIL."""
        runner = RegressionRunner(baseline_dir=temp_dir, degradation_threshold=0.05)

        baseline = {
            "accuracy": {"exact_match_rate": 80.0, "avg_fuzzy_score": 0.85},
            "hallucination": {"hallucination_rate": 10.0},
            "consistency": {"avg_consistency": 0.82},
            "bias": {"bias_rate": 5.0},
            "overall_score": 75.0,
        }
        runner.run_comparison(baseline, model_name="test")

        # Degraded accuracy
        degraded = {
            "accuracy": {"exact_match_rate": 50.0, "avg_fuzzy_score": 0.60},
            "hallucination": {"hallucination_rate": 10.0},
            "consistency": {"avg_consistency": 0.82},
            "bias": {"bias_rate": 5.0},
            "overall_score": 55.0,
        }
        result = runner.run_comparison(degraded, model_name="test")
        assert result["status"] == "FAIL"
        assert len(result["regressions"]) > 0

    def test_improvement_detected(self, temp_dir):
        """Improved metrics should PASS and note improvements."""
        runner = RegressionRunner(baseline_dir=temp_dir, improvement_threshold=0.02)

        baseline = {
            "accuracy": {"exact_match_rate": 70.0},
            "hallucination": {"hallucination_rate": 20.0},
            "consistency": {"avg_consistency": 0.75},
            "bias": {"bias_rate": 10.0},
            "overall_score": 65.0,
        }
        runner.run_comparison(baseline, model_name="test")

        improved = {
            "accuracy": {"exact_match_rate": 90.0},
            "hallucination": {"hallucination_rate": 5.0},
            "consistency": {"avg_consistency": 0.95},
            "bias": {"bias_rate": 2.0},
            "overall_score": 90.0,
        }
        result = runner.run_comparison(improved, model_name="test")
        assert result["status"] == "PASS"
        assert len(result["improvements"]) > 0

    def test_summary_format(self, temp_dir):
        """Verify summary string is generated."""
        runner = RegressionRunner(baseline_dir=temp_dir)
        metrics = {"accuracy": {"exact_match_rate": 80.0}, "overall_score": 75.0}
        runner.run_comparison(metrics, model_name="test")
        result = runner.run_comparison(metrics, model_name="test")
        assert "summary" in result
        assert isinstance(result["summary"], str)
