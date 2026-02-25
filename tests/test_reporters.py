"""Tests for the CSV and visual reporting modules."""

import os
import pytest

from reports.csv_reporter import CSVReporter
from reports.visual_reporter import VisualReporter
from reports.report_generator import ReportGenerator


class TestCSVReporter:
    """Test CSVReporter file generation."""

    def test_write_accuracy_report(self, temp_dir):
        """Test writing accuracy per-item CSV."""
        reporter = CSVReporter(os.path.join(temp_dir, "csv"))
        results = [
            {"predicted": "Paris", "expected": "Paris",
             "exact_match": True, "fuzzy_score": 1.0, "semantic_score": 1.0},
            {"predicted": "London", "expected": "Berlin",
             "exact_match": False, "fuzzy_score": 0.3, "semantic_score": 0.2},
        ]
        path = reporter.write_accuracy_report(results)
        assert os.path.exists(path)
        assert path.endswith(".csv")

        # Verify content
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 rows

    def test_write_hallucination_report(self, temp_dir):
        """Test writing hallucination detection CSV."""
        reporter = CSVReporter(os.path.join(temp_dir, "csv"))
        results = [
            {"hallucination_score": 0.1, "is_hallucination": False,
             "contradiction_detected": False, "fabrication_detected": False,
             "hedging_detected": False, "response_relevance": 0.9, "details": "Clean"},
        ]
        path = reporter.write_hallucination_report(results)
        assert os.path.exists(path)

    def test_write_summary_report(self, temp_dir, sample_evaluation_metrics):
        """Test writing summary metrics CSV."""
        reporter = CSVReporter(os.path.join(temp_dir, "csv"))
        path = reporter.write_summary_report(sample_evaluation_metrics)
        assert os.path.exists(path)

    def test_empty_results(self, temp_dir):
        """Empty results should not create a file."""
        reporter = CSVReporter(os.path.join(temp_dir, "csv"))
        path = reporter.write_accuracy_report([])
        assert path == ""

    def test_write_regression_report(self, temp_dir):
        """Test writing regression comparison CSV."""
        reporter = CSVReporter(os.path.join(temp_dir, "csv"))
        regression = {
            "details": [
                {"metric": "exact_match_rate", "baseline": 80.0,
                 "current": 75.0, "change": -5.0, "pct_change": -6.25, "status": "REGRESSION"},
            ]
        }
        path = reporter.write_regression_report(regression)
        assert os.path.exists(path)


class TestVisualReporter:
    """Test VisualReporter chart generation."""

    def test_plot_accuracy_chart(self, temp_dir, sample_evaluation_metrics):
        """Test accuracy chart generation."""
        reporter = VisualReporter(os.path.join(temp_dir, "charts"))
        path = reporter.plot_accuracy_chart(sample_evaluation_metrics["accuracy"])
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_plot_hallucination_chart(self, temp_dir, sample_evaluation_metrics):
        """Test hallucination chart generation."""
        reporter = VisualReporter(os.path.join(temp_dir, "charts"))
        path = reporter.plot_hallucination_chart(sample_evaluation_metrics["hallucination"])
        assert os.path.exists(path)

    def test_plot_consistency_chart(self, temp_dir, sample_evaluation_metrics):
        """Test consistency chart generation."""
        reporter = VisualReporter(os.path.join(temp_dir, "charts"))
        path = reporter.plot_consistency_chart(sample_evaluation_metrics["consistency"])
        assert os.path.exists(path)

    def test_plot_overall_dashboard(self, temp_dir, sample_evaluation_metrics):
        """Test overall dashboard generation."""
        reporter = VisualReporter(os.path.join(temp_dir, "charts"))
        path = reporter.plot_overall_dashboard(sample_evaluation_metrics)
        assert os.path.exists(path)

    def test_plot_regression_chart(self, temp_dir):
        """Test regression comparison chart."""
        reporter = VisualReporter(os.path.join(temp_dir, "charts"))
        regression = {
            "details": [
                {"metric": "accuracy", "baseline": 80.0, "current": 75.0,
                 "change": -5.0, "pct_change": -6.25, "status": "REGRESSION"},
                {"metric": "consistency", "baseline": 0.82, "current": 0.85,
                 "change": 0.03, "pct_change": 3.66, "status": "IMPROVED"},
            ]
        }
        path = reporter.plot_regression_chart(regression)
        assert os.path.exists(path)


class TestReportGenerator:
    """Test ReportGenerator orchestration."""

    def test_generate_all_csv_and_charts(self, temp_dir, sample_evaluation_metrics):
        """Test full report generation."""
        config = {
            "reports": {
                "csv_dir": os.path.join(temp_dir, "csv"),
                "charts_dir": os.path.join(temp_dir, "charts"),
                "generate_csv": True,
                "generate_charts": True,
                "chart_dpi": 72,
            }
        }
        generator = ReportGenerator(config)
        accuracy_items = [
            {"predicted": "Paris", "expected": "Paris",
             "exact_match": True, "fuzzy_score": 1.0, "semantic_score": 1.0},
        ]
        result = generator.generate_all(
            metrics=sample_evaluation_metrics,
            accuracy_per_item=accuracy_items,
        )
        assert len(result["csv"]) > 0
        assert len(result["charts"]) > 0

    def test_csv_only(self, temp_dir, sample_evaluation_metrics):
        """Test generating only CSV (no charts)."""
        config = {
            "reports": {
                "csv_dir": os.path.join(temp_dir, "csv"),
                "generate_csv": True,
                "generate_charts": False,
            }
        }
        generator = ReportGenerator(config)
        result = generator.generate_all(metrics=sample_evaluation_metrics)
        assert len(result["csv"]) > 0
        assert len(result["charts"]) == 0
