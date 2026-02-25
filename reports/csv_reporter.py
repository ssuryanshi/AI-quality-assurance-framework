"""
CSV Reporter
============
Generates CSV reports from evaluation results.
Outputs both per-item detail files and summary metrics.

Usage:
    reporter = CSVReporter("reports/csv")
    reporter.write_accuracy_report(accuracy_results)
    reporter.write_summary_report(metrics)
"""

import csv
import os
import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class CSVReporter:
    """
    Generates CSV files from evaluation results.

    Creates separate files for:
        - Per-item accuracy details
        - Hallucination detection details
        - Summary metrics overview
    """

    def __init__(self, output_dir: str = "reports/csv"):
        """
        Initialize the CSV reporter.

        Args:
            output_dir: Directory to write CSV files to.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def write_accuracy_report(
        self,
        results: List[Dict[str, Any]],
        filename: str = None,
    ) -> str:
        """
        Write per-item accuracy scores to CSV.

        Args:
            results: List of accuracy result dicts (from calculate_accuracy's per_item).
            filename: Optional custom filename.

        Returns:
            Path to the generated CSV file.
        """
        if not results:
            logger.warning("No accuracy results to write")
            return ""

        filename = filename or f"accuracy_detail_{self._timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)

        fieldnames = ["predicted", "expected", "exact_match", "fuzzy_score", "semantic_score"]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        logger.info(f"Accuracy report written: {filepath} ({len(results)} rows)")
        return filepath

    def write_hallucination_report(
        self,
        results: List[Dict[str, Any]],
        filename: str = None,
    ) -> str:
        """
        Write hallucination detection details to CSV.

        Args:
            results: List of hallucination detection result dicts.
            filename: Optional custom filename.

        Returns:
            Path to the generated CSV file.
        """
        if not results:
            logger.warning("No hallucination results to write")
            return ""

        filename = filename or f"hallucination_detail_{self._timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)

        fieldnames = [
            "hallucination_score", "is_hallucination", "contradiction_detected",
            "fabrication_detected", "hedging_detected", "response_relevance", "details",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        logger.info(f"Hallucination report written: {filepath} ({len(results)} rows)")
        return filepath

    def write_summary_report(
        self,
        metrics: Dict[str, Any],
        filename: str = None,
    ) -> str:
        """
        Write a summary metrics CSV with one row per metric category.

        Args:
            metrics: EvaluationMetrics dict (from .to_dict()).
            filename: Optional custom filename.

        Returns:
            Path to the generated CSV file.
        """
        filename = filename or f"summary_{self._timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)

        rows = []

        # Flatten metrics into rows
        for category in ["accuracy", "hallucination", "consistency", "bias"]:
            cat_data = metrics.get(category, {})
            if isinstance(cat_data, dict):
                for key, value in cat_data.items():
                    if isinstance(value, (int, float, str, bool)):
                        rows.append({
                            "category": category,
                            "metric": key,
                            "value": value,
                        })

        # Add overall score
        rows.append({
            "category": "overall",
            "metric": "overall_score",
            "value": metrics.get("overall_score", "N/A"),
        })

        # Add metadata
        rows.append({
            "category": "meta",
            "metric": "model_name",
            "value": metrics.get("model_name", "unknown"),
        })
        rows.append({
            "category": "meta",
            "metric": "timestamp",
            "value": metrics.get("timestamp", ""),
        })

        fieldnames = ["category", "metric", "value"]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        logger.info(f"Summary report written: {filepath} ({len(rows)} rows)")
        return filepath

    def write_regression_report(
        self,
        regression_result: Dict[str, Any],
        filename: str = None,
    ) -> str:
        """
        Write regression comparison details to CSV.

        Args:
            regression_result: Dict from RegressionRunner.run_comparison().
            filename: Optional custom filename.

        Returns:
            Path to the generated CSV file.
        """
        filename = filename or f"regression_{self._timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)

        details = regression_result.get("details", [])
        if not details:
            logger.warning("No regression details to write")
            return ""

        fieldnames = ["metric", "baseline", "current", "change", "pct_change", "status"]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in details:
                writer.writerow(row)

        logger.info(f"Regression report written: {filepath} ({len(details)} rows)")
        return filepath
