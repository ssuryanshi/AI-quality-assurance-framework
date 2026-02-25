"""
Report Generator
================
Orchestrates CSV and visual report generation from evaluation metrics.

Usage:
    from reports.report_generator import ReportGenerator

    generator = ReportGenerator(config)
    generator.generate_all(metrics, accuracy_per_item, hallucination_per_item)
"""

import logging
from typing import Any, Dict, List, Optional

from reports.csv_reporter import CSVReporter
from reports.visual_reporter import VisualReporter

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Orchestrates the generation of all report types.
    Runs CSV and visual reporters and produces a combined output summary.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the report generator.

        Args:
            config: Configuration dict. Expected keys under 'reports':
                - csv_dir: Directory for CSV files
                - charts_dir: Directory for chart PNGs
                - generate_csv: Boolean to enable CSV reports
                - generate_charts: Boolean to enable visual reports
                - chart_dpi: Resolution for charts
        """
        reports_config = (config or {}).get("reports", {})

        self.generate_csv = reports_config.get("generate_csv", True)
        self.generate_charts = reports_config.get("generate_charts", True)

        csv_dir = reports_config.get("csv_dir", "reports/csv")
        charts_dir = reports_config.get("charts_dir", "reports/charts")
        chart_dpi = reports_config.get("chart_dpi", 150)

        self.csv_reporter = CSVReporter(csv_dir) if self.generate_csv else None
        self.visual_reporter = VisualReporter(charts_dir, chart_dpi) if self.generate_charts else None

    def generate_all(
        self,
        metrics: Dict[str, Any],
        accuracy_per_item: Optional[List[Dict]] = None,
        hallucination_per_item: Optional[List[Dict]] = None,
        regression_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        """
        Generate all configured reports.

        Args:
            metrics: EvaluationMetrics dict (from .to_dict()).
            accuracy_per_item: Per-item accuracy details.
            hallucination_per_item: Per-item hallucination detection details.
            regression_result: Regression comparison result dict.

        Returns:
            Dictionary mapping report type to list of generated file paths.
        """
        generated_files = {"csv": [], "charts": []}

        # ── CSV Reports ──
        if self.csv_reporter:
            logger.info("Generating CSV reports...")

            # Summary
            path = self.csv_reporter.write_summary_report(metrics)
            if path:
                generated_files["csv"].append(path)

            # Accuracy details
            if accuracy_per_item:
                path = self.csv_reporter.write_accuracy_report(accuracy_per_item)
                if path:
                    generated_files["csv"].append(path)

            # Hallucination details
            if hallucination_per_item:
                path = self.csv_reporter.write_hallucination_report(hallucination_per_item)
                if path:
                    generated_files["csv"].append(path)

            # Regression details
            if regression_result:
                path = self.csv_reporter.write_regression_report(regression_result)
                if path:
                    generated_files["csv"].append(path)

        # ── Visual Reports ──
        if self.visual_reporter:
            logger.info("Generating visual reports...")

            # Accuracy chart
            accuracy_data = metrics.get("accuracy", {})
            if accuracy_data.get("total", 0) > 0:
                path = self.visual_reporter.plot_accuracy_chart(accuracy_data)
                if path:
                    generated_files["charts"].append(path)

            # Hallucination chart
            hall_data = metrics.get("hallucination", {})
            if hall_data.get("total", 0) > 0:
                path = self.visual_reporter.plot_hallucination_chart(hall_data)
                if path:
                    generated_files["charts"].append(path)

            # Consistency chart
            cons_data = metrics.get("consistency", {})
            if cons_data.get("total_topics", 0) > 0:
                path = self.visual_reporter.plot_consistency_chart(cons_data)
                if path:
                    generated_files["charts"].append(path)

            # Regression comparison chart
            if regression_result and regression_result.get("details"):
                path = self.visual_reporter.plot_regression_chart(regression_result)
                if path:
                    generated_files["charts"].append(path)

            # Overall dashboard
            path = self.visual_reporter.plot_overall_dashboard(metrics)
            if path:
                generated_files["charts"].append(path)

        total_files = sum(len(v) for v in generated_files.values())
        logger.info(f"Report generation complete: {total_files} files generated")

        return generated_files
