"""
Visual Reporter
===============
Generates matplotlib charts from evaluation metrics.
Creates publication-ready PNG plots for dashboards and reports.

Charts generated:
    - Accuracy bar chart by category
    - Hallucination rate pie chart
    - Consistency score distribution
    - Regression comparison (baseline vs current)
    - Overall metrics radar/summary chart

Usage:
    reporter = VisualReporter("reports/charts")
    reporter.plot_accuracy_chart(accuracy_data)
    reporter.plot_hallucination_chart(hallucination_data)
"""

import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class VisualReporter:
    """
    Generates matplotlib visualization charts from evaluation metrics.

    All charts are saved as PNG files to the specified output directory.
    """

    # Color palette for consistent styling
    COLORS = {
        "primary": "#2196F3",
        "success": "#4CAF50",
        "warning": "#FF9800",
        "danger": "#F44336",
        "info": "#00BCD4",
        "purple": "#9C27B0",
        "dark": "#37474F",
        "light": "#ECEFF1",
        "accent": "#FF5722",
    }

    CATEGORY_COLORS = [
        "#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0",
        "#00BCD4", "#795548", "#607D8B", "#E91E63", "#3F51B5",
    ]

    def __init__(self, output_dir: str = "reports/charts", dpi: int = 150):
        """
        Initialize the visual reporter.

        Args:
            output_dir: Directory to save chart PNG files.
            dpi: Resolution for saved charts.
        """
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)

    def plot_accuracy_chart(
        self,
        accuracy_data: Dict[str, Any],
        filename: str = "accuracy_metrics.png",
    ) -> str:
        """
        Create a bar chart showing accuracy metrics.

        Args:
            accuracy_data: Accuracy metrics dict from calculate_accuracy().
            filename: Output filename.

        Returns:
            Path to the saved chart.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Accuracy Metrics", fontsize=16, fontweight="bold")

        # Chart 1: Score comparison
        ax1 = axes[0]
        metrics = {
            "Exact Match\nRate (%)": accuracy_data.get("exact_match_rate", 0),
            "Avg Fuzzy\nScore (×100)": accuracy_data.get("avg_fuzzy_score", 0) * 100,
            "Avg Semantic\nScore (×100)": accuracy_data.get("avg_semantic_score", 0) * 100,
        }
        bars = ax1.bar(
            metrics.keys(), metrics.values(),
            color=[self.COLORS["primary"], self.COLORS["info"], self.COLORS["purple"]],
            edgecolor="white", linewidth=1.5,
        )
        ax1.set_ylim(0, 105)
        ax1.set_ylabel("Score", fontsize=12)
        ax1.set_title("Score Comparison", fontsize=13)
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f"{height:.1f}", ha="center", va="bottom", fontsize=11)

        # Chart 2: Match breakdown
        ax2 = axes[1]
        total = accuracy_data.get("total", 0)
        exact = accuracy_data.get("exact_matches", 0)
        non_exact = total - exact
        if total > 0:
            ax2.pie(
                [exact, non_exact],
                labels=[f"Exact Match\n({exact})", f"Non-Exact\n({non_exact})"],
                colors=[self.COLORS["success"], self.COLORS["warning"]],
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 11},
            )
        ax2.set_title(f"Match Distribution (n={total})", fontsize=13)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Accuracy chart saved: {filepath}")
        return filepath

    def plot_hallucination_chart(
        self,
        hallucination_data: Dict[str, Any],
        filename: str = "hallucination_metrics.png",
    ) -> str:
        """
        Create charts showing hallucination detection results.

        Args:
            hallucination_data: Hallucination metrics from calculate_hallucination_rate().
            filename: Output filename.

        Returns:
            Path to the saved chart.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Hallucination Detection Results", fontsize=16, fontweight="bold")

        # Chart 1: Hallucination rate pie chart
        ax1 = axes[0]
        total = hallucination_data.get("total", 0)
        hall_count = hallucination_data.get("hallucination_count", 0)
        clean_count = total - hall_count

        if total > 0:
            ax1.pie(
                [clean_count, hall_count],
                labels=[f"Clean\n({clean_count})", f"Hallucinated\n({hall_count})"],
                colors=[self.COLORS["success"], self.COLORS["danger"]],
                autopct="%1.1f%%",
                startangle=90,
                explode=(0, 0.05),
                textprops={"fontsize": 11},
            )
        ax1.set_title(f"Hallucination Rate (n={total})", fontsize=13)

        # Chart 2: Detection type breakdown
        ax2 = axes[1]
        types = {
            "Contradictions": hallucination_data.get("contradiction_count", 0),
            "Fabrications": hallucination_data.get("fabrication_count", 0),
        }
        bars = ax2.barh(
            list(types.keys()), list(types.values()),
            color=[self.COLORS["warning"], self.COLORS["danger"]],
            edgecolor="white", linewidth=1.5,
        )
        ax2.set_xlabel("Count", fontsize=12)
        ax2.set_title("Hallucination Types", fontsize=13)
        ax2.grid(axis="x", alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                    f"{int(width)}", ha="left", va="center", fontsize=11)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Hallucination chart saved: {filepath}")
        return filepath

    def plot_consistency_chart(
        self,
        consistency_data: Dict[str, Any],
        filename: str = "consistency_metrics.png",
    ) -> str:
        """
        Create a chart showing consistency scores.

        Args:
            consistency_data: Consistency report dict.
            filename: Output filename.

        Returns:
            Path to the saved chart.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Response Consistency Analysis", fontsize=16, fontweight="bold")

        metrics = {
            "Avg\nConsistency": consistency_data.get("avg_consistency", 0),
            "Min\nConsistency": consistency_data.get("min_consistency", 0),
            "Max\nConsistency": consistency_data.get("max_consistency", 0),
        }

        colors = [self.COLORS["primary"], self.COLORS["warning"], self.COLORS["success"]]
        bars = ax.bar(metrics.keys(), metrics.values(), color=colors,
                     edgecolor="white", linewidth=1.5, width=0.5)

        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Consistency Score", fontsize=12)
        ax.axhline(y=0.8, color=self.COLORS["danger"], linestyle="--",
                   alpha=0.7, label="Minimum Threshold (0.80)")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f"{height:.3f}", ha="center", va="bottom", fontsize=12)

        # Add contradiction info as text
        contradictions = consistency_data.get("total_contradictions", 0)
        topics_with = consistency_data.get("topics_with_contradictions", 0)
        total_topics = consistency_data.get("total_topics", 0)
        ax.text(0.95, 0.05,
               f"Topics: {total_topics}\nContradictions: {contradictions}\n"
               f"Topics w/ issues: {topics_with}",
               transform=ax.transAxes, fontsize=10, verticalalignment="bottom",
               horizontalalignment="right",
               bbox=dict(boxstyle="round", facecolor=self.COLORS["light"], alpha=0.8))

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Consistency chart saved: {filepath}")
        return filepath

    def plot_regression_chart(
        self,
        regression_data: Dict[str, Any],
        filename: str = "regression_comparison.png",
    ) -> str:
        """
        Create a comparison chart showing baseline vs current metrics.

        Args:
            regression_data: Dict from RegressionRunner.run_comparison().
            filename: Output filename.

        Returns:
            Path to the saved chart.
        """
        details = regression_data.get("details", [])
        if not details:
            logger.warning("No regression details to plot")
            return ""

        fig, ax = plt.subplots(figsize=(14, 7))
        fig.suptitle("Regression Analysis: Baseline vs Current",
                     fontsize=16, fontweight="bold")

        metric_names = [d["metric"] for d in details]
        baseline_vals = [d["baseline"] for d in details]
        current_vals = [d["current"] for d in details]
        statuses = [d.get("status", "TRACKED") for d in details]

        x = range(len(metric_names))
        width = 0.35

        bars1 = ax.bar([i - width/2 for i in x], baseline_vals, width,
                      label="Baseline", color=self.COLORS["info"], alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], current_vals, width,
                      label="Current", color=self.COLORS["primary"], alpha=0.8)

        # Color-code current bars by status
        status_colors = {
            "REGRESSION": self.COLORS["danger"],
            "IMPROVED": self.COLORS["success"],
            "STABLE": self.COLORS["primary"],
            "TRACKED": self.COLORS["dark"],
        }
        for bar, status in zip(bars2, statuses):
            bar.set_color(status_colors.get(status, self.COLORS["primary"]))

        ax.set_ylabel("Value", fontsize=12)
        ax.set_xticks(list(x))
        ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        # Status legend
        patches = [
            mpatches.Patch(color=self.COLORS["danger"], label="Regression"),
            mpatches.Patch(color=self.COLORS["success"], label="Improved"),
            mpatches.Patch(color=self.COLORS["primary"], label="Stable"),
        ]
        ax.legend(handles=[bars1[0], *patches], loc="upper right", fontsize=9)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Regression chart saved: {filepath}")
        return filepath

    def plot_overall_dashboard(
        self,
        metrics: Dict[str, Any],
        filename: str = "overall_dashboard.png",
    ) -> str:
        """
        Create an overall metrics dashboard combining all evaluation dimensions.

        Args:
            metrics: Full EvaluationMetrics dict.
            filename: Output filename.

        Returns:
            Path to the saved chart.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"AI Model QA Dashboard — {metrics.get('model_name', 'Unknown')}",
            fontsize=18, fontweight="bold", y=1.02,
        )

        # Top-left: Overall score gauge
        ax = axes[0][0]
        overall = metrics.get("overall_score", 0)
        color = (
            self.COLORS["success"] if overall >= 80 else
            self.COLORS["warning"] if overall >= 60 else
            self.COLORS["danger"]
        )
        ax.pie(
            [overall, 100 - overall],
            colors=[color, self.COLORS["light"]],
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.3},
        )
        ax.text(0, 0, f"{overall:.0f}", ha="center", va="center",
               fontsize=36, fontweight="bold", color=color)
        ax.text(0, -0.15, "Overall Score", ha="center", va="center",
               fontsize=12, color=self.COLORS["dark"])

        # Top-right: Key metrics summary
        ax = axes[0][1]
        ax.axis("off")
        acc = metrics.get("accuracy", {})
        hall = metrics.get("hallucination", {})
        cons = metrics.get("consistency", {})
        bias = metrics.get("bias", {})

        summary_text = (
            f"📊 Accuracy\n"
            f"    Exact Match Rate: {acc.get('exact_match_rate', 'N/A')}%\n"
            f"    Avg Semantic:     {acc.get('avg_semantic_score', 'N/A')}\n\n"
            f"🔍 Hallucination\n"
            f"    Rate: {hall.get('hallucination_rate', 'N/A')}%\n"
            f"    Fabrications: {hall.get('fabrication_count', 'N/A')}\n\n"
            f"🔄 Consistency\n"
            f"    Avg Score: {cons.get('avg_consistency', 'N/A')}\n"
            f"    Contradictions: {cons.get('total_contradictions', 'N/A')}\n\n"
            f"⚖️  Bias\n"
            f"    Rate: {bias.get('bias_rate', 'N/A')}%\n"
            f"    Avg Score: {bias.get('avg_bias_score', 'N/A')}"
        )
        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment="top", fontfamily="monospace",
               bbox=dict(boxstyle="round,pad=0.8", facecolor=self.COLORS["light"], alpha=0.9))

        # Bottom-left: Category comparison bars
        ax = axes[1][0]
        categories = ["Accuracy", "Anti-Halluc.", "Consistency", "Anti-Bias"]
        values = [
            acc.get("exact_match_rate", 0),
            100 - hall.get("hallucination_rate", 0),
            cons.get("avg_consistency", 0) * 100,
            100 - bias.get("bias_rate", 0),
        ]
        colors = [self.COLORS["primary"], self.COLORS["success"],
                 self.COLORS["info"], self.COLORS["purple"]]
        bars = ax.barh(categories, values, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Score (%)", fontsize=12)
        ax.set_title("Category Scores", fontsize=13)
        ax.grid(axis="x", alpha=0.3)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                   f"{width:.1f}%", ha="left", va="center", fontsize=10)

        # Bottom-right: Items evaluated
        ax = axes[1][1]
        meta = metrics.get("metadata", {})
        item_labels = ["Accuracy\nItems", "Hallucination\nItems",
                      "Consistency\nTopics", "Bias\nItems"]
        item_counts = [
            meta.get("accuracy_items", 0),
            meta.get("hallucination_items", 0),
            meta.get("consistency_topics", 0),
            meta.get("bias_items", 0),
        ]
        ax.bar(item_labels, item_counts, color=self.CATEGORY_COLORS[:4],
              edgecolor="white", linewidth=1.5)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Items Evaluated", fontsize=13)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Dashboard saved: {filepath}")
        return filepath
