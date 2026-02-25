"""
Regression Runner
=================
Compares current evaluation metrics against a saved baseline to detect
performance regressions or improvements after model updates.

Usage:
    runner = RegressionRunner(baseline_dir="baselines", degradation_threshold=0.05)
    result = runner.run_comparison(current_metrics, model_name="gpt-4")
    print(result["status"])  # "PASS", "FAIL", or "NO_BASELINE"
"""

import logging
from typing import Any, Dict, List, Optional

from regression.baseline_manager import BaselineManager

logger = logging.getLogger(__name__)


class RegressionRunner:
    """
    Compares current evaluation metrics against baseline to detect regressions.

    A regression is flagged when any key metric drops below the baseline
    by more than the degradation threshold.
    """

    # Metrics where higher is better
    HIGHER_BETTER = [
        "exact_match_rate", "avg_fuzzy_score", "avg_semantic_score",
        "avg_consistency",
    ]

    # Metrics where lower is better
    LOWER_BETTER = [
        "hallucination_rate", "avg_score", "bias_rate", "avg_bias_score",
        "total_contradictions",
    ]

    def __init__(
        self,
        baseline_dir: str = "baselines",
        degradation_threshold: float = 0.05,
        improvement_threshold: float = 0.02,
    ):
        """
        Initialize the regression runner.

        Args:
            baseline_dir: Directory for baseline storage.
            degradation_threshold: Flag regression if metric drops by more than this (0.05 = 5%).
            improvement_threshold: Note improvement if metric improves by more than this.
        """
        self.baseline_manager = BaselineManager(baseline_dir)
        self.degradation_threshold = degradation_threshold
        self.improvement_threshold = improvement_threshold

    def run_comparison(
        self,
        current_metrics: Dict[str, Any],
        model_name: str = "unknown",
        save_as_new_baseline: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare current metrics against the latest baseline.

        Args:
            current_metrics: Current evaluation metrics dict (from EvaluationMetrics.to_dict()).
            model_name: Model name for baseline lookup.
            save_as_new_baseline: If True, save current metrics as new baseline after comparison.

        Returns:
            Dictionary with:
                - status: "PASS", "FAIL", or "NO_BASELINE"
                - regressions: List of metrics that degraded
                - improvements: List of metrics that improved
                - unchanged: List of stable metrics
                - details: Per-metric comparison details
        """
        # Load the latest baseline
        baseline = self.baseline_manager.load_latest_baseline(model_name)

        if baseline is None:
            logger.info("No baseline found. Saving current metrics as first baseline.")
            if save_as_new_baseline:
                self.baseline_manager.save_baseline(current_metrics, model_name)
            return {
                "status": "NO_BASELINE",
                "message": "First run — no baseline to compare. Current metrics saved as baseline.",
                "regressions": [],
                "improvements": [],
                "unchanged": [],
                "details": [],
            }

        baseline_metrics = baseline.get("metrics", {})

        # Compare metrics across all categories
        regressions = []
        improvements = []
        unchanged = []
        details = []

        comparison_pairs = self._extract_comparison_pairs(
            current_metrics, baseline_metrics
        )

        for metric_name, current_val, baseline_val in comparison_pairs:
            if current_val is None or baseline_val is None:
                continue

            try:
                current_val = float(current_val)
                baseline_val = float(baseline_val)
            except (ValueError, TypeError):
                continue

            diff = current_val - baseline_val
            pct_change = (diff / baseline_val * 100) if baseline_val != 0 else 0

            detail = {
                "metric": metric_name,
                "baseline": round(baseline_val, 4),
                "current": round(current_val, 4),
                "change": round(diff, 4),
                "pct_change": round(pct_change, 2),
            }

            # Determine if this is a regression, improvement, or unchanged
            is_higher_better = metric_name in self.HIGHER_BETTER
            is_lower_better = metric_name in self.LOWER_BETTER

            if is_higher_better:
                if diff < -self.degradation_threshold * baseline_val and baseline_val > 0:
                    detail["status"] = "REGRESSION"
                    regressions.append(detail)
                elif diff > self.improvement_threshold * baseline_val:
                    detail["status"] = "IMPROVED"
                    improvements.append(detail)
                else:
                    detail["status"] = "STABLE"
                    unchanged.append(detail)
            elif is_lower_better:
                if diff > self.degradation_threshold * baseline_val and baseline_val > 0:
                    detail["status"] = "REGRESSION"
                    regressions.append(detail)
                elif diff < -self.improvement_threshold * baseline_val:
                    detail["status"] = "IMPROVED"
                    improvements.append(detail)
                else:
                    detail["status"] = "STABLE"
                    unchanged.append(detail)
            else:
                detail["status"] = "TRACKED"
                unchanged.append(detail)

            details.append(detail)

        # Determine overall status
        status = "FAIL" if regressions else "PASS"

        # Save new baseline if requested
        if save_as_new_baseline:
            self.baseline_manager.save_baseline(current_metrics, model_name)

        result = {
            "status": status,
            "baseline_version": baseline.get("version", "unknown"),
            "baseline_timestamp": baseline.get("timestamp", ""),
            "regressions": regressions,
            "improvements": improvements,
            "unchanged": unchanged,
            "details": details,
            "summary": self._build_summary(status, regressions, improvements),
        }

        logger.info(f"Regression check: {status} ({len(regressions)} regressions, "
                    f"{len(improvements)} improvements)")
        return result

    def _extract_comparison_pairs(
        self,
        current: Dict[str, Any],
        baseline: Dict[str, Any],
    ) -> List[tuple]:
        """
        Extract metric name/value pairs for comparison from nested dicts.

        Returns:
            List of (metric_name, current_value, baseline_value) tuples.
        """
        pairs = []
        categories = ["accuracy", "hallucination", "consistency", "bias"]

        for category in categories:
            current_cat = current.get(category, {})
            baseline_cat = baseline.get(category, {})

            if isinstance(current_cat, dict) and isinstance(baseline_cat, dict):
                all_keys = set(current_cat.keys()) | set(baseline_cat.keys())
                for key in all_keys:
                    c_val = current_cat.get(key)
                    b_val = baseline_cat.get(key)
                    if isinstance(c_val, (int, float)) and isinstance(b_val, (int, float)):
                        pairs.append((key, c_val, b_val))

        # Also compare overall_score
        c_overall = current.get("overall_score")
        b_overall = baseline.get("overall_score")
        if c_overall is not None and b_overall is not None:
            pairs.append(("overall_score", c_overall, b_overall))

        return pairs

    def _build_summary(
        self,
        status: str,
        regressions: List[Dict],
        improvements: List[Dict],
    ) -> str:
        """Build a human-readable summary of the regression check."""
        lines = [f"Regression Check: {status}"]

        if regressions:
            lines.append(f"\n⚠️  REGRESSIONS DETECTED ({len(regressions)}):")
            for r in regressions:
                lines.append(
                    f"   ↓ {r['metric']}: {r['baseline']} → {r['current']} "
                    f"({r['pct_change']:+.2f}%)"
                )

        if improvements:
            lines.append(f"\n✅ IMPROVEMENTS ({len(improvements)}):")
            for imp in improvements:
                lines.append(
                    f"   ↑ {imp['metric']}: {imp['baseline']} → {imp['current']} "
                    f"({imp['pct_change']:+.2f}%)"
                )

        if not regressions and not improvements:
            lines.append("\n→ All metrics stable. No significant changes detected.")

        return "\n".join(lines)
