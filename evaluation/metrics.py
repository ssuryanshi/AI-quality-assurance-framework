"""
Evaluation Metrics Aggregator
=============================
Defines the EvaluationMetrics dataclass and a function to compute
all metrics from raw evaluation results.

Usage:
    from evaluation.metrics import compute_all_metrics

    metrics = compute_all_metrics(
        accuracy_results=accuracy_data,
        hallucination_results=hallucination_data,
        consistency_results=consistency_data,
        bias_results=bias_data,
    )
    print(metrics)
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from evaluation.accuracy import calculate_accuracy
from evaluation.hallucination import calculate_hallucination_rate
from evaluation.consistency import consistency_report
from evaluation.bias_detector import calculate_bias_score

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """
    Aggregated evaluation metrics from all assessment dimensions.

    Attributes:
        timestamp: When the evaluation was run.
        model_name: Name of the evaluated model.
        accuracy: Accuracy metrics dict.
        hallucination: Hallucination metrics dict.
        consistency: Consistency metrics dict.
        bias: Bias metrics dict.
        overall_score: Weighted composite score (0-100).
        metadata: Additional metadata about the evaluation run.
    """
    timestamp: str = ""
    model_name: str = ""
    accuracy: Dict[str, Any] = field(default_factory=dict)
    hallucination: Dict[str, Any] = field(default_factory=dict)
    consistency: Dict[str, Any] = field(default_factory=dict)
    bias: Dict[str, Any] = field(default_factory=dict)
    overall_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Generate a human-readable summary string."""
        lines = [
            f"═══ Evaluation Summary ═══",
            f"Model: {self.model_name}",
            f"Timestamp: {self.timestamp}",
            f"",
            f"📊 Accuracy:",
            f"   Exact Match Rate: {self.accuracy.get('exact_match_rate', 'N/A')}%",
            f"   Avg Fuzzy Score:  {self.accuracy.get('avg_fuzzy_score', 'N/A')}",
            f"   Avg Semantic:     {self.accuracy.get('avg_semantic_score', 'N/A')}",
            f"",
            f"🔍 Hallucination:",
            f"   Hallucination Rate: {self.hallucination.get('hallucination_rate', 'N/A')}%",
            f"   Avg Score:          {self.hallucination.get('avg_score', 'N/A')}",
            f"   Fabrications:       {self.hallucination.get('fabrication_count', 'N/A')}",
            f"",
            f"🔄 Consistency:",
            f"   Avg Consistency:    {self.consistency.get('avg_consistency', 'N/A')}",
            f"   Contradictions:     {self.consistency.get('total_contradictions', 'N/A')}",
            f"",
            f"⚖️  Bias:",
            f"   Bias Rate:          {self.bias.get('bias_rate', 'N/A')}%",
            f"   Avg Bias Score:     {self.bias.get('avg_bias_score', 'N/A')}",
            f"",
            f"🏆 Overall Score: {self.overall_score}/100",
            f"═══════════════════════════",
        ]
        return "\n".join(lines)


def compute_all_metrics(
    accuracy_results: Optional[List[Dict[str, Any]]] = None,
    hallucination_results: Optional[List[Dict[str, Any]]] = None,
    consistency_results: Optional[List[Dict[str, Any]]] = None,
    bias_results: Optional[List[Dict[str, Any]]] = None,
    model_name: str = "unknown",
) -> EvaluationMetrics:
    """
    Compute all evaluation metrics from raw results.

    Args:
        accuracy_results: List of {"predicted": str, "expected": str} dicts.
        hallucination_results: List of dicts from detect_hallucination().
        consistency_results: List of dicts with consistency_score and contradictions.
        bias_results: List of dicts from detect_bias().
        model_name: Name of the model being evaluated.

    Returns:
        EvaluationMetrics dataclass with all aggregated metrics.
    """
    # Calculate each metric category
    accuracy = calculate_accuracy(accuracy_results or [])
    hallucination = calculate_hallucination_rate(hallucination_results or [])
    consistency = consistency_report(consistency_results or [])
    bias = calculate_bias_score(bias_results or [])

    # Remove per_item from accuracy to keep summary clean
    accuracy_summary = {k: v for k, v in accuracy.items() if k != "per_item"}

    # Calculate overall composite score (0-100)
    overall = _calculate_overall_score(accuracy, hallucination, consistency, bias)

    metrics = EvaluationMetrics(
        timestamp=datetime.now().isoformat(),
        model_name=model_name,
        accuracy=accuracy_summary,
        hallucination=hallucination,
        consistency=consistency,
        bias=bias,
        overall_score=overall,
        metadata={
            "accuracy_items": accuracy.get("total", 0),
            "hallucination_items": hallucination.get("total", 0),
            "consistency_topics": consistency.get("total_topics", 0),
            "bias_items": bias.get("total", 0),
        },
    )

    logger.info(f"Evaluation complete. Overall score: {overall}/100")
    return metrics


def _calculate_overall_score(
    accuracy: Dict[str, Any],
    hallucination: Dict[str, Any],
    consistency: Dict[str, Any],
    bias: Dict[str, Any],
) -> float:
    """
    Calculate a weighted overall quality score (0-100).

    Weights:
        - Accuracy: 35%
        - Hallucination (inverted): 30%
        - Consistency: 20%
        - Bias (inverted): 15%
    """
    scores = []
    weights = []

    # Accuracy component (higher is better)
    if accuracy.get("total", 0) > 0:
        # Combine exact match rate and semantic score
        acc_score = (
            accuracy.get("exact_match_rate", 0) * 0.4 +
            accuracy.get("avg_fuzzy_score", 0) * 100 * 0.3 +
            accuracy.get("avg_semantic_score", 0) * 100 * 0.3
        )
        scores.append(min(acc_score, 100))
        weights.append(0.35)

    # Hallucination component (lower is better, so invert)
    if hallucination.get("total", 0) > 0:
        hall_score = 100 - hallucination.get("hallucination_rate", 0)
        scores.append(max(hall_score, 0))
        weights.append(0.30)

    # Consistency component (higher is better)
    if consistency.get("total_topics", 0) > 0:
        cons_score = consistency.get("avg_consistency", 0) * 100
        scores.append(min(cons_score, 100))
        weights.append(0.20)

    # Bias component (lower is better, so invert)
    if bias.get("total", 0) > 0:
        bias_score = 100 - bias.get("bias_rate", 0)
        scores.append(max(bias_score, 0))
        weights.append(0.15)

    if not scores:
        return 0.0

    # Normalize weights to sum to 1.0
    total_weight = sum(weights)
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    overall = weighted_sum / total_weight

    return round(overall, 2)
