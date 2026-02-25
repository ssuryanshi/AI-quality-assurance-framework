"""
Evaluation package for AI QA Framework.
Provides metrics for accuracy, hallucination detection, consistency, and bias.
"""

from evaluation.accuracy import exact_match, fuzzy_match, semantic_similarity, calculate_accuracy
from evaluation.hallucination import detect_hallucination, calculate_hallucination_rate
from evaluation.consistency import measure_consistency, detect_contradictions
from evaluation.bias_detector import detect_bias, calculate_bias_score
from evaluation.metrics import EvaluationMetrics, compute_all_metrics

__all__ = [
    "exact_match", "fuzzy_match", "semantic_similarity", "calculate_accuracy",
    "detect_hallucination", "calculate_hallucination_rate",
    "measure_consistency", "detect_contradictions",
    "detect_bias", "calculate_bias_score",
    "EvaluationMetrics", "compute_all_metrics",
]
