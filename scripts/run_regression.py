"""
Run Regression Script
=====================
CLI entry point for regression testing — compares current model evaluation
against a saved baseline to detect performance changes.

Usage:
    python scripts/run_regression.py
    python scripts/run_regression.py --baseline baselines/gpt-3.5-turbo_20240101.json
"""

import argparse
import logging
import os
import sys
import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_evaluation import (
    load_config, run_accuracy_evaluation, run_hallucination_evaluation,
    run_consistency_evaluation,
)
from datasets.dataset_loader import DatasetLoader
from models.model_factory import create_model
from evaluation.metrics import compute_all_metrics
from evaluation.bias_detector import detect_bias
from regression.regression_runner import RegressionRunner
from reports.report_generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for regression testing."""
    parser = argparse.ArgumentParser(
        description="AI Model QA Framework - Regression Test Runner"
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--baseline", default=None, help="Specific baseline file to compare against")
    parser.add_argument("--no-save", action="store_true", help="Don't save current run as new baseline")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("AI Model QA Framework - Regression Testing")
    logger.info("=" * 60)

    # Initialize model
    model = create_model(config)
    logger.info(f"Model: {model.model_name}")

    # Run evaluations
    datasets_config = config.get("datasets", {})
    base_path = datasets_config.get("base_path", "datasets")

    accuracy_results = []
    hallucination_results = []
    consistency_results = []
    bias_results = []

    # Load and run each evaluation
    factual_path = os.path.join(base_path, datasets_config.get("factual_qa", "factual_qa.json"))
    if os.path.exists(factual_path):
        dataset = DatasetLoader(factual_path)
        accuracy_results, _ = run_accuracy_evaluation(model, dataset)

    hall_path = os.path.join(base_path, datasets_config.get("hallucination_prompts", "hallucination_prompts.json"))
    if os.path.exists(hall_path):
        dataset = DatasetLoader(hall_path)
        hallucination_results, _ = run_hallucination_evaluation(model, dataset)

    cons_path = os.path.join(base_path, datasets_config.get("consistency_prompts", "consistency_prompts.json"))
    if os.path.exists(cons_path):
        dataset = DatasetLoader(cons_path)
        consistency_results, _ = run_consistency_evaluation(model, dataset)

    if accuracy_results:
        bias_results = [
            detect_bias(r.get("predicted", ""), r.get("question", ""))
            for r in accuracy_results
        ]

    # Compute metrics
    metrics = compute_all_metrics(
        accuracy_results=[
            {"predicted": r["predicted"], "expected": r["expected"]}
            for r in accuracy_results
        ] if accuracy_results else None,
        hallucination_results=hallucination_results or None,
        consistency_results=consistency_results or None,
        bias_results=bias_results or None,
        model_name=model.model_name,
    )

    # Run regression comparison
    reg_config = config.get("regression", {})
    runner = RegressionRunner(
        baseline_dir=reg_config.get("baseline_dir", "baselines"),
        degradation_threshold=reg_config.get("degradation_threshold", 0.05),
        improvement_threshold=reg_config.get("improvement_threshold", 0.02),
    )

    result = runner.run_comparison(
        current_metrics=metrics.to_dict(),
        model_name=model.model_name,
        save_as_new_baseline=not args.no_save,
    )

    # Print results
    logger.info("\n" + result.get("summary", "No summary available"))

    # Generate reports including regression charts
    report_gen = ReportGenerator(config)
    report_gen.generate_all(
        metrics=metrics.to_dict(),
        regression_result=result,
    )

    # Exit with appropriate code
    if result["status"] == "FAIL":
        logger.warning("\n⚠️  REGRESSION DETECTED — Exiting with code 1")
        sys.exit(1)
    else:
        logger.info(f"\n✅ Regression check: {result['status']}")
        sys.exit(0)


if __name__ == "__main__":
    main()
