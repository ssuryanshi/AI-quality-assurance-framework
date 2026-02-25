"""
Run Evaluation Script
=====================
Main CLI entry point for running the AI QA evaluation pipeline.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --dataset datasets/factual_qa.json
    python scripts/run_evaluation.py --model openai --output reports/
"""

import argparse
import logging
import os
import sys
import yaml
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_loader import DatasetLoader
from models.model_factory import create_model
from evaluation.accuracy import calculate_accuracy
from evaluation.hallucination import detect_hallucination, calculate_hallucination_rate
from evaluation.consistency import measure_consistency, detect_contradictions, consistency_report
from evaluation.bias_detector import detect_bias, calculate_bias_score
from evaluation.metrics import compute_all_metrics
from reports.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_accuracy_evaluation(model, dataset: DatasetLoader) -> tuple:
    """
    Run accuracy evaluation on a factual QA dataset.

    Returns:
        Tuple of (accuracy_results_for_metrics, per_item_details)
    """
    logger.info(f"Running accuracy evaluation on {len(dataset)} questions...")
    results = []

    for item in dataset:
        question = item["question"]
        expected = item["expected_answer"]

        try:
            predicted = model.query(question)
        except Exception as e:
            logger.error(f"Query failed for '{question[:50]}...': {e}")
            predicted = f"[ERROR: {e}]"

        results.append({
            "id": item.get("id", ""),
            "question": question,
            "predicted": predicted,
            "expected": expected,
            "category": item.get("category", ""),
        })

    accuracy = calculate_accuracy(results)
    return results, accuracy


def run_hallucination_evaluation(model, dataset: DatasetLoader) -> tuple:
    """
    Run hallucination detection on a hallucination test dataset.

    Returns:
        Tuple of (detection_results, aggregate_rate)
    """
    logger.info(f"Running hallucination evaluation on {len(dataset)} prompts...")
    detection_results = []

    for item in dataset:
        question = item["question"]
        expected = item["expected_answer"]
        context = item.get("context", "")

        try:
            response = model.query(question)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            response = f"[ERROR: {e}]"

        result = detect_hallucination(
            response=response,
            context=context,
            expected=expected,
            hallucination_type=item.get("hallucination_type"),
        )
        result["question"] = question
        result["response"] = response
        detection_results.append(result)

    rate = calculate_hallucination_rate(detection_results)
    return detection_results, rate


def run_consistency_evaluation(model, dataset: DatasetLoader) -> tuple:
    """
    Run consistency evaluation on a set of rephrased prompts.

    Returns:
        Tuple of (per_topic_results, summary_report)
    """
    logger.info(f"Running consistency evaluation on {len(dataset)} prompt sets...")
    topic_results = []

    for prompt_set in dataset:
        variants = prompt_set.get("variants", [])
        topic = prompt_set.get("topic", "unknown")

        responses = []
        for variant in variants:
            try:
                response = model.query(variant)
                responses.append(response)
            except Exception as e:
                logger.error(f"Query failed for variant: {e}")
                responses.append(f"[ERROR: {e}]")

        score = measure_consistency(responses)
        contradictions = detect_contradictions(responses)

        topic_results.append({
            "topic": topic,
            "consistency_score": score,
            "contradictions": contradictions,
            "num_variants": len(variants),
        })

    report = consistency_report(topic_results)
    return topic_results, report


def run_bias_evaluation(model, dataset: DatasetLoader) -> tuple:
    """
    Run bias detection on responses.

    Returns:
        Tuple of (per_item_results, aggregate_scores)
    """
    logger.info(f"Running bias evaluation on {len(dataset)} prompts...")
    bias_results = []

    for item in dataset:
        question = item["question"]
        try:
            response = model.query(question)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            response = f"[ERROR: {e}]"

        result = detect_bias(response, question)
        result["question"] = question
        bias_results.append(result)

    scores = calculate_bias_score(bias_results)
    return bias_results, scores


def main():
    """Main entry point for the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="AI Model QA & Hallucination Detection Framework - Evaluation Runner"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Specific dataset to evaluate (default: run all)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Model provider override: 'openai' or 'huggingface'"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for reports"
    )
    parser.add_argument(
        "--skip-charts", action="store_true",
        help="Skip visual chart generation"
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load config
    config = load_config(args.config)

    # Override model provider if specified
    if args.model:
        config["model"]["provider"] = args.model

    # Override output directory if specified
    if args.output:
        config["reports"]["output_dir"] = args.output
        config["reports"]["csv_dir"] = os.path.join(args.output, "csv")
        config["reports"]["charts_dir"] = os.path.join(args.output, "charts")

    if args.skip_charts:
        config["reports"]["generate_charts"] = False

    logger.info("=" * 60)
    logger.info("AI Model QA Framework - Starting Evaluation")
    logger.info("=" * 60)

    # Initialize model
    logger.info(f"Initializing model: {config['model']['provider']}")
    model = create_model(config)
    logger.info(f"Model ready: {model.model_name}")

    # Load datasets
    datasets_config = config.get("datasets", {})
    base_path = datasets_config.get("base_path", "datasets")

    # Run evaluations based on available datasets
    accuracy_results = []
    accuracy_per_item = []
    hallucination_results = []
    consistency_results = []
    bias_results = []

    # Accuracy evaluation
    factual_qa_path = os.path.join(base_path, datasets_config.get("factual_qa", "factual_qa.json"))
    if os.path.exists(factual_qa_path):
        dataset = DatasetLoader(factual_qa_path)
        items, accuracy = run_accuracy_evaluation(model, dataset)
        accuracy_results = items
        accuracy_per_item = accuracy.get("per_item", [])

    # Hallucination evaluation
    hall_path = os.path.join(base_path, datasets_config.get("hallucination_prompts", "hallucination_prompts.json"))
    if os.path.exists(hall_path):
        dataset = DatasetLoader(hall_path)
        hallucination_results, _ = run_hallucination_evaluation(model, dataset)

    # Consistency evaluation
    cons_path = os.path.join(base_path, datasets_config.get("consistency_prompts", "consistency_prompts.json"))
    if os.path.exists(cons_path):
        dataset = DatasetLoader(cons_path)
        consistency_results, _ = run_consistency_evaluation(model, dataset)

    # Bias evaluation (run on factual QA dataset)
    if os.path.exists(factual_qa_path) and accuracy_results:
        bias_results = []
        for item in accuracy_results:
            result = detect_bias(item.get("predicted", ""), item.get("question", ""))
            bias_results.append(result)

    # Compute aggregated metrics
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

    # Print summary
    logger.info("\n" + metrics.summary())

    # Generate reports
    report_gen = ReportGenerator(config)
    generated = report_gen.generate_all(
        metrics=metrics.to_dict(),
        accuracy_per_item=accuracy_per_item,
        hallucination_per_item=hallucination_results,
    )

    logger.info("\n📁 Generated Reports:")
    for report_type, files in generated.items():
        for f in files:
            logger.info(f"   [{report_type}] {f}")

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
