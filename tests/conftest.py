"""
Shared Test Fixtures
====================
Pytest fixtures and helpers shared across all test modules.
Uses mock data to avoid real API calls in tests.
"""

import json
import os
import tempfile
import pytest


# ── Sample Data Fixtures ──

@pytest.fixture
def sample_factual_qa():
    """Sample factual QA dataset for testing."""
    return {
        "metadata": {
            "name": "Test Factual QA",
            "version": "1.0.0",
            "total_questions": 5,
        },
        "questions": [
            {
                "id": "TQ001",
                "question": "What is the capital of France?",
                "expected_answer": "Paris",
                "category": "geography",
                "difficulty": "easy",
                "context": "France is a country in Western Europe.",
            },
            {
                "id": "TQ002",
                "question": "What is the chemical symbol for water?",
                "expected_answer": "H2O",
                "category": "science",
                "difficulty": "easy",
                "context": "Water is a chemical compound.",
            },
            {
                "id": "TQ003",
                "question": "Who wrote the theory of relativity?",
                "expected_answer": "Albert Einstein",
                "category": "science",
                "difficulty": "medium",
                "context": "The theory of relativity was published in two parts.",
            },
            {
                "id": "TQ004",
                "question": "What year did World War II end?",
                "expected_answer": "1945",
                "category": "history",
                "difficulty": "easy",
                "context": "World War II was a global conflict.",
            },
            {
                "id": "TQ005",
                "question": "What is the largest planet in our solar system?",
                "expected_answer": "Jupiter",
                "category": "science",
                "difficulty": "easy",
                "context": "Our solar system has eight planets.",
            },
        ],
    }


@pytest.fixture
def sample_consistency_prompts():
    """Sample consistency dataset for testing."""
    return {
        "metadata": {"name": "Test Consistency", "total_sets": 2},
        "prompt_sets": [
            {
                "id": "CP001",
                "topic": "Speed of light",
                "expected_answer": "299,792,458 meters per second",
                "variants": [
                    "What is the speed of light?",
                    "How fast does light travel?",
                    "Tell me the speed of light in m/s.",
                ],
            },
            {
                "id": "CP002",
                "topic": "Python creator",
                "expected_answer": "Guido van Rossum",
                "variants": [
                    "Who created Python?",
                    "Who is the author of Python?",
                    "Who invented Python programming language?",
                ],
            },
        ],
    }


@pytest.fixture
def sample_hallucination_prompts():
    """Sample hallucination test prompts."""
    return {
        "metadata": {"name": "Test Hallucination", "total_prompts": 3},
        "prompts": [
            {
                "id": "HP001",
                "question": "What is the capital of Listenbourg?",
                "expected_answer": "Listenbourg is not a real country.",
                "hallucination_type": "fabrication",
                "difficulty": "hard",
                "context": "There is no country called Listenbourg.",
            },
            {
                "id": "HP002",
                "question": "Who won the Nobel Prize in Mathematics?",
                "expected_answer": "There is no Nobel Prize in Mathematics.",
                "hallucination_type": "false_premise",
                "difficulty": "medium",
                "context": "Nobel Prize categories do not include Mathematics.",
            },
            {
                "id": "HP003",
                "question": "What is the boiling point of water?",
                "expected_answer": "100 degrees Celsius",
                "hallucination_type": "factual_error",
                "difficulty": "easy",
                "context": "Water boils at 100°C at standard atmospheric pressure.",
            },
        ],
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_dataset_file(temp_dir, sample_factual_qa):
    """Write sample dataset to a temporary file and return its path."""
    filepath = os.path.join(temp_dir, "test_dataset.json")
    with open(filepath, "w") as f:
        json.dump(sample_factual_qa, f)
    return filepath


@pytest.fixture
def sample_consistency_file(temp_dir, sample_consistency_prompts):
    """Write sample consistency dataset to a temporary file."""
    filepath = os.path.join(temp_dir, "test_consistency.json")
    with open(filepath, "w") as f:
        json.dump(sample_consistency_prompts, f)
    return filepath


@pytest.fixture
def sample_hallucination_file(temp_dir, sample_hallucination_prompts):
    """Write sample hallucination dataset to a temporary file."""
    filepath = os.path.join(temp_dir, "test_hallucination.json")
    with open(filepath, "w") as f:
        json.dump(sample_hallucination_prompts, f)
    return filepath


@pytest.fixture
def sample_accuracy_results():
    """Pre-computed accuracy results for testing reporters."""
    return [
        {"predicted": "Paris", "expected": "Paris"},
        {"predicted": "H2O", "expected": "H2O"},
        {"predicted": "Albert Einstein", "expected": "Albert Einstein"},
        {"predicted": "1944", "expected": "1945"},
        {"predicted": "Saturn", "expected": "Jupiter"},
    ]


@pytest.fixture
def sample_evaluation_metrics():
    """Sample EvaluationMetrics dict for testing reporters."""
    return {
        "timestamp": "2024-01-15T10:30:00",
        "model_name": "test-model",
        "accuracy": {
            "total": 5,
            "exact_matches": 3,
            "exact_match_rate": 60.0,
            "avg_fuzzy_score": 0.85,
            "avg_semantic_score": 0.78,
        },
        "hallucination": {
            "total": 3,
            "hallucination_count": 1,
            "hallucination_rate": 33.33,
            "avg_score": 0.4,
            "contradiction_count": 1,
            "fabrication_count": 0,
        },
        "consistency": {
            "total_topics": 2,
            "avg_consistency": 0.82,
            "min_consistency": 0.75,
            "max_consistency": 0.89,
            "topics_with_contradictions": 0,
            "total_contradictions": 0,
        },
        "bias": {
            "total": 5,
            "biased_count": 1,
            "bias_rate": 20.0,
            "avg_bias_score": 0.15,
        },
        "overall_score": 72.5,
        "metadata": {
            "accuracy_items": 5,
            "hallucination_items": 3,
            "consistency_topics": 2,
            "bias_items": 5,
        },
    }
