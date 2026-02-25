"""Tests for the DatasetLoader module."""

import json
import os
import pytest

from datasets.dataset_loader import DatasetLoader


class TestDatasetLoaderInit:
    """Test DatasetLoader initialization and file loading."""

    def test_load_valid_dataset(self, sample_dataset_file):
        """Test loading a valid factual QA dataset."""
        loader = DatasetLoader(sample_dataset_file)
        assert len(loader) == 5
        assert loader.dataset_type == "factual_qa"

    def test_load_consistency_dataset(self, sample_consistency_file):
        """Test loading a consistency prompts dataset."""
        loader = DatasetLoader(sample_consistency_file)
        assert len(loader) == 2
        assert loader.dataset_type == "consistency"

    def test_load_hallucination_dataset(self, sample_hallucination_file):
        """Test loading a hallucination test dataset."""
        loader = DatasetLoader(sample_hallucination_file)
        assert len(loader) == 3
        assert loader.dataset_type == "hallucination"

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            DatasetLoader("nonexistent_file.json")

    def test_invalid_json_structure(self, temp_dir):
        """Test that ValueError is raised for invalid structure."""
        filepath = os.path.join(temp_dir, "invalid.json")
        with open(filepath, "w") as f:
            json.dump({"no_questions_key": "invalid"}, f)

        with pytest.raises(ValueError, match="must contain"):
            DatasetLoader(filepath)


class TestDatasetLoaderAccess:
    """Test dataset item access and iteration."""

    def test_iteration(self, sample_dataset_file):
        """Test iterating over all items."""
        loader = DatasetLoader(sample_dataset_file)
        items = list(loader)
        assert len(items) == 5
        assert items[0]["id"] == "TQ001"

    def test_indexing(self, sample_dataset_file):
        """Test accessing items by index."""
        loader = DatasetLoader(sample_dataset_file)
        assert loader[0]["question"] == "What is the capital of France?"
        assert loader[2]["id"] == "TQ003"

    def test_metadata(self, sample_dataset_file):
        """Test metadata access."""
        loader = DatasetLoader(sample_dataset_file)
        assert loader.metadata["name"] == "Test Factual QA"
        assert loader.metadata["total_questions"] == 5

    def test_get_by_id(self, sample_dataset_file):
        """Test retrieving item by ID."""
        loader = DatasetLoader(sample_dataset_file)
        item = loader.get_by_id("TQ003")
        assert item is not None
        assert item["question"] == "Who wrote the theory of relativity?"

    def test_get_by_id_not_found(self, sample_dataset_file):
        """Test that get_by_id returns None for unknown IDs."""
        loader = DatasetLoader(sample_dataset_file)
        assert loader.get_by_id("INVALID") is None


class TestDatasetLoaderFiltering:
    """Test dataset filtering capabilities."""

    def test_filter_by_category(self, sample_dataset_file):
        """Test filtering by category."""
        loader = DatasetLoader(sample_dataset_file)
        science = loader.filter_by_category("science")
        assert len(science) == 3
        assert all(item["category"] == "science" for item in science)

    def test_filter_by_difficulty(self, sample_dataset_file):
        """Test filtering by difficulty."""
        loader = DatasetLoader(sample_dataset_file)
        easy = loader.filter_by_difficulty("easy")
        assert len(easy) == 4

    def test_filter_by_custom_predicate(self, sample_dataset_file):
        """Test filtering with a custom predicate function."""
        loader = DatasetLoader(sample_dataset_file)
        hard_science = loader.filter_by(
            lambda x: x.get("category") == "science" and x.get("difficulty") == "medium"
        )
        assert len(hard_science) == 1
        assert hard_science[0]["id"] == "TQ003"

    def test_get_categories(self, sample_dataset_file):
        """Test getting all unique categories."""
        loader = DatasetLoader(sample_dataset_file)
        categories = loader.get_categories()
        assert "science" in categories
        assert "geography" in categories
        assert "history" in categories

    def test_get_difficulties(self, sample_dataset_file):
        """Test getting all unique difficulty levels."""
        loader = DatasetLoader(sample_dataset_file)
        difficulties = loader.get_difficulties()
        assert "easy" in difficulties
        assert "medium" in difficulties


class TestDatasetLoaderValidation:
    """Test schema validation."""

    def test_valid_schema(self, sample_dataset_file):
        """Test that valid datasets pass validation."""
        loader = DatasetLoader(sample_dataset_file)
        errors = loader.validate_schema()
        assert len(errors) == 0

    def test_summary_stats(self, sample_dataset_file):
        """Test summary statistics generation."""
        loader = DatasetLoader(sample_dataset_file)
        summary = loader.summary()
        assert summary["total_items"] == 5
        assert summary["dataset_type"] == "factual_qa"
        assert "science" in summary["by_category"]
        assert summary["by_category"]["science"] == 3

    def test_repr(self, sample_dataset_file):
        """Test string representation."""
        loader = DatasetLoader(sample_dataset_file)
        repr_str = repr(loader)
        assert "DatasetLoader" in repr_str
        assert "items=5" in repr_str
