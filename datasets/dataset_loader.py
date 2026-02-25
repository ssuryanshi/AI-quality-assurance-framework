"""
Dataset Loader Module
=====================
Loads, validates, and provides access to JSON test datasets.
Supports filtering by category, difficulty, and custom predicates.

Usage:
    loader = DatasetLoader("datasets/factual_qa.json")
    for item in loader:
        print(item["question"])
"""

import json
import os
from typing import Any, Callable, Dict, Iterator, List, Optional


class DatasetLoader:
    """
    Loads and manages test datasets from JSON files.

    Supports:
        - Schema validation
        - Filtering by category and difficulty
        - Iteration over questions/prompts
        - Summary statistics

    Example:
        >>> loader = DatasetLoader("datasets/factual_qa.json")
        >>> print(loader.metadata)
        >>> for q in loader.filter_by_category("science"):
        ...     print(q["question"])
    """

    # Required fields for each dataset type
    SCHEMA_FIELDS = {
        "factual_qa": ["id", "question", "expected_answer", "category", "difficulty"],
        "consistency": ["id", "topic", "expected_answer", "variants"],
        "hallucination": ["id", "question", "expected_answer", "hallucination_type"],
    }

    def __init__(self, filepath: str):
        """
        Initialize the DatasetLoader with a JSON file path.

        Args:
            filepath: Path to the JSON dataset file.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the dataset fails schema validation.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        self.filepath = filepath
        self._raw_data = self._load_file(filepath)
        self._metadata = self._raw_data.get("metadata", {})
        self._items = self._extract_items()
        self._dataset_type = self._detect_type()

    def _load_file(self, filepath: str) -> Dict[str, Any]:
        """Load and parse a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Dataset must be a JSON object, got {type(data).__name__}")
        return data

    def _extract_items(self) -> List[Dict[str, Any]]:
        """Extract the list of items from the dataset (questions, prompts, or prompt_sets)."""
        for key in ["questions", "prompts", "prompt_sets"]:
            if key in self._raw_data:
                items = self._raw_data[key]
                if not isinstance(items, list):
                    raise ValueError(f"'{key}' must be a list, got {type(items).__name__}")
                return items
        raise ValueError(
            "Dataset must contain 'questions', 'prompts', or 'prompt_sets' key"
        )

    def _detect_type(self) -> str:
        """Detect dataset type based on content structure."""
        if "questions" in self._raw_data:
            first = self._items[0] if self._items else {}
            if "hallucination_type" in first:
                return "hallucination"
            return "factual_qa"
        elif "prompt_sets" in self._raw_data:
            return "consistency"
        elif "prompts" in self._raw_data:
            first = self._items[0] if self._items else {}
            if "hallucination_type" in first:
                return "hallucination"
            return "general"
        return "unknown"

    def validate_schema(self) -> List[str]:
        """
        Validate that all items have the required fields for this dataset type.

        Returns:
            List of validation error messages. Empty list means valid.
        """
        errors = []
        expected_fields = self.SCHEMA_FIELDS.get(self._dataset_type, [])
        if not expected_fields:
            return errors

        for i, item in enumerate(self._items):
            for field in expected_fields:
                if field not in item:
                    errors.append(f"Item {i} missing required field: '{field}'")

        return errors

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return dataset metadata."""
        return self._metadata

    @property
    def dataset_type(self) -> str:
        """Return the detected dataset type."""
        return self._dataset_type

    @property
    def items(self) -> List[Dict[str, Any]]:
        """Return all dataset items."""
        return self._items

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self._items)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all items."""
        return iter(self._items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item by index."""
        return self._items[index]

    def filter_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Filter items by category.

        Args:
            category: Category string to filter by (case-insensitive).

        Returns:
            List of matching items.
        """
        return [
            item for item in self._items
            if item.get("category", "").lower() == category.lower()
        ]

    def filter_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """
        Filter items by difficulty level.

        Args:
            difficulty: Difficulty level ('easy', 'medium', 'hard').

        Returns:
            List of matching items.
        """
        return [
            item for item in self._items
            if item.get("difficulty", "").lower() == difficulty.lower()
        ]

    def filter_by(self, predicate: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """
        Filter items using a custom predicate function.

        Args:
            predicate: A function that takes an item dict and returns True/False.

        Returns:
            List of items where predicate returns True.

        Example:
            >>> hard_science = loader.filter_by(
            ...     lambda x: x.get("category") == "science" and x.get("difficulty") == "hard"
            ... )
        """
        return [item for item in self._items if predicate(item)]

    def get_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single item by its ID.

        Args:
            item_id: The ID string to search for.

        Returns:
            The matching item dict, or None if not found.
        """
        for item in self._items:
            if item.get("id") == item_id:
                return item
        return None

    def get_categories(self) -> List[str]:
        """Return a sorted list of unique categories in the dataset."""
        categories = set()
        for item in self._items:
            cat = item.get("category")
            if cat:
                categories.add(cat)
        return sorted(categories)

    def get_difficulties(self) -> List[str]:
        """Return a sorted list of unique difficulty levels in the dataset."""
        difficulties = set()
        for item in self._items:
            diff = item.get("difficulty")
            if diff:
                difficulties.add(diff)
        return sorted(difficulties)

    def summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics for the dataset.

        Returns:
            Dictionary with total count, counts by category and difficulty.
        """
        stats = {
            "total_items": len(self._items),
            "dataset_type": self._dataset_type,
            "file": self.filepath,
        }

        # Category breakdown
        categories = {}
        for item in self._items:
            cat = item.get("category", "uncategorized")
            categories[cat] = categories.get(cat, 0) + 1
        stats["by_category"] = categories

        # Difficulty breakdown
        difficulties = {}
        for item in self._items:
            diff = item.get("difficulty", "unspecified")
            difficulties[diff] = difficulties.get(diff, 0) + 1
        stats["by_difficulty"] = difficulties

        return stats

    def to_dataframe(self):
        """
        Convert dataset items to a pandas DataFrame.

        Returns:
            pandas.DataFrame with one row per item.
        """
        try:
            import pandas as pd
            return pd.DataFrame(self._items)
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")

    def __repr__(self) -> str:
        return (
            f"DatasetLoader(file='{self.filepath}', "
            f"type='{self._dataset_type}', "
            f"items={len(self._items)})"
        )
