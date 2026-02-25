"""
Baseline Manager
================
Manages baseline metrics files for regression testing.
Saves and loads JSON files with timestamps, model info, and metrics snapshots.

Usage:
    manager = BaselineManager("baselines")
    manager.save_baseline(metrics_dict, model_name="gpt-4")

    baseline = manager.load_latest_baseline("gpt-4")
    print(baseline["metrics"]["accuracy"])
"""

import json
import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaselineManager:
    """
    Manages persistence of evaluation baselines for regression testing.

    Baselines are JSON files stored with timestamps and model version info.
    Supports loading the latest baseline for comparison or a specific version.
    """

    def __init__(self, baseline_dir: str = "baselines"):
        """
        Initialize the baseline manager.

        Args:
            baseline_dir: Directory to store baseline JSON files.
        """
        self.baseline_dir = baseline_dir
        os.makedirs(baseline_dir, exist_ok=True)

    def save_baseline(
        self,
        metrics: Dict[str, Any],
        model_name: str = "unknown",
        version: Optional[str] = None,
    ) -> str:
        """
        Save a metrics snapshot as a baseline.

        Args:
            metrics: Dictionary of evaluation metrics to save.
            model_name: Name of the model.
            version: Optional version string. Auto-generated if not provided.

        Returns:
            Path to the saved baseline file.
        """
        timestamp = datetime.now()
        if version is None:
            version = timestamp.strftime("%Y%m%d_%H%M%S")

        baseline = {
            "model_name": model_name,
            "version": version,
            "timestamp": timestamp.isoformat(),
            "metrics": metrics,
        }

        # Create filename: model_version_timestamp.json
        safe_model_name = model_name.replace("/", "_").replace(" ", "_")
        filename = f"{safe_model_name}_{version}.json"
        filepath = os.path.join(self.baseline_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2, default=str)

        logger.info(f"Baseline saved: {filepath}")
        return filepath

    def load_baseline(self, filepath: str) -> Dict[str, Any]:
        """
        Load a specific baseline file.

        Args:
            filepath: Path to the baseline JSON file.

        Returns:
            Baseline dictionary with model info and metrics.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Baseline file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_latest_baseline(self, model_name: str = "") -> Optional[Dict[str, Any]]:
        """
        Load the most recent baseline file, optionally filtered by model name.

        Args:
            model_name: If provided, only consider baselines for this model.

        Returns:
            Latest baseline dict, or None if no baselines exist.
        """
        baselines = self.list_baselines(model_name)
        if not baselines:
            logger.info(f"No baselines found for model: '{model_name or 'any'}'")
            return None

        # Sort by timestamp (newest first)
        baselines.sort(key=lambda b: b.get("timestamp", ""), reverse=True)
        latest = baselines[0]

        filepath = latest["filepath"]
        return self.load_baseline(filepath)

    def list_baselines(self, model_name: str = "") -> List[Dict[str, Any]]:
        """
        List all available baseline files.

        Args:
            model_name: If provided, filter by model name.

        Returns:
            List of baseline info dicts with filepath, model_name, version, timestamp.
        """
        baselines = []
        safe_filter = model_name.replace("/", "_").replace(" ", "_").lower()

        for filename in os.listdir(self.baseline_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.baseline_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Filter by model name if specified
                if safe_filter:
                    file_model = data.get("model_name", "").replace("/", "_").replace(" ", "_").lower()
                    if safe_filter not in file_model and safe_filter not in filename.lower():
                        continue

                baselines.append({
                    "filepath": filepath,
                    "model_name": data.get("model_name", "unknown"),
                    "version": data.get("version", "unknown"),
                    "timestamp": data.get("timestamp", ""),
                })
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Skipping invalid baseline file {filename}: {e}")

        return baselines

    def delete_baseline(self, filepath: str) -> bool:
        """
        Delete a baseline file.

        Args:
            filepath: Path to the baseline file to delete.

        Returns:
            True if deleted, False if not found.
        """
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted baseline: {filepath}")
            return True
        return False
