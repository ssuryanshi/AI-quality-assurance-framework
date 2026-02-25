"""
Regression Testing package for AI QA Framework.
Tracks model performance across versions and detects regressions.
"""

from regression.regression_runner import RegressionRunner
from regression.baseline_manager import BaselineManager

__all__ = ["RegressionRunner", "BaselineManager"]
