"""
Base Model Interface
====================
Abstract base class that all model adapters must implement.
Provides retry logic, timeout handling, and a standard query interface.

All model adapters (OpenAI, HuggingFace, etc.) inherit from this class.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for LLM model adapters.

    Subclasses must implement:
        - _call_api(prompt, **kwargs) -> str
        - model_name (property)

    Provides:
        - Automatic retry with exponential backoff
        - Timeout handling
        - Response logging

    Example:
        class MyModel(BaseModel):
            @property
            def model_name(self) -> str:
                return "my-model-v1"

            def _call_api(self, prompt, **kwargs) -> str:
                # Call your API and return the response text
                return "response"
    """

    def __init__(
        self,
        max_retries: int = 3,
        timeout: int = 30,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the base model.

        Args:
            max_retries: Maximum number of retry attempts on failure.
            timeout: Timeout in seconds for API calls.
            retry_delay: Initial delay between retries (doubles each retry).
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay = retry_delay
        self._call_count = 0
        self._error_count = 0

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name/identifier of the model."""
        pass

    @abstractmethod
    def _call_api(self, prompt: str, **kwargs) -> str:
        """
        Make the actual API call to the model.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional model-specific parameters.

        Returns:
            The model's response as a string.

        Raises:
            Exception: On API errors (will be caught by retry logic).
        """
        pass

    def query(self, prompt: str, **kwargs) -> str:
        """
        Send a prompt to the model and return the response.
        Includes automatic retry logic with exponential backoff.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional parameters passed to _call_api.

        Returns:
            The model's text response.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        last_exception = None
        delay = self.retry_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    f"[{self.model_name}] Query attempt {attempt}/{self.max_retries}"
                )
                self._call_count += 1
                response = self._call_api(prompt, **kwargs)

                if not isinstance(response, str):
                    response = str(response)

                logger.info(
                    f"[{self.model_name}] Query successful on attempt {attempt}"
                )
                return response.strip()

            except Exception as e:
                self._error_count += 1
                last_exception = e
                logger.warning(
                    f"[{self.model_name}] Attempt {attempt} failed: {type(e).__name__}: {e}"
                )

                if attempt < self.max_retries:
                    logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff

        raise RuntimeError(
            f"[{self.model_name}] All {self.max_retries} attempts failed. "
            f"Last error: {last_exception}"
        )

    def get_stats(self) -> dict:
        """
        Return usage statistics for this model instance.

        Returns:
            Dict with call_count, error_count, and success_rate.
        """
        success_rate = (
            (self._call_count - self._error_count) / self._call_count * 100
            if self._call_count > 0
            else 0.0
        )
        return {
            "model_name": self.model_name,
            "total_calls": self._call_count,
            "errors": self._error_count,
            "success_rate": round(success_rate, 2),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model_name}')"
