"""
HuggingFace Model Adapter
=========================
Wraps the HuggingFace Inference API for use in evaluations.

Requires:
    - HUGGINGFACE_API_TOKEN set in environment or .env file
    - requests package installed

Usage:
    model = HuggingFaceModel(model_name="mistralai/Mistral-7B-Instruct-v0.2")
    response = model.query("What is machine learning?")
"""

import os
import logging
from typing import Optional

import requests

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class HuggingFaceModel(BaseModel):
    """
    HuggingFace Inference API adapter.

    Uses the HuggingFace serverless Inference API to query text-generation models.

    Supports:
        - Any text-generation model hosted on HuggingFace
        - Configurable generation parameters
        - Automatic retry with backoff (inherited from BaseModel)
    """

    API_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model_name}"

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_tokens: int = 512,
        temperature: float = 0.0,
        api_token: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        """
        Initialize the HuggingFace model adapter.

        Args:
            model_name: HuggingFace model identifier (e.g., 'mistralai/Mistral-7B-Instruct-v0.2').
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0 = deterministic).
            api_token: HuggingFace API token. Falls back to HUGGINGFACE_API_TOKEN env var.
            max_retries: Maximum retry attempts.
            timeout: Request timeout in seconds.
        """
        super().__init__(max_retries=max_retries, timeout=timeout)

        self._model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Resolve API token
        self._api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        if not self._api_token:
            raise ValueError(
                "HuggingFace API token not found. Set HUGGINGFACE_API_TOKEN "
                "environment variable or pass api_token parameter."
            )

        self._api_url = self.API_URL_TEMPLATE.format(model_name=self._model_name)
        self._headers = {"Authorization": f"Bearer {self._api_token}"}

        logger.info(f"HuggingFace model initialized: {self._model_name}")

    @property
    def model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        return self._model_name

    def _call_api(self, prompt: str, **kwargs) -> str:
        """
        Make a call to the HuggingFace Inference API.

        Args:
            prompt: The user's input prompt.
            **kwargs: Optional overrides for generation parameters.

        Returns:
            The model's response text.
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature) or 0.01,
                "return_full_text": False,
            },
        }

        response = requests.post(
            self._api_url,
            headers=self._headers,
            json=payload,
            timeout=self.timeout,
        )

        # Handle API errors
        if response.status_code != 200:
            error_msg = response.json().get("error", response.text)
            raise RuntimeError(
                f"HuggingFace API error ({response.status_code}): {error_msg}"
            )

        result = response.json()

        # Parse response format
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            generated_text = result.get("generated_text", "")
        else:
            raise RuntimeError(f"Unexpected response format: {type(result)}")

        return generated_text
