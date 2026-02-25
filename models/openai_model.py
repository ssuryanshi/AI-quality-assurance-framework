"""
OpenAI Model Adapter
====================
Wraps the OpenAI Chat Completions API for use in evaluations.

Requires:
    - OPENAI_API_KEY set in environment or .env file
    - openai package installed

Usage:
    model = OpenAIModel(model_name="gpt-3.5-turbo", temperature=0.0)
    response = model.query("What is the capital of France?")
"""

import os
import logging
from typing import Optional

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class OpenAIModel(BaseModel):
    """
    OpenAI API adapter using the Chat Completions endpoint.

    Supports:
        - Configurable model selection (gpt-3.5-turbo, gpt-4, etc.)
        - Temperature, max_tokens, and other generation parameters
        - System prompt customization
        - Automatic retry with rate-limit backoff (inherited from BaseModel)
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """
        Initialize the OpenAI model adapter.

        Args:
            model_name: OpenAI model identifier (e.g., 'gpt-3.5-turbo', 'gpt-4').
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
            system_prompt: Optional system message to prepend to all queries.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            max_retries: Maximum retry attempts.
            timeout: Request timeout in seconds.
        """
        super().__init__(max_retries=max_retries, timeout=timeout)

        self._model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant. Provide accurate, concise answers."
        )

        # Resolve API key
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize the client
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self._api_key,
                timeout=self.timeout,
            )
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        logger.info(f"OpenAI model initialized: {self._model_name}")

    @property
    def model_name(self) -> str:
        """Return the OpenAI model identifier."""
        return self._model_name

    def _call_api(self, prompt: str, **kwargs) -> str:
        """
        Make a call to the OpenAI Chat Completions API.

        Args:
            prompt: The user's input prompt.
            **kwargs: Optional overrides for temperature, max_tokens, etc.

        Returns:
            The model's response text.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        # Extract the response text
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content
