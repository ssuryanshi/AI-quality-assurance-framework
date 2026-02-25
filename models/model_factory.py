"""
Model Factory
=============
Factory function to create the appropriate model adapter
based on configuration settings.

Usage:
    config = load_config("config.yaml")
    model = create_model(config)
    response = model.query("Hello!")
"""

import logging
from typing import Any, Dict

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


def create_model(config: Dict[str, Any]) -> BaseModel:
    """
    Create a model adapter based on configuration.

    Reads the 'model.provider' field from config and instantiates
    the appropriate model class with its settings.

    Args:
        config: Configuration dictionary (from config.yaml).
            Expected structure:
                model:
                    provider: "openai" | "huggingface"
                    openai:
                        model_name: str
                        temperature: float
                        max_tokens: int
                        ...
                    huggingface:
                        model_name: str
                        max_tokens: int
                        ...

    Returns:
        An instance of BaseModel (OpenAIModel or HuggingFaceModel).

    Raises:
        ValueError: If the provider is not recognized.
    """
    model_config = config.get("model", {})
    provider = model_config.get("provider", "openai").lower()

    if provider == "openai":
        from models.openai_model import OpenAIModel

        openai_config = model_config.get("openai", {})
        model = OpenAIModel(
            model_name=openai_config.get("model_name", "gpt-3.5-turbo"),
            temperature=openai_config.get("temperature", 0.0),
            max_tokens=openai_config.get("max_tokens", 512),
            max_retries=openai_config.get("max_retries", 3),
            timeout=openai_config.get("timeout", 30),
        )
        logger.info(f"Created OpenAI model: {model.model_name}")
        return model

    elif provider == "huggingface":
        from models.huggingface_model import HuggingFaceModel

        hf_config = model_config.get("huggingface", {})
        model = HuggingFaceModel(
            model_name=hf_config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.2"),
            max_tokens=hf_config.get("max_tokens", 512),
            temperature=hf_config.get("temperature", 0.0),
            max_retries=hf_config.get("max_retries", 3),
            timeout=hf_config.get("timeout", 60),
        )
        logger.info(f"Created HuggingFace model: {model.model_name}")
        return model

    else:
        raise ValueError(
            f"Unknown model provider: '{provider}'. "
            f"Supported providers: 'openai', 'huggingface'"
        )
