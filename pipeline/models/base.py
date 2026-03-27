"""Base model class and model registry."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pipeline.utils import ModelResult


class BaseModel(ABC):
    """Abstract base class for all QA models.

    Every model must implement run_model() with the exact interface
    Ivana specified: question + text_context + image_context -> answer + sources.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def run_model(
        self,
        question: str,
        text_context: List[str],
        image_context: List[Any],
    ) -> ModelResult:
        """Run the model on a question with retrieved context.

        Args:
            question: The user's question.
            text_context: Retrieved text chunks.
            image_context: Retrieved images (PIL Images or paths).

        Returns:
            ModelResult with answer and sources.
        """
        raise NotImplementedError


# Registry for models
MODEL_REGISTRY: Dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(config: Dict[str, Any]) -> BaseModel:
    """Factory function to get the right model from config."""
    name = config["model"]["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](config)
