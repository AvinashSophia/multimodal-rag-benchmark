"""Base dataset class and dataset loader registry."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Iterator
from pipeline.utils import UnifiedSample


class BaseDataset(ABC):
    """Abstract base class for all dataset loaders.

    Every dataset must convert its native format into UnifiedSample objects.
    This ensures the rest of the pipeline (retrieval, models, evaluation)
    works with any dataset without modification.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = config["dataset"].get("data_dir", "data/")
        self.split = config["dataset"].get("split", "validation")
        self.max_samples = config["dataset"].get("max_samples", None)
        self.samples: List[UnifiedSample] = []

    @abstractmethod
    def load(self) -> None:
        """Load and convert dataset to unified format.

        Must populate self.samples with UnifiedSample objects.
        """
        raise NotImplementedError

    @abstractmethod
    def get_corpus(self) -> List[str]:
        """Return the full text corpus for indexing by retrievers."""
        raise NotImplementedError

    @abstractmethod
    def get_images(self) -> List[Any]:
        """Return all images for indexing by image retrievers."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[UnifiedSample]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> UnifiedSample:
        return self.samples[idx]


# Registry for dataset loaders
DATASET_REGISTRY: Dict[str, type] = {}


def register_dataset(name: str):
    """Decorator to register a dataset loader."""
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset(config: Dict[str, Any]) -> BaseDataset:
    """Factory function to get the right dataset loader from config."""
    name = config["dataset"]["name"]
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' not found. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name](config)
