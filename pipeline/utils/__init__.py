"""Shared utilities for config loading, logging, and common helpers."""

import os
import json
import yaml
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
from PIL import Image


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_output_dirs(config: Dict[str, Any], has_images: bool = False) -> Path:
    """Create output directories for a benchmark run.

    The directory name reflects only the retrievers actually used:
      {dataset}_{text_method}_{image_method}_{model}_{timestamp}   (with images)
      {dataset}_{text_method}_{model}_{timestamp}                  (text-only)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = config["dataset"]["name"]
    text_method = config["retrieval"]["text"]["method"].replace("_", "")
    model = config["model"]["name"]

    if has_images:
        image_method = config["retrieval"]["image"]["method"].replace("_", "")
        run_name = f"{dataset}_{text_method}_{image_method}_{model}_{timestamp}"
    else:
        run_name = f"{dataset}_{text_method}_{model}_{timestamp}"

    run_dir = Path(config["output"]["log_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def save_json(data: Any, filepath: Union[str, Path]) -> None:
    """Save data as JSON."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


@dataclass
class UnifiedSample:
    """Unified data format for all datasets.

    Every dataset loader must convert its data into this format.
    """

    id: str
    question: str
    text_corpus: list = field(default_factory=list)  # List[str]
    images: list = field(default_factory=list)  # List[PIL.Image or image path]
    image_ids: list = field(default_factory=list)  # List[str], one ID per image
    ground_truth: str = ""
    metadata: dict = field(default_factory=dict)  # Dataset-specific extra info

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (images stored as paths, not objects)."""
        return {
            "id": self.id,
            "question": self.question,
            "text_corpus": self.text_corpus,
            "images": [str(img) if not isinstance(img, Image.Image) else "PIL_IMAGE" for img in self.images],
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }

@dataclass
class RetrievalResult:
    text_chunks: list = field(default_factory=list)
    text_scores: list = field(default_factory=list)
    text_ids: list = field(default_factory=list)
    images: list = field(default_factory=list)
    image_scores: list = field(default_factory=list)
    image_ids: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

@dataclass
class ModelResult:
    """Standardized output from any QA model."""

    answer: str = ""
    sources: list = field(default_factory=list)  # Which chunks/images were used
    raw_response: str = ""  # Full model response for debugging
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete result for a single benchmark sample."""

    sample_id: str = ""
    question: str = ""
    ground_truth: str = ""
    predicted_answer: str = ""
    raw_answer: str = ""
    retrieved_context: dict = field(default_factory=dict)
    attribution: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
