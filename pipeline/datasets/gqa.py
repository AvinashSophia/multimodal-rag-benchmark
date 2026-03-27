"""GQA dataset loader for visual question answering evaluation."""

from typing import Dict, List, Any
from datasets import load_dataset as hf_load_dataset
from pipeline.datasets.base import BaseDataset, register_dataset
from pipeline.utils import UnifiedSample


@register_dataset("gqa")
class GQADataset(BaseDataset):
    """GQA: Visual Question Answering with scene graph structured questions.

    Tests the system's ability to understand natural images and answer
    questions requiring spatial reasoning and object recognition.

    Example:
        Image: A kitchen scene
        Q: "What color is the object to the left of the table?"
    """

    def load(self) -> None:
        """Load GQA from HuggingFace and convert to unified format."""
        dataset = hf_load_dataset("lmms-lab/GQA", split=self.split)

        if self.max_samples:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))

        for idx, item in enumerate(dataset):
            sample = UnifiedSample(
                id=f"gqa_{idx}",
                question=item["question"],
                text_corpus=[],  # GQA is purely visual
                images=[item["image"]],  # PIL Image
                ground_truth=item["answer"],
                metadata={
                    "question_id": item.get("question_id", ""),
                    "image_id": item.get("imageId", ""),
                    "types": item.get("types", {}),
                },
            )
            self.samples.append(sample)

    def get_corpus(self) -> List[str]:
        """GQA is image-based, no text corpus."""
        return []

    def get_images(self) -> List[Any]:
        """Return all images for indexing."""
        images = []
        for sample in self.samples:
            images.extend(sample.images)
        return images
