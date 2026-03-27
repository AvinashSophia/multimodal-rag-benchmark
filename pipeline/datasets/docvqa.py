"""DocVQA dataset loader for document + multimodal QA evaluation."""

from typing import Dict, List, Any
from datasets import load_dataset as hf_load_dataset
from pipeline.datasets.base import BaseDataset, register_dataset
from pipeline.utils import UnifiedSample


@register_dataset("docvqa")
class DocVQADataset(BaseDataset):
    """DocVQA: Visual question answering on document images.

    Tests the system's ability to understand document images (invoices,
    forms, reports, tables) and answer questions about their content.

    Example:
        Image: A scanned invoice
        Q: "What is the total amount due?"
    """

    def load(self) -> None:
        """Load DocVQA from HuggingFace and convert to unified format."""
        dataset = hf_load_dataset("lmms-lab/DocVQA",'DocVQA',split=self.split)

        if self.max_samples:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))

        for idx, item in enumerate(dataset):
            sample = UnifiedSample(
                id=f"docvqa_{idx}",
                question=item["question"],
                text_corpus=[],  # DocVQA is primarily image-based
                images=[item["image"]],  # PIL Image from HuggingFace
                ground_truth=item["answers"][0] if item["answers"] else "",
                metadata={
                    "all_answers": item.get("answers", []),
                    "question_id": item.get("questionId", ""),
                },
            )
            self.samples.append(sample)

    def get_corpus(self) -> List[str]:
        """DocVQA is image-based, no text corpus."""
        return []

    def get_images(self) -> List[Any]:
        """Return all document images for indexing."""
        images = []
        for sample in self.samples:
            images.extend(sample.images)
        return images
