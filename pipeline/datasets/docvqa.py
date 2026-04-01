"""DocVQA dataset loader for document + multimodal QA evaluation."""

from typing import Dict, List, Any, Tuple
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
            question_id = str(item.get("questionId", f"docvqa_{idx}"))
            sample = UnifiedSample(
                id=f"docvqa_{idx}",
                question=item["question"],
                text_corpus=[],  # DocVQA is primarily image-based
                images=[item["image"]],  # PIL Image from HuggingFace
                image_ids=[question_id],
                ground_truth=item["answers"][0] if item["answers"] else "",
                metadata={
                    "all_answers": item.get("answers", []),
                    "question_id": question_id,
                },
            )
            self.samples.append(sample)

    def get_corpus(self) -> Tuple[List[str], List[str]]:
        """DocVQA is image-based, no text corpus."""
        return [], []

    def get_images(self) -> Tuple[List[Any], List[str]]:
        """Return all document images and their IDs for indexing."""
        images, image_ids = [], []
        for sample in self.samples:
            images.extend(sample.images)
            image_ids.extend(sample.image_ids)
        return images, image_ids
