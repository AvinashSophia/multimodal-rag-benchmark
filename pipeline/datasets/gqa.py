"""GQA dataset loader for visual question answering evaluation."""

from typing import Dict, List, Any, Tuple
from datasets import load_dataset as hf_load_dataset
from pipeline.datasets.base import BaseDataset, register_dataset
from pipeline.utils import UnifiedSample


@register_dataset("gqa")
class GQADataset(BaseDataset):
    """GQA: Visual Question Answering with scene graph structured questions.

    Tests the system's ability to understand natural images and answer
    questions requiring spatial reasoning and object recognition.

    Loads two subsets from lmms-lab/GQA and joins them on imageId:
    - val_balanced_images: imageId → PIL Image
    - val_balanced_instructions: imageId, question, answer, types, ...

    Example:
        Image: A kitchen scene
        Q: "What color is the object to the left of the table?"
    """

    def load(self) -> None:
        """Load GQA from HuggingFace, join image and instruction subsets."""
        # Load image subset and build imageId → PIL Image lookup
        images_ds = hf_load_dataset("lmms-lab/GQA", "val_balanced_images", split="val")
        image_lookup: Dict[str, Any] = {
            str(item["id"]): item["image"] for item in images_ds
        }

        # Load instruction subset (questions + answers)
        instructions_ds = hf_load_dataset(
            "lmms-lab/GQA", "val_balanced_instructions", split="val"
        )

        if self.max_samples:
            instructions_ds = instructions_ds.select(
                range(min(self.max_samples, len(instructions_ds)))
            )

        for idx, item in enumerate(instructions_ds):
            image_id = str(item["imageId"])
            image = image_lookup.get(image_id)
            if image is None:
                # Skip questions whose image is not in the images subset
                continue

            sample = UnifiedSample(
                id=f"gqa_{idx}",
                question=item["question"],
                text_corpus=[],  # GQA is purely visual
                images=[image],
                image_ids=[image_id],
                ground_truth=item["answer"],
                metadata={
                    "question_id": str(item.get("id", idx)),
                    "image_id": image_id,
                    "all_answers": [item["answer"]],
                    "types": item.get("types", {}),
                },
            )
            self.samples.append(sample)

    def get_corpus(self) -> Tuple[List[str], List[str]]:
        """GQA is image-based, no text corpus."""
        return [], []

    def get_images(self) -> Tuple[List[Any], List[str]]:
        """Return unique images and their IDs for indexing.

        Deduplicates by imageId since multiple questions share the same image.
        """
        images, image_ids = [], []
        seen: set = set()
        for sample in self.samples:
            for img, img_id in zip(sample.images, sample.image_ids):
                if img_id not in seen:
                    images.append(img)
                    image_ids.append(img_id)
                    seen.add(img_id)
        return images, image_ids
