"""GQA dataset loader — reads QA pairs from S3, images fetched on demand.

AWS variant of pipeline/datasets/gqa.py. Dataset structure, UnifiedSample
fields, and image deduplication logic are identical. Images are fetched from
S3 on demand instead of loading all PIL images from HuggingFace into memory.

S3 paths (under bucket spatial-ai-staging-processing-632872792182):
    benchmarking/images/gqa/{image_id}.jpg     ← scene images (deduplicated)
    benchmarking/datasets/gqa/qa_pairs.json    ← pre-generated QA pairs

To populate S3:
    uv run python -m scripts.upload_gqa_to_s3 --config configs/aws.yaml
"""

from io import BytesIO
from typing import Any, Dict, List, Tuple

from PIL import Image

from pipeline.datasets.base import BaseDataset, register_dataset
from pipeline.utils import UnifiedSample
from pipeline.utils.s3 import S3Client

DATASET_NAME = "gqa"


@register_dataset("gqa_aws")
class GQAAWSDataset(BaseDataset):
    """GQA visual QA dataset — S3-backed.

    QA pairs are read from S3. Images are fetched on demand from S3 by
    image_key stored in each QA pair. Images are deduplicated by image_id
    since multiple questions share the same scene image — same as local variant.
    GQA is image-only — get_corpus() always returns empty lists.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.s3 = S3Client(config)
        self._images: List[Image.Image] = []
        self._image_ids: List[str] = []

    def _fetch_image(self, s3_key: str) -> Image.Image:
        """Download an image from S3 and return as PIL Image."""
        data = self.s3.download_bytes(s3_key)
        return Image.open(BytesIO(data)).convert("RGB")

    def load(self) -> None:
        """Load QA pairs from S3 and fetch unique images for corpus indexing.

        Builds the full image corpus from ALL qa_pairs first (for complete indexing),
        then applies max_samples only to self.samples for evaluation. This ensures
        is_indexed() checks always see the full corpus count regardless of max_samples.
        """
        qa_key = self.s3.dataset_key(DATASET_NAME, "qa_pairs.json")
        try:
            all_qa_pairs: List[Dict[str, Any]] = self.s3.download_json(qa_key)
        except Exception as exc:
            raise FileNotFoundError(
                f"QA pairs not found at s3://{self.s3.bucket}/{qa_key}\n"
                f"Underlying error: {exc}\n"
                f"Run: uv run python -m scripts.upload_gqa_to_s3 --config configs/aws.yaml"
            ) from exc

        # Build full image corpus from ALL qa_pairs — never truncated by max_samples.
        # Multiple questions share the same scene image, so deduplicate by image_id.
        seen_images: set = set()
        for qa in all_qa_pairs:
            image_id: str = qa["image_id"]
            img_key: str = qa["image_key"]

            if image_id not in seen_images:
                try:
                    img = self._fetch_image(img_key)
                except Exception:
                    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
                self._images.append(img)
                self._image_ids.append(image_id)
                seen_images.add(image_id)

        # Apply max_samples only to evaluation QA pairs.
        eval_qa_pairs = all_qa_pairs[: self.max_samples] if self.max_samples else all_qa_pairs

        for qa in eval_qa_pairs:
            image_id = qa["image_id"]

            sample = UnifiedSample(
                id=qa["id"],
                question=qa["question"],
                text_corpus=[],
                images=[],
                image_ids=[image_id],
                ground_truth=qa["answer"],
                metadata={
                    "question_id": qa.get("question_id", qa["id"]),
                    "image_id": image_id,
                    "all_answers": [qa["answer"]],
                    "types": qa.get("types", {}),
                    "relevant_image_ids": [image_id],
                },
            )
            self.samples.append(sample)

        print(f"  [GQAAWS] Loaded {len(self.samples)} QA samples, "
              f"{len(self._images)} unique images from S3")

    def load_qa_only(self) -> None:
        """Load only QA pairs from S3 — skips image corpus loading.

        Used by the batch runner when the image index already exists. Images are
        fetched on demand from S3 by the retriever at query time; no images need
        to be loaded into memory.
        """
        qa_key = self.s3.dataset_key(DATASET_NAME, "qa_pairs.json")
        try:
            qa_pairs: List[Dict[str, Any]] = self.s3.download_json(qa_key)
        except Exception as exc:
            raise FileNotFoundError(
                f"QA pairs not found at s3://{self.s3.bucket}/{qa_key}\n"
                f"Underlying error: {exc}\n"
                f"Run: uv run python -m scripts.upload_gqa_to_s3 --config configs/aws.yaml"
            ) from exc

        if self.max_samples:
            qa_pairs = qa_pairs[: self.max_samples]

        for qa in qa_pairs:
            image_id: str = qa["image_id"]

            sample = UnifiedSample(
                id=qa["id"],
                question=qa["question"],
                text_corpus=[],
                images=[],
                image_ids=[image_id],
                ground_truth=qa["answer"],
                metadata={
                    "question_id": qa.get("question_id", qa["id"]),
                    "image_id": image_id,
                    "all_answers": [qa["answer"]],
                    "types": qa.get("types", {}),
                    "relevant_image_ids": [image_id],
                },
            )
            self.samples.append(sample)

        print(f"  [GQAAWS] Loaded {len(self.samples)} QA samples "
              f"— images skipped (already indexed)")

    def get_corpus(self) -> Tuple[List[str], List[str]]:
        """GQA is image-only — no text corpus."""
        return [], []

    def get_images(self) -> Tuple[List[Image.Image], List[str]]:
        """Return deduplicated scene images and their IDs for image retrieval indexing."""
        return self._images, self._image_ids
