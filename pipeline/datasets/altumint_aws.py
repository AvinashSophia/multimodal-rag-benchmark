"""Altumint FLVM dataset loader — reads corpus and QA pairs from S3.

AWS variant of pipeline/datasets/altumint.py. Dataset structure, UnifiedSample
fields, query types, and corpus format are identical. The only difference is that
parsed page JSONs, images, and qa_pairs.json are fetched from S3 instead of local disk.

S3 paths (under bucket spatial-ai-staging-processing-632872792182):
    benchmarking/datasets/{s3_prefix}/parsed/*.json   ← per-page structured JSONs
    benchmarking/datasets/{s3_prefix}/qa_pairs.json   ← pre-generated QA pairs
    benchmarking/images/{s3_prefix}/figures/*.png     ← page screenshots + figure crops

`path` fields inside each page JSON store S3 keys (set by parse_documents_aws.py).
`query_image_path` in QA pairs stores an S3 key (set by generate_qa_aws.py).

Config keys (under dataset):
    s3_prefix:   S3 key prefix for this dataset (default: "altumint")
    max_samples: Optional limit on QA pairs loaded (default: all)

To populate S3:
    uv run python scripts/parse_documents_aws.py --input data/altumint/ --dataset altumint
    uv run python scripts/generate_qa_aws.py --dataset altumint
"""

import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from pipeline.datasets.base import BaseDataset, register_dataset
from pipeline.utils import UnifiedSample
from pipeline.utils.s3 import S3Client


@register_dataset("altumint_aws")
class AltumintAWSDataset(BaseDataset):
    """Altumint FLVM technical document QA dataset — S3-backed.

    Each sample is a QA pair generated from a specific page of a proprietary
    PDF document. The retrieval corpus is built from Docling-parsed per-page
    JSONs stored in S3.

    Supports two query types:
      text   — pure text query; image retrieval not triggered
      visual — diagram crop (fetched from S3) used as query image
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        dataset_cfg = config["dataset"]
        self.s3_prefix = dataset_cfg.get("s3_prefix", "altumint")
        self.s3 = S3Client(config)

        # Populated lazily in get_corpus() / get_images()
        self._corpus_texts: Optional[List[str]] = None
        self._corpus_ids: Optional[List[str]] = None
        self._corpus_images: Optional[List[Image.Image]] = None
        self._corpus_image_ids: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_image(self, s3_key: str) -> Image.Image:
        """Download an image from S3 and return as PIL Image."""
        data = self.s3.download_bytes(s3_key)
        return Image.open(BytesIO(data)).convert("RGB")

    def _list_parsed_keys(self) -> List[str]:
        """List all per-page JSON keys from S3, sorted."""
        prefix = self.s3.dataset_key(self.s3_prefix, "parsed/")
        return sorted(k for k in self.s3.list_keys(prefix) if k.endswith(".json"))

    def _build_corpus(self) -> None:
        """Build text and image corpus caches from S3-hosted parsed page JSONs."""
        if self._corpus_texts is not None:
            return

        page_keys = self._list_parsed_keys()
        if not page_keys:
            raise FileNotFoundError(
                f"No parsed JSONs found at s3://{self.s3.bucket}/"
                f"{self.s3.dataset_key(self.s3_prefix, 'parsed/')}\n"
                f"Run: uv run python scripts/parse_documents_aws.py --input data/altumint/ --dataset {self.s3_prefix}"
            )

        texts: List[str] = []
        text_ids: List[str] = []
        images: List[Image.Image] = []
        image_ids: List[str] = []

        for s3_key in page_keys:
            page = self.s3.download_json(s3_key)
            page_id = page["page_id"]

            # Text corpus: full_text + table summaries and markdown
            corpus_text = page.get("full_text", "")
            for tbl in page.get("tables", []):
                if tbl.get("summary"):
                    corpus_text += "\n" + tbl["summary"]
                if tbl.get("markdown"):
                    corpus_text += "\n" + tbl["markdown"]
            texts.append(corpus_text)
            text_ids.append(page_id)

            # Image corpus: page screenshot from S3
            screenshot = next(
                (f for f in page.get("figures", []) if f.get("label") == "page_screenshot"),
                None,
            )
            if screenshot and screenshot.get("path"):
                try:
                    images.append(self._fetch_image(screenshot["path"]))
                except Exception:
                    images.append(Image.new("RGB", (100, 100), color=(255, 255, 255)))
            else:
                images.append(Image.new("RGB", (100, 100), color=(255, 255, 255)))
            image_ids.append(page_id)

        self._corpus_texts = texts
        self._corpus_ids = text_ids
        self._corpus_images = images
        self._corpus_image_ids = image_ids

        print(f"  [AltumintAWS] Corpus built: {len(texts)} pages from S3 ({self.s3_prefix})")

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load QA pairs from S3 and build UnifiedSamples."""
        qa_key = self.s3.dataset_key(self.s3_prefix, "qa_pairs.json")
        try:
            qa_pairs: List[Dict[str, Any]] = self.s3.download_json(qa_key)
        except Exception as exc:
            raise FileNotFoundError(
                f"QA pairs not found at s3://{self.s3.bucket}/{qa_key}\n"
                f"Underlying error: {exc}\n"
                f"Run: uv run python scripts/generate_qa_aws.py --dataset {self.s3_prefix}"
            ) from exc

        if self.max_samples:
            qa_pairs = qa_pairs[: self.max_samples]

        self._build_corpus()

        for qa in qa_pairs:
            query_type = qa.get("query_type", "text")
            query_image_path = qa.get("query_image_path")  # S3 key or None

            if query_type == "visual" and query_image_path:
                try:
                    query_images: List[Image.Image] = [self._fetch_image(query_image_path)]
                    query_image_ids: List[str] = [qa["source_page_id"]]
                except Exception:
                    # Missing figure in S3 — fall back to text query
                    query_images = []
                    query_image_ids = []
            else:
                query_images = []
                query_image_ids = []

            sample = UnifiedSample(
                id=qa["id"],
                question=qa["question"],
                text_corpus=[],
                images=query_images,
                image_ids=query_image_ids,
                ground_truth=qa["answer"],
                metadata={
                    "source_doc":         qa["source_doc"],
                    "source_page":        qa["source_page"],
                    "source_page_id":     qa["source_page_id"],
                    "query_type":         query_type,
                    "question_type":      qa.get("question_type", "factual"),
                    "all_answers":        [qa["answer"]],
                    "relevant_text_ids":  qa.get("relevant_page_ids", [qa["source_page_id"]]),
                    "relevant_image_ids": qa.get("relevant_page_ids", [qa["source_page_id"]]),
                },
            )
            self.samples.append(sample)

        text_count   = sum(1 for qa in qa_pairs if qa.get("query_type", "text") == "text")
        visual_count = sum(1 for qa in qa_pairs if qa.get("query_type") == "visual")
        print(f"  [AltumintAWS] Loaded {len(self.samples)} QA samples "
              f"({text_count} text, {visual_count} visual)")

    def load_qa_only(self) -> None:
        """Load only QA pairs from S3 — skips corpus and image loading.

        Used by the batch runner when the index already exists. The retriever
        fetches images on demand from S3; no corpus needs to be in memory.
        """
        qa_key = self.s3.dataset_key(self.s3_prefix, "qa_pairs.json")
        try:
            qa_pairs: List[Dict[str, Any]] = self.s3.download_json(qa_key)
        except Exception as exc:
            raise FileNotFoundError(
                f"QA pairs not found at s3://{self.s3.bucket}/{qa_key}\n"
                f"Underlying error: {exc}\n"
                f"Run: uv run python scripts/generate_qa_aws.py --dataset {self.s3_prefix}"
            ) from exc

        if self.max_samples:
            qa_pairs = qa_pairs[: self.max_samples]

        for qa in qa_pairs:
            query_type = qa.get("query_type", "text")
            query_image_path = qa.get("query_image_path")  # S3 key or None

            if query_type == "visual" and query_image_path:
                try:
                    query_images: List[Image.Image] = [self._fetch_image(query_image_path)]
                    query_image_ids: List[str] = [qa["source_page_id"]]
                except Exception:
                    query_images = []
                    query_image_ids = []
            else:
                query_images = []
                query_image_ids = []

            sample = UnifiedSample(
                id=qa["id"],
                question=qa["question"],
                text_corpus=[],
                images=query_images,
                image_ids=query_image_ids,
                ground_truth=qa["answer"],
                metadata={
                    "source_doc":         qa["source_doc"],
                    "source_page":        qa["source_page"],
                    "source_page_id":     qa["source_page_id"],
                    "query_type":         query_type,
                    "question_type":      qa.get("question_type", "factual"),
                    "all_answers":        [qa["answer"]],
                    "relevant_text_ids":  qa.get("relevant_page_ids", [qa["source_page_id"]]),
                    "relevant_image_ids": qa.get("relevant_page_ids", [qa["source_page_id"]]),
                },
            )
            self.samples.append(sample)

        text_count   = sum(1 for qa in qa_pairs if qa.get("query_type", "text") == "text")
        visual_count = sum(1 for qa in qa_pairs if qa.get("query_type") == "visual")
        print(f"  [AltumintAWS] Loaded {len(self.samples)} QA samples "
              f"({text_count} text, {visual_count} visual) — corpus skipped (already indexed)")

    def get_corpus(self) -> Tuple[List[str], List[str]]:
        """Return all page full_text values and their page IDs for text retrieval."""
        self._build_corpus()
        assert self._corpus_texts is not None and self._corpus_ids is not None
        return self._corpus_texts, self._corpus_ids

    def get_images(self) -> Tuple[List[Image.Image], List[str]]:
        """Return all page screenshot images and their page IDs for image indexing."""
        self._build_corpus()
        assert self._corpus_images is not None and self._corpus_image_ids is not None
        return self._corpus_images, self._corpus_image_ids
