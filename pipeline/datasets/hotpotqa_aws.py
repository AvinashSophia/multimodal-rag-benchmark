"""HotpotQA dataset loader — reads corpus and QA pairs from S3.

AWS variant of pipeline/datasets/hotpotqa.py. Dataset structure, UnifiedSample
fields, and corpus format are identical. The only difference is that corpus
and QA pairs are fetched from S3 instead of downloading from HuggingFace.

S3 paths (under bucket spatial-ai-staging-processing-632872792182):
    benchmarking/datasets/hotpotqa/corpus.jsonl   ← deduplicated text passages
    benchmarking/datasets/hotpotqa/qa_pairs.json  ← pre-generated QA pairs

To populate S3:
    uv run python -m scripts.upload_hotpotqa_to_s3 --config configs/aws.yaml
"""

import json
from typing import Any, Dict, List, Tuple

from pipeline.datasets.base import BaseDataset, register_dataset
from pipeline.utils import UnifiedSample
from pipeline.utils.s3 import S3Client

DATASET_NAME = "hotpotqa"


@register_dataset("hotpotqa_aws")
class HotpotQAAWSDataset(BaseDataset):
    """HotpotQA multi-hop QA dataset — S3-backed.

    Text corpus and QA pairs are read from S3 (uploaded once via
    scripts/upload_hotpotqa_to_s3.py). No HuggingFace download at runtime.
    HotpotQA is text-only — get_images() always returns empty lists.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.s3 = S3Client(config)
        self._corpus_texts: List[str] = []
        self._corpus_ids: List[str] = []

    def load(self) -> None:
        """Load corpus and QA pairs from S3."""
        # Load corpus
        corpus_key = self.s3.dataset_key(DATASET_NAME, "corpus.jsonl")
        try:
            raw = self.s3.download_bytes(corpus_key).decode("utf-8")
        except Exception as exc:
            raise FileNotFoundError(
                f"Corpus not found at s3://{self.s3.bucket}/{corpus_key}\n"
                f"Underlying error: {exc}\n"
                f"Run: uv run python -m scripts.upload_hotpotqa_to_s3 --config configs/aws.yaml"
            ) from exc

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            self._corpus_texts.append(entry["text"])
            self._corpus_ids.append(entry["id"])

        print(f"  [HotpotQAAWS] Corpus loaded: {len(self._corpus_texts)} passages from S3")

        # Load QA pairs
        qa_key = self.s3.dataset_key(DATASET_NAME, "qa_pairs.json")
        try:
            qa_pairs: List[Dict[str, Any]] = self.s3.download_json(qa_key)
        except Exception as exc:
            raise FileNotFoundError(
                f"QA pairs not found at s3://{self.s3.bucket}/{qa_key}\n"
                f"Underlying error: {exc}\n"
                f"Run: uv run python -m scripts.upload_hotpotqa_to_s3 --config configs/aws.yaml"
            ) from exc

        if self.max_samples:
            qa_pairs = qa_pairs[: self.max_samples]

        for qa in qa_pairs:
            sample = UnifiedSample(
                id=qa["id"],
                question=qa["question"],
                text_corpus=[],
                images=[],
                image_ids=[],
                ground_truth=qa["answer"],
                metadata={
                    "type": qa.get("type", ""),
                    "level": qa.get("level", ""),
                    "supporting_facts": qa.get("supporting_facts", {}),
                    "all_answers": [qa["answer"]],
                },
            )
            self.samples.append(sample)

        print(f"  [HotpotQAAWS] Loaded {len(self.samples)} QA samples")

    def load_qa_only(self) -> None:
        """Load only QA pairs from S3 — skips corpus.jsonl entirely.

        Used by the batch runner when the index already exists. The text
        retriever queries the existing Elasticsearch/Qdrant index directly;
        no corpus needs to be in memory.
        """
        qa_key = self.s3.dataset_key(DATASET_NAME, "qa_pairs.json")
        try:
            qa_pairs: List[Dict[str, Any]] = self.s3.download_json(qa_key)
        except Exception as exc:
            raise FileNotFoundError(
                f"QA pairs not found at s3://{self.s3.bucket}/{qa_key}\n"
                f"Underlying error: {exc}\n"
                f"Run: uv run python -m scripts.upload_hotpotqa_to_s3 --config configs/aws.yaml"
            ) from exc

        if self.max_samples:
            qa_pairs = qa_pairs[: self.max_samples]

        for qa in qa_pairs:
            sample = UnifiedSample(
                id=qa["id"],
                question=qa["question"],
                text_corpus=[],
                images=[],
                image_ids=[],
                ground_truth=qa["answer"],
                metadata={
                    "type": qa.get("type", ""),
                    "level": qa.get("level", ""),
                    "supporting_facts": qa.get("supporting_facts", {}),
                    "all_answers": [qa["answer"]],
                },
            )
            self.samples.append(sample)

        print(f"  [HotpotQAAWS] Loaded {len(self.samples)} QA samples "
              f"— corpus skipped (already indexed)")

    def get_corpus(self) -> Tuple[List[str], List[str]]:
        """Return all text passages and their IDs for text retrieval indexing."""
        return self._corpus_texts, self._corpus_ids

    def get_images(self) -> Tuple[List[Any], List[str]]:
        """HotpotQA is text-only — no images."""
        return [], []
