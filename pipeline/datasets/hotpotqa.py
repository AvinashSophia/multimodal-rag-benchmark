"""HotpotQA dataset loader for multi-document reasoning evaluation."""

from typing import Dict, List, Any
from datasets import load_dataset as hf_load_dataset
from pipeline.datasets.base import BaseDataset, register_dataset
from pipeline.utils import UnifiedSample


@register_dataset("hotpotqa")
class HotpotQADataset(BaseDataset):
    """HotpotQA: Multi-hop question answering over multiple documents.

    Tests the system's ability to reason across multiple text documents
    to answer complex questions that require combining information.

    Example:
        Q: "Were Scott Derrickson and Ed Wood of the same nationality?"
        Requires finding nationality info from two separate documents.
    """

    def load(self) -> None:
        """Load HotpotQA from HuggingFace and convert to unified format."""
        dataset = hf_load_dataset("hotpot_qa", "distractor", split=self.split)

        if self.max_samples:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))

        for idx, item in enumerate(dataset):
            # HotpotQA provides context as list of [title, sentences] pairs
            text_corpus = []
            for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
                doc_text = f"[{title}] " + " ".join(sentences)
                text_corpus.append(doc_text)

            sample = UnifiedSample(
                id=f"hotpotqa_{idx}",
                question=item["question"],
                text_corpus=text_corpus,
                images=[],  # HotpotQA is text-only
                ground_truth=item["answer"],
                metadata={
                    "type": item.get("type", ""),
                    "level": item.get("level", ""),
                    "supporting_facts": {
                        "titles": item["supporting_facts"]["title"],
                        "sent_ids": item["supporting_facts"]["sent_id"],
                    },
                },
            )
            self.samples.append(sample)

    def get_corpus(self) -> tuple:
        """Return all text passages across all samples for indexing."""
        corpus = []
        corpus_ids = []
        seen = set()
        for sample in self.samples:
            for doc in sample.text_corpus:
                if doc not in seen:
                    title = doc.split("]")[0].replace("[", "").strip()
                    corpus.append(doc)
                    corpus_ids.append(title)
                    seen.add(doc)
        return corpus, corpus_ids

    def get_images(self) -> List[Any]:
        """HotpotQA has no images."""
        return []
