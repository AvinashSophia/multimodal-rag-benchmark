"""Upload HotpotQA corpus and QA pairs to S3.

Downloads hotpot_qa (distractor) from HuggingFace, serializes the deduplicated
text corpus as JSONL and QA pairs as JSON, then uploads both to S3.

Run once per environment before any batch job that uses hotpotqa_aws.
Subsequent batch runs read directly from S3 — no HuggingFace download needed.

S3 output:
    benchmarking/datasets/hotpotqa/corpus.jsonl   ← one passage per line
    benchmarking/datasets/hotpotqa/qa_pairs.json  ← list of QA dicts

Usage:
    uv run python -m scripts.upload_hotpotqa_to_s3 --config configs/aws.yaml
    uv run python -m scripts.upload_hotpotqa_to_s3 --config configs/aws.yaml --split train
"""

import argparse
import json

from datasets import load_dataset as hf_load_dataset
from dotenv import load_dotenv

from pipeline.utils import load_config
from pipeline.utils.s3 import S3Client

load_dotenv()

DATASET_NAME = "hotpotqa"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload HotpotQA to S3")
    parser.add_argument("--config", default="configs/aws.yaml")
    parser.add_argument("--split", default="validation", help="HuggingFace split to upload")
    args = parser.parse_args()

    config = load_config(args.config)
    s3 = S3Client(config)

    print(f"Downloading HotpotQA ({args.split}) from HuggingFace...")
    dataset = hf_load_dataset("hotpot_qa", "distractor", split=args.split)
    print(f"  {len(dataset)} samples loaded")

    # Build deduplicated text corpus and QA pairs in one pass
    corpus: dict = {}   # title → passage text (deduped)
    qa_pairs = []

    for idx, item in enumerate(dataset):
        for title, sentences in zip(
            item["context"]["title"], item["context"]["sentences"]
        ):
            if title not in corpus:
                corpus[title] = f"[{title}] " + " ".join(sentences)

        qa_pairs.append({
            "id": f"hotpotqa_{idx}",
            "question": item["question"],
            "answer": item["answer"],
            "supporting_facts": {
                "titles": item["supporting_facts"]["title"],
                "sent_ids": item["supporting_facts"]["sent_id"],
            },
            "type": item.get("type", ""),
            "level": item.get("level", ""),
        })

        if (idx + 1) % 10_000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} samples "
                  f"({len(corpus)} unique passages so far)...")

    print(f"\n  {len(corpus)} unique passages, {len(qa_pairs)} QA pairs")

    # Upload corpus as JSONL (one JSON object per line)
    corpus_key = s3.dataset_key(DATASET_NAME, "corpus.jsonl")
    print(f"\nUploading corpus → s3://{s3.bucket}/{corpus_key}")
    corpus_bytes = "\n".join(
        json.dumps({"id": title, "text": text}, ensure_ascii=False)
        for title, text in corpus.items()
    ).encode("utf-8")
    s3.upload_bytes(corpus_bytes, corpus_key, content_type="application/x-ndjson")
    print(f"  {len(corpus)} passages uploaded")

    # Upload QA pairs
    qa_key = s3.dataset_key(DATASET_NAME, "qa_pairs.json")
    print(f"Uploading QA pairs → s3://{s3.bucket}/{qa_key}")
    s3.upload_json(qa_pairs, qa_key)
    print(f"  {len(qa_pairs)} QA pairs uploaded")

    print(f"\nDone. HotpotQA ({args.split}) is ready for benchmarking.")
    print(f"Set dataset.name = 'hotpotqa_aws' in configs/aws.yaml to use it.")


if __name__ == "__main__":
    main()
