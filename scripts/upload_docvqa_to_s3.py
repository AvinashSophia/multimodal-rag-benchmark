"""Upload DocVQA images and QA pairs to S3.

Downloads lmms-lab/DocVQA from HuggingFace, uploads each document image
as PNG to S3 under benchmarking/images/docvqa/, and uploads QA pairs as JSON.

Run once per environment before any batch job that uses docvqa_aws.
Subsequent batch runs fetch images on demand from S3 — no HuggingFace download.

S3 output:
    benchmarking/images/docvqa/{question_id}.png  ← document page images
    benchmarking/datasets/docvqa/qa_pairs.json    ← list of QA dicts

Usage:
    uv run python -m scripts.upload_docvqa_to_s3 --config configs/aws.yaml
    uv run python -m scripts.upload_docvqa_to_s3 --config configs/aws.yaml --split train
"""

import argparse
from io import BytesIO

from datasets import load_dataset as hf_load_dataset
from dotenv import load_dotenv

from pipeline.utils import load_config
from pipeline.utils.s3 import S3Client

load_dotenv()

DATASET_NAME = "docvqa"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload DocVQA to S3")
    parser.add_argument("--config", default="configs/aws.yaml")
    parser.add_argument("--split", default="validation", help="HuggingFace split to upload")
    args = parser.parse_args()

    config = load_config(args.config)
    s3 = S3Client(config)

    print(f"Downloading DocVQA ({args.split}) from HuggingFace...")
    dataset = hf_load_dataset("lmms-lab/DocVQA", "DocVQA", split=args.split)
    print(f"  {len(dataset)} samples loaded")

    qa_pairs = []
    uploaded = 0
    skipped = 0

    for idx, item in enumerate(dataset):
        question_id = str(item.get("questionId", f"docvqa_{idx}"))
        img_key = s3.image_key(DATASET_NAME, f"{question_id}.png")

        # Skip if already uploaded
        if s3.object_exists(img_key):
            skipped += 1
        else:
            buf = BytesIO()
            item["image"].convert("RGB").save(buf, format="PNG")
            s3.upload_bytes(buf.getvalue(), img_key, content_type="image/png")
            uploaded += 1

        qa_pairs.append({
            "id": f"docvqa_{idx}",
            "question_id": question_id,
            "question": item["question"],
            "answers": item.get("answers", []),
            "image_key": img_key,
        })

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(dataset)} — uploaded: {uploaded}, skipped: {skipped}")

    # Upload QA pairs
    qa_key = s3.dataset_key(DATASET_NAME, "qa_pairs.json")
    print(f"\nUploading QA pairs → s3://{s3.bucket}/{qa_key}")
    s3.upload_json(qa_pairs, qa_key)

    print(f"\nDone. {uploaded} images uploaded, {skipped} already existed, "
          f"{len(qa_pairs)} QA pairs uploaded.")
    print(f"Set dataset.name = 'docvqa_aws' in configs/aws.yaml to use it.")


if __name__ == "__main__":
    main()
