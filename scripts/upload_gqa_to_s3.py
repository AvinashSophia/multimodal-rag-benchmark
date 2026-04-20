"""Upload GQA images and QA pairs to S3.

Downloads lmms-lab/GQA (val_balanced_images + val_balanced_instructions) from
HuggingFace, uploads unique scene images as JPEG to S3 under
benchmarking/images/gqa/, and uploads QA pairs as JSON.

Run once per environment before any batch job that uses gqa_aws.
Subsequent batch runs fetch images on demand from S3 — no HuggingFace download.

S3 output:
    benchmarking/images/gqa/{image_id}.jpg     ← scene images (deduplicated)
    benchmarking/datasets/gqa/qa_pairs.json    ← list of QA dicts

Usage:
    uv run python -m scripts.upload_gqa_to_s3 --config configs/aws.yaml
"""

import argparse
from io import BytesIO

from datasets import load_dataset as hf_load_dataset
from dotenv import load_dotenv

from pipeline.utils import load_config
from pipeline.utils.s3 import S3Client

load_dotenv()

DATASET_NAME = "gqa"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload GQA to S3")
    parser.add_argument("--config", default="configs/aws.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    s3 = S3Client(config)

    # Load image subset first — build imageId → PIL Image lookup
    print("Downloading GQA val_balanced_images from HuggingFace...")
    images_ds = hf_load_dataset("lmms-lab/GQA", "val_balanced_images", split="val")
    print(f"  {len(images_ds)} images loaded")

    # Upload unique images to S3
    print("Uploading images to S3...")
    uploaded = 0
    skipped = 0
    image_lookup: dict = {}  # imageId → img_key (S3 key)

    for item in images_ds:
        image_id = str(item["id"])
        img_key = s3.image_key(DATASET_NAME, f"{image_id}.jpg")
        image_lookup[image_id] = img_key

        if s3.object_exists(img_key):
            skipped += 1
            continue

        buf = BytesIO()
        item["image"].convert("RGB").save(buf, format="JPEG", quality=90)
        s3.upload_bytes(buf.getvalue(), img_key, content_type="image/jpeg")
        uploaded += 1

        if (uploaded + skipped) % 500 == 0:
            print(f"  {uploaded + skipped}/{len(images_ds)} — "
                  f"uploaded: {uploaded}, skipped: {skipped}")

    print(f"  Images done — uploaded: {uploaded}, skipped: {skipped}")

    # Load instruction subset — questions + answers
    print("\nDownloading GQA val_balanced_instructions from HuggingFace...")
    instructions_ds = hf_load_dataset(
        "lmms-lab/GQA", "val_balanced_instructions", split="val"
    )
    print(f"  {len(instructions_ds)} instructions loaded")

    qa_pairs = []
    missing_images = 0

    for idx, item in enumerate(instructions_ds):
        image_id = str(item["imageId"])
        qa_img_key = image_lookup.get(image_id)
        if qa_img_key is None:
            missing_images += 1
            continue

        qa_pairs.append({
            "id": f"gqa_{idx}",
            "question_id": str(item.get("id", idx)),
            "image_id": image_id,
            "question": item["question"],
            "answer": item["answer"],
            "image_key": qa_img_key,
            "types": item.get("types", {}),
        })

        if (idx + 1) % 10_000 == 0:
            print(f"  Processed {idx + 1}/{len(instructions_ds)} instructions...")

    if missing_images:
        print(f"  WARNING: {missing_images} instructions skipped (image not found in images subset)")

    # Upload QA pairs
    qa_key = s3.dataset_key(DATASET_NAME, "qa_pairs.json")
    print(f"\nUploading QA pairs → s3://{s3.bucket}/{qa_key}")
    s3.upload_json(qa_pairs, qa_key)

    print(f"\nDone. {uploaded} images uploaded, {skipped} already existed, "
          f"{len(qa_pairs)} QA pairs uploaded.")
    print(f"Set dataset.name = 'gqa_aws' in configs/aws.yaml to use it.")


if __name__ == "__main__":
    main()
