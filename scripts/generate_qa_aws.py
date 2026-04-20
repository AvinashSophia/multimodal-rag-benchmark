"""Generate QA pairs from S3-hosted parsed page JSONs — uploads result to S3.

AWS variant of scripts/generate_qa.py. QA generation logic (GPT-4o prompts,
text/visual pair structure, schema) is identical. Input is read from S3 and
output is uploaded to S3 instead of local disk.

`query_image_path` in each QA pair stores an S3 key (not a local path).
altumint_aws.py (the AWS dataset loader) reads these keys and fetches images
from S3 at benchmark time.

Usage
-----
    uv run python scripts/generate_qa_aws.py --dataset altumint
    uv run python scripts/generate_qa_aws.py --dataset altumint --text-per-page 3
"""

import argparse
import base64
import json
import os
import re
from io import BytesIO
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from pipeline.utils import load_config
from pipeline.utils.s3 import S3Client

load_dotenv()

DEFAULT_TEXT_PER_PAGE = 2
MIN_TEXT_LEN = 30
MODEL_ID = "gpt-4o"


# ---------------------------------------------------------------------------
# Helpers  (identical to generate_qa.py)
# ---------------------------------------------------------------------------

def _image_bytes_to_b64(data: bytes) -> str:
    """Convert raw PNG bytes to a base64 string for the OpenAI API."""
    img = Image.open(BytesIO(data)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_json_response(raw: str) -> Any:
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return None


# ---------------------------------------------------------------------------
# QA generation functions  (identical logic to generate_qa.py)
# ---------------------------------------------------------------------------

def generate_text_qa(page: dict, client: OpenAI, n: int) -> list[dict]:
    full_text = page.get("full_text", "").strip()
    tables = page.get("tables", [])
    if len(full_text) < MIN_TEXT_LEN and not tables:
        return []

    content_parts = []
    if full_text:
        content_parts.append(f"Text content:\n{full_text}")
    for i, tbl in enumerate(tables):
        md = tbl.get("markdown", "")
        if md:
            content_parts.append(f"Table {i + 1}:\n{md}")
    content_block = "\n\n".join(content_parts)

    table_instruction = (
        " Include at least one question about a specific value from the table "
        "(e.g. a measurement, expected reading, or component name from a row)."
        if tables else ""
    )

    prompt = (
        f"You are a field technician or engineer working with the FLVM (Flashing Light Video Monitor) system.\n"
        f"Generate {n} practical QA pairs that you would realistically need answered while "
        f"assembling, installing, wiring, testing, or configuring this equipment.\n\n"
        f"Page {page['page_no']} of \"{page['source_file']}\"\n\n"
        f"{content_block}\n\n"
        f"Rules:\n"
        f"- Questions must reflect real on-the-job needs: specific measurements, component specs, "
        f"wire lengths, continuity values, assembly steps, configuration settings\n"
        f"- Vary question_type across: factual (component names, specs), numerical (measurements, quantities), "
        f"procedural (steps, sequences)\n"
        f"- Answers must be concise (1–20 words), using exact values from the source\n"
        f"- NEVER ask about document titles, document codes, document IDs, scale, sheet numbers, "
        f"revision numbers, who drew it, or any other document metadata\n"
        f"- NEVER ask questions whose answer is just a document name or file reference\n"
        f"{table_instruction}\n\n"
        f"Good examples: 'What wire gauge is used for BATT+ SW?', "
        f"'What is the expected continuity between LOAD CB top screw and bottom screw?'\n"
        f"Bad examples: 'What is the document title?', 'What is the document code?'\n\n"
        f"Return a JSON array only — no explanation:\n"
        f'[{{"question": "...", "answer": "...", "question_type": "factual|numerical|procedural"}}]'
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a technical writer generating realistic QA pairs for a "
                        "multimodal RAG benchmark from engineering documents. Questions must "
                        "reflect what a field technician would genuinely need to know. "
                        "Return ONLY valid JSON. No explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.3,
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_json_response(raw)
        if isinstance(parsed, list):
            return parsed[:n]
        if isinstance(parsed, dict):
            return [parsed]
    except Exception as exc:
        print(f"    WARNING: text QA generation failed for {page['page_id']}: {exc}")

    return []


def generate_visual_qa(page: dict, fig: dict, client: OpenAI, s3: S3Client) -> dict | None:
    """Generate 1 visual QA pair from a figure — image fetched from S3."""
    fig_s3_key = fig.get("path", "")
    if not fig_s3_key:
        return None

    caption = fig.get("caption", "")
    full_text = page.get("full_text", "").strip()
    context_note = f"Surrounding text: {full_text[:300]}" if full_text else ""
    caption_note = f"Caption: {caption}" if caption else ""

    try:
        img_bytes = s3.download_bytes(fig_s3_key)
        b64 = _image_bytes_to_b64(img_bytes)
    except Exception as exc:
        print(f"    WARNING: could not load figure from S3 {fig_s3_key}: {exc}")
        return None

    prompt_text = (
        f"You are a field technician working with the FLVM system looking at this diagram.\n"
        f"Generate 1 QA pair about something you would genuinely need to identify or verify "
        f"from this image while doing your job (assembling, wiring, installing, or testing).\n\n"
        f"Source: page {page['page_no']} of \"{page['source_file']}\"\n"
        f"{caption_note}\n{context_note}\n\n"
        f"Rules:\n"
        f"- The question must require looking at the image to answer — not answerable from text alone\n"
        f"- Ask about visible components, their layout, connections, labels, or physical properties\n"
        f"- The answer must be specific and derivable from what is visible in the image\n"
        f"- NEVER ask about document titles, codes, scale, revision, or who drew it\n"
        f'Return JSON only: {{"question": "...", "answer": "...", "question_type": "visual"}}'
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate visual QA pairs for a multimodal RAG benchmark. "
                        "The question must require the image to answer. "
                        "Return ONLY valid JSON. No explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ],
            max_tokens=300,
            temperature=0.3,
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_json_response(raw)
        if isinstance(parsed, dict) and "question" in parsed and "answer" in parsed:
            return parsed
        if isinstance(parsed, list) and parsed:
            return parsed[0]
    except Exception as exc:
        print(f"    WARNING: visual QA generation failed for {fig_s3_key}: {exc}")

    return None


def process_page(page: dict, client: OpenAI, s3: S3Client, text_per_page: int) -> list[dict]:
    results: list[dict] = []

    text_pairs = generate_text_qa(page, client, n=text_per_page)
    for pair in text_pairs:
        results.append({
            "question":         pair.get("question", ""),
            "answer":           pair.get("answer", ""),
            "query_type":       "text",
            "question_type":    pair.get("question_type", "factual"),
            "query_image_path": None,
        })

    diagram_figures = [f for f in page.get("figures", []) if f.get("label") == "figure"]
    for fig in diagram_figures[:2]:
        vqa = generate_visual_qa(page, fig, client, s3)
        if vqa:
            results.append({
                "question":         vqa.get("question", ""),
                "answer":           vqa.get("answer", ""),
                "query_type":       "visual",
                "question_type":    vqa.get("question_type", "visual"),
                "query_image_path": fig["path"],  # S3 key — not a local path
            })

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from S3-hosted parsed page JSONs using GPT-4o."
    )
    parser.add_argument("--dataset",       required=True,                           help="Dataset name (e.g. altumint) — used as S3 prefix.")
    parser.add_argument("--text-per-page", type=int, default=DEFAULT_TEXT_PER_PAGE, help=f"Text QA pairs per page (default: {DEFAULT_TEXT_PER_PAGE}).")
    parser.add_argument("--config",        default=None,                            help="Path to config YAML (default: uses built-in S3 defaults).")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    config = load_config(args.config) if args.config else None
    s3 = S3Client(config)
    client = OpenAI(api_key=api_key)

    # List parsed page JSONs from S3
    prefix = s3.dataset_key(args.dataset, "parsed/")
    page_keys = sorted(k for k in s3.list_keys(prefix) if k.endswith(".json"))

    if not page_keys:
        print(f"No parsed JSON files found at s3://{s3.bucket}/{prefix}")
        return

    output_key = s3.dataset_key(args.dataset, "qa_pairs.json")

    print("=" * 60)
    print("QA Pair Generator (AWS)")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Input    : s3://{s3.bucket}/{prefix}")
    print(f"  Output   : s3://{s3.bucket}/{output_key}")
    print(f"  Pages    : {len(page_keys)}")
    print(f"  Text QA/page : {args.text_per_page}")
    print("=" * 60)

    all_pairs: list[dict] = []
    counter = 0

    for s3_key in page_keys:
        page = s3.download_json(s3_key)
        page_id = page["page_id"]
        print(f"  [page] {page_id}")

        partial_pairs = process_page(page, client, s3, text_per_page=args.text_per_page)

        for pair in partial_pairs:
            pair["id"]                = f"{args.dataset}_{counter:04d}"
            pair["source_doc"]        = page["doc_id"]
            pair["source_page"]       = page["page_no"]
            pair["source_page_id"]    = page_id
            pair["relevant_page_ids"] = [page_id]
            all_pairs.append({
                "id":                pair["id"],
                "question":          pair["question"],
                "answer":            pair["answer"],
                "query_type":        pair["query_type"],
                "question_type":     pair["question_type"],
                "source_doc":        pair["source_doc"],
                "source_page":       pair["source_page"],
                "source_page_id":    pair["source_page_id"],
                "relevant_page_ids": pair["relevant_page_ids"],
                "query_image_path":  pair["query_image_path"],  # S3 key or None
            })
            counter += 1

        text_n  = sum(1 for p in partial_pairs if p["query_type"] == "text")
        visual_n = sum(1 for p in partial_pairs if p["query_type"] == "visual")
        print(f"    → {len(partial_pairs)} pairs ({text_n} text, {visual_n} visual)")

    s3.upload_json(all_pairs, output_key)

    text_count   = sum(1 for p in all_pairs if p["query_type"] == "text")
    visual_count = sum(1 for p in all_pairs if p["query_type"] == "visual")
    print(f"\nDone. {len(all_pairs)} QA pairs uploaded to s3://{s3.bucket}/{output_key}")
    print(f"  text: {text_count}  visual: {visual_count}")


if __name__ == "__main__":
    main()
