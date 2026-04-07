"""Generate QA pairs from Docling-parsed per-page JSONs.

Reads any {company}/parsed/ folder produced by scripts/parse_documents.py,
calls GPT-4o to generate structured QA pairs, and writes qa_pairs.json to the
parent directory (i.e. data/{company}/qa_pairs.json).

Two query types are generated:
  text   — question answered from text/table content; no query image
  visual — question requires reading a diagram; figure crop used as query image

QA pair schema
--------------
{
  "id":                "altumint_0000",
  "question":          "What component is beneath the modem?",
  "answer":            "POE Injector",
  "query_type":        "text",        # "text" | "visual"
  "question_type":     "factual",     # factual | numerical | procedural | visual
  "source_doc":        "altumint_dc001_01_...",
  "source_page":       1,
  "source_page_id":    "altumint_dc001_01_..._p01",
  "relevant_page_ids": ["altumint_dc001_01_..._p01"],
  "query_image_path":  null           # str path for visual queries, null otherwise
}

Usage
-----
    uv run python scripts/generate_qa.py --input data/altumint/parsed/
    uv run python scripts/generate_qa.py --input data/altumint/parsed/ --text-per-page 3
"""

import argparse
import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TEXT_PER_PAGE = 2   # text QA pairs per page with substantive text
MIN_TEXT_LEN = 30           # minimum full_text length to attempt text QA
MODEL_ID = "gpt-4o"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_b64(path: str) -> str:
    """Load a PNG from disk and return a base64 string for the OpenAI API."""
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_json_response(raw: str) -> Any:
    """Extract and parse the first JSON object or array in a GPT response."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the first [...] or {...} block
        match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return None


# ---------------------------------------------------------------------------
# QA generation functions
# ---------------------------------------------------------------------------

def generate_text_qa(page: dict, client: OpenAI, n: int) -> list[dict]:
    """Generate up to n text QA pairs from a page's text and table content.

    Returns a list of raw dicts with keys: question, answer, question_type.
    Returns [] if the page has insufficient content or GPT fails.
    """
    full_text = page.get("full_text", "").strip()
    tables = page.get("tables", [])

    if len(full_text) < MIN_TEXT_LEN and not tables:
        return []

    # Build content block: text + any table markdown
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
        f"'What is the expected continuity between LOAD CB top screw and bottom screw?', "
        f"'What torque should be applied when mounting the bracket?'\n"
        f"Bad examples: 'What is the document title?', 'What is the document code?', "
        f"'Who drew this document?'\n\n"
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


def generate_visual_qa(page: dict, fig: dict, client: OpenAI) -> dict | None:
    """Generate 1 visual QA pair from a figure crop image.

    The question must require looking at the diagram to answer.
    Returns a raw dict with keys: question, answer, question_type.
    Returns None if the figure cannot be loaded or GPT fails.
    """
    fig_path = fig.get("path", "")
    if not fig_path or not Path(fig_path).exists():
        return None

    caption = fig.get("caption", "")
    full_text = page.get("full_text", "").strip()
    context_note = f"Surrounding text: {full_text[:300]}" if full_text else ""
    caption_note = f"Caption: {caption}" if caption else ""

    try:
        b64 = _image_to_b64(fig_path)
    except Exception as exc:
        print(f"    WARNING: could not load figure {fig_path}: {exc}")
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
        f"Good examples: 'Which component is mounted in the top-left of the enclosure?', "
        f"'What color wire connects the POE injector to the distribution block?', "
        f"'How many batteries are shown in the assembly?'\n\n"
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
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
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
        print(f"    WARNING: visual QA generation failed for {fig_path}: {exc}")

    return None


def process_page(page: dict, client: OpenAI, text_per_page: int) -> list[dict]:
    """Generate all QA pairs for one page — text and visual.

    Returns a list of partial QA dicts (without id, source_doc, source_page,
    source_page_id, relevant_page_ids fields — those are filled by the caller).
    """
    results: list[dict] = []

    # Text QA from full_text + tables
    text_pairs = generate_text_qa(page, client, n=text_per_page)
    for pair in text_pairs:
        results.append({
            "question":      pair.get("question", ""),
            "answer":        pair.get("answer", ""),
            "query_type":    "text",
            "question_type": pair.get("question_type", "factual"),
            "query_image_path": None,
        })

    # Visual QA from diagram figure crops (exclude page screenshots and table images)
    diagram_figures = [
        f for f in page.get("figures", [])
        if f.get("label") == "figure"
    ]
    for fig in diagram_figures[:2]:   # cap at 2 visual QA per page
        vqa = generate_visual_qa(page, fig, client)
        if vqa:
            results.append({
                "question":      vqa.get("question", ""),
                "answer":        vqa.get("answer", ""),
                "query_type":    "visual",
                "question_type": vqa.get("question_type", "visual"),
                "query_image_path": fig["path"],
            })

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from Docling-parsed page JSONs using GPT-4o."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the parsed/ directory containing per-page JSON files.",
    )
    parser.add_argument(
        "--text-per-page", type=int, default=DEFAULT_TEXT_PER_PAGE,
        help=f"Number of text QA pairs to generate per page (default: {DEFAULT_TEXT_PER_PAGE}).",
    )
    args = parser.parse_args()

    parsed_dir = Path(args.input)
    if not parsed_dir.is_dir():
        parser.error(f"--input must be an existing directory: {parsed_dir}")

    output_path = parsed_dir.parent / "qa_pairs.json"

    page_files = sorted(parsed_dir.glob("*.json"))
    if not page_files:
        print(f"No JSON files found in {parsed_dir}")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key)

    print("=" * 60)
    print("QA Pair Generator")
    print(f"  Input  : {parsed_dir}")
    print(f"  Output : {output_path}")
    print(f"  Pages  : {len(page_files)}")
    print(f"  Text QA/page : {args.text_per_page}")
    print("=" * 60)

    all_pairs: list[dict] = []
    counter = 0

    for page_file in page_files:
        page = json.loads(page_file.read_text())
        page_id = page["page_id"]
        print(f"  [page] {page_id}")

        partial_pairs = process_page(page, client, text_per_page=args.text_per_page)

        for pair in partial_pairs:
            pair["id"] = f"{parsed_dir.parent.name}_{counter:04d}"
            pair["source_doc"] = page["doc_id"]
            pair["source_page"] = page["page_no"]
            pair["source_page_id"] = page_id
            pair["relevant_page_ids"] = [page_id]
            # Reorder keys for readability
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
                "query_image_path":  pair["query_image_path"],
            })
            counter += 1

        print(f"    → {len(partial_pairs)} pairs ({sum(1 for p in partial_pairs if p['query_type'] == 'text')} text, "
              f"{sum(1 for p in partial_pairs if p['query_type'] == 'visual')} visual)")

    output_path.write_text(json.dumps(all_pairs, indent=2, ensure_ascii=False))
    text_count = sum(1 for p in all_pairs if p["query_type"] == "text")
    visual_count = sum(1 for p in all_pairs if p["query_type"] == "visual")
    print(f"\nDone. {len(all_pairs)} QA pairs written to {output_path}")
    print(f"  text: {text_count}  visual: {visual_count}")


if __name__ == "__main__":
    main()
