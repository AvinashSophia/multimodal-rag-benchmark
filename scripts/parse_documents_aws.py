"""Multimodal document parser using Docling — reads PDFs from S3, uploads output to S3.

AWS variant of scripts/parse_documents.py. Parsing logic is identical;
both input (PDFs) and output (parsed JSONs + images) are on S3 — no local disk storage.

S3 input layout:
    benchmarking/datasets/{dataset}/pdfs/*.pdf       ← raw PDFs (upload these first)

S3 output layout:
    benchmarking/datasets/{dataset}/parsed/{page_id}.json
    benchmarking/images/{dataset}/figures/{page_id}_fig_{k}.png
    benchmarking/images/{dataset}/figures/{page_id}_tbl_{k}.png
    benchmarking/images/{dataset}/figures/{page_id}_page.png

The `path` field inside each page JSON stores the S3 key of the image,
not a local filesystem path. altumint_aws.py (the AWS dataset loader)
reads these keys and fetches images from S3.

Upload PDFs to S3 first:
    aws s3 cp data/altumint/ s3://spatial-ai-staging-processing-632872792182/benchmarking/datasets/altumint/pdfs/ --recursive --include "*.pdf"

Then parse:
    uv run python scripts/parse_documents_aws.py --dataset altumint --config configs/aws.yaml
"""

import argparse
import base64
import io
import json
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import fitz
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from PIL import Image

from pipeline.utils import load_config
from pipeline.utils.s3 import S3Client

MIN_FIG_PX = 50


# ---------------------------------------------------------------------------
# Text utilities  (identical to parse_documents.py)
# ---------------------------------------------------------------------------

def fix_ligatures(text: str) -> str:
    for _ in range(3):
        text = re.sub(r"(\w) i (\w)", r"\1i\2", text)
    return re.sub(r"  +", " ", text).strip()


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    text = re.sub(r"_\d{6,8}(_\d{6})?$", "", text)
    return text


# ---------------------------------------------------------------------------
# Table utilities  (identical to parse_documents.py)
# ---------------------------------------------------------------------------

def cells_to_table(cells: list[dict], num_rows: int, num_cols: int) -> dict[str, Any]:
    grid: list[list[str]] = [[""] * num_cols for _ in range(num_rows)]
    header_row_idx = -1
    for cell in cells:
        r = cell.get("start_row_offset_idx", 0)
        c = cell.get("start_col_offset_idx", 0)
        if 0 <= r < num_rows and 0 <= c < num_cols:
            grid[r][c] = fix_ligatures(cell.get("text", "").strip())
        if cell.get("column_header") and r > header_row_idx:
            header_row_idx = r
    if header_row_idx < 0:
        headers: list[str] = []
        rows = grid
    else:
        headers = grid[header_row_idx]
        rows = [row for i, row in enumerate(grid) if i != header_row_idx]
    while rows and all(c == "" for c in rows[-1]):
        rows.pop()
    return {"headers": headers, "rows": rows, "markdown": _to_markdown(headers, rows)}


def _to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    all_rows = ([headers] if headers else []) + rows
    if not all_rows:
        return ""
    col_widths = [max(len(r[c]) for r in all_rows) for c in range(len(all_rows[0]))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"

    lines = []
    if headers:
        lines.append(fmt(headers))
        lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")
    for row in rows:
        lines.append(fmt(row))
    return "\n".join(lines)


def generate_table_summary(headers: list[str], rows: list[list[str]], page_no: int, source_file: str) -> str:
    n_rows = len(rows)
    n_cols = len(headers) if headers else (len(rows[0]) if rows else 0)
    col_str = ", ".join(h for h in headers[:6] if h)
    if n_cols > 6:
        col_str += f", ... ({n_cols} total)"
    summary = f"Table on page {page_no} of {source_file} with {n_rows} rows and {n_cols} columns."
    if col_str:
        summary += f" Columns: {col_str}."
    if rows and headers:
        sample = ", ".join(f"{h}: {v}" for h, v in zip(headers[:4], rows[0][:4]) if h and v)
        if sample:
            summary += f" Sample: {sample}."
    return summary


def render_table_image(headers: list[str], rows: list[list[str]]) -> Image.Image | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
        if not rows:
            return None
        n_cols = len(headers) if headers else len(rows[0])
        fig_w = min(14, max(4, n_cols * 1.8))
        fig_h = min(10, max(2, len(rows) * 0.4 + 1.2))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        tbl = ax.table(cellText=rows, colLabels=headers if headers else None, loc="center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width(col=list(range(n_cols)))
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Figure utilities  (identical to parse_documents.py)
# ---------------------------------------------------------------------------

def decode_figure_image(uri: str) -> Image.Image | None:
    if not uri.startswith("data:image"):
        return None
    try:
        _, b64 = uri.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception:
        return None


def render_page_screenshot(pdf_path: Path, page_no: int, dpi: int = 150) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    pix = doc[page_no - 1].get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


# ---------------------------------------------------------------------------
# Per-page extraction  (identical to parse_documents.py)
# ---------------------------------------------------------------------------

def extract_page(doc_dict: dict[str, Any], page_no: int) -> dict[str, Any]:
    text_blocks: list[dict] = []
    tables: list[dict] = []
    figures: list[dict] = []

    for item in doc_dict.get("texts", []):
        for prov in item.get("prov", []):
            if prov.get("page_no") != page_no:
                continue
            text = fix_ligatures(item.get("text", "").strip())
            if text:
                text_blocks.append({"text": text, "label": item.get("label", "text"), "bbox": prov.get("bbox", {})})

    for item in doc_dict.get("tables", []):
        for prov in item.get("prov", []):
            if prov.get("page_no") != page_no:
                continue
            data = item.get("data", {})
            n_rows = data.get("num_rows", 0)
            n_cols = data.get("num_cols", 0)
            if n_rows > 0 and n_cols > 0:
                tables.append(cells_to_table(data.get("table_cells", []), n_rows, n_cols))

    for item in doc_dict.get("pictures", []):
        for prov in item.get("prov", []):
            if prov.get("page_no") != page_no:
                continue
            caption = ""
            for cap_ref in item.get("captions", []):
                ref = cap_ref.get("$ref", "")
                if ref.startswith("#/texts/"):
                    idx = int(ref.split("/")[-1])
                    cap_items = doc_dict.get("texts", [])
                    if idx < len(cap_items):
                        caption = fix_ligatures(cap_items[idx].get("text", ""))
            img_meta = item.get("image", {})
            figures.append({
                "uri":     img_meta.get("uri", ""),
                "width":   img_meta.get("size", {}).get("width", 0),
                "height":  img_meta.get("size", {}).get("height", 0),
                "bbox":    prov.get("bbox", {}),
                "caption": caption,
            })

    return {"text_blocks": text_blocks, "tables": tables, "figures": figures}


# ---------------------------------------------------------------------------
# S3 image upload helper
# ---------------------------------------------------------------------------

def _pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Single PDF parser — uploads directly to S3
# ---------------------------------------------------------------------------

def parse_pdf_aws(pdf_path: Path, dataset: str, s3: S3Client) -> int:
    """Parse one PDF and upload per-page JSON + images to S3. Returns pages written."""
    doc_id = slugify(pdf_path.stem)

    print(f"  [parse_aws] {pdf_path.name}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.do_ocr = False
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result = converter.convert(str(pdf_path))
    doc_dict = result.document.export_to_dict()

    page_numbers = sorted(int(k) for k in doc_dict.get("pages", {"1": {}}).keys())
    pages_written = 0

    for page_no in page_numbers:
        raw = extract_page(doc_dict, page_no)
        page_id = f"{doc_id}_p{page_no:02d}"

        # ---- Figure crops -----------------------------------------------
        figures_out: list[dict] = []
        for k, fig in enumerate(raw["figures"]):
            w, h = fig["width"], fig["height"]
            if w < MIN_FIG_PX or h < MIN_FIG_PX:
                continue
            img = decode_figure_image(fig["uri"])
            if img is None:
                continue
            filename = f"{page_id}_fig_{k:02d}.png"
            s3_key = s3.image_key(dataset, f"figures/{filename}")
            s3.upload_bytes(_pil_to_bytes(img), s3_key, content_type="image/png")
            figures_out.append({
                "path":    s3_key,    # S3 key — not a local path
                "width":   w,
                "height":  h,
                "bbox":    fig["bbox"],
                "caption": fig["caption"],
                "label":   "figure",
            })

        # ---- Page screenshot --------------------------------------------
        try:
            screenshot = render_page_screenshot(pdf_path, page_no)
            filename = f"{page_id}_page.png"
            s3_key = s3.image_key(dataset, f"figures/{filename}")
            s3.upload_bytes(_pil_to_bytes(screenshot), s3_key, content_type="image/png")
            figures_out.append({
                "path":    s3_key,
                "width":   screenshot.width,
                "height":  screenshot.height,
                "bbox":    {},
                "caption": f"Full page screenshot — page {page_no} of {pdf_path.name}",
                "label":   "page_screenshot",
            })
        except Exception as exc:
            print(f"    WARNING: page screenshot failed for {page_id}: {exc}")

        # ---- Tables: summary + rendered image ---------------------------
        tables_out: list[dict] = []
        for k, tbl in enumerate(raw["tables"]):
            tbl["summary"] = generate_table_summary(tbl["headers"], tbl["rows"], page_no, pdf_path.name)
            tbl_img = render_table_image(tbl["headers"], tbl["rows"])
            if tbl_img is not None:
                filename = f"{page_id}_tbl_{k:02d}.png"
                s3_key = s3.image_key(dataset, f"figures/{filename}")
                s3.upload_bytes(_pil_to_bytes(tbl_img), s3_key, content_type="image/png")
                tbl["image_path"] = s3_key
                figures_out.append({
                    "path":    s3_key,
                    "width":   tbl_img.width,
                    "height":  tbl_img.height,
                    "bbox":    {},
                    "caption": tbl["summary"],
                    "label":   "table_image",
                })
            else:
                tbl["image_path"] = None
            tables_out.append(tbl)

        # ---- Assemble page JSON and upload ------------------------------
        full_text = " ".join(b["text"] for b in raw["text_blocks"])
        page_out = {
            "page_id":     page_id,
            "doc_id":      doc_id,
            "source_file": pdf_path.name,
            "page_no":     page_no,
            "text_blocks": raw["text_blocks"],
            "tables":      tables_out,
            "figures":     figures_out,
            "full_text":   full_text,
        }

        json_key = s3.dataset_key(dataset, f"parsed/{page_id}.json")
        s3.upload_json(page_out, json_key)
        pages_written += 1

    print(f"    → {pages_written} pages uploaded to s3://{s3.bucket}/benchmarking/datasets/{dataset}/parsed/")
    return pages_written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse PDFs from S3 and upload per-page structured JSON + images to S3."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name — used as S3 key prefix (e.g. altumint).")
    parser.add_argument("--ocr",     action="store_true", help="Enable Docling OCR.")
    parser.add_argument("--config",  default=None,        help="Path to config YAML (default: uses built-in S3 defaults).")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    s3 = S3Client(config)

    # List PDF keys under benchmarking/datasets/{dataset}/pdfs/
    pdf_prefix = s3.pdfs_key(args.dataset, "")
    pdf_keys = sorted(k for k in s3.list_keys(pdf_prefix) if k.lower().endswith(".pdf"))

    if not pdf_keys:
        print(f"No PDFs found at s3://{s3.bucket}/{pdf_prefix}")
        print(f"Upload PDFs first:")
        print(f"  aws s3 cp data/{args.dataset}/ s3://{s3.bucket}/{pdf_prefix} --recursive --include '*.pdf'")
        return

    print("=" * 60)
    print("Document Parser (AWS)")
    print(f"  Dataset : {args.dataset}")
    print(f"  PDFs    : {len(pdf_keys)}")
    print(f"  OCR     : {'on' if args.ocr else 'off'}")
    print(f"  Bucket  : {s3.bucket}")
    print(f"  Input   : s3://{s3.bucket}/{pdf_prefix}")
    print("=" * 60)

    total_pages = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        for pdf_key in pdf_keys:
            filename = pdf_key.split("/")[-1]
            tmp_path = Path(tmp_dir) / filename
            print(f"  [download] s3://{s3.bucket}/{pdf_key} → {tmp_path.name}")
            s3.download_file(pdf_key, tmp_path)
            total_pages += parse_pdf_aws(tmp_path, args.dataset, s3)
            tmp_path.unlink()  # free disk space between PDFs

    print(f"\nDone. {total_pages} pages uploaded to S3.")


if __name__ == "__main__":
    main()
