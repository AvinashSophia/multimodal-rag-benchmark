"""Generic multimodal document parser using Docling.

Takes any folder of PDFs (or a single PDF) and produces clean structured
per-page JSON files ready to feed into a multimodal RAG pipeline.

Per-page JSON schema
--------------------
{
  "page_id":     "assembly_instructions_p03",
  "doc_id":      "assembly_instructions",
  "source_file": "Assembly Instructions.pdf",
  "page_no":     3,
  "text_blocks": [
    {"text": "Step 3: attach the bracket...", "label": "list_item", "bbox": {...}}
  ],
  "tables": [
    {
      "headers":  ["Part", "Qty", "Length"],
      "rows":     [["BATT+", "1", "180mm"], ...],
      "markdown": "| Part | Qty | Length |\n|---...",
      "summary":  "Table on page 3 with 8 rows, 3 columns: Part, Qty, Length. Sample: Part: BATT+, Qty: 1, Length: 180mm.",
      "image_path": "parsed/figures/assembly_instructions_p03_tbl_00.png"
    }
  ],
  "figures": [
    {
      "path":   "parsed/figures/assembly_instructions_p03_fig_00.png",
      "width":  420, "height": 310,
      "bbox":   {...}, "caption": "",
      "label":  "figure"              # "figure" | "page_screenshot"
    }
  ],
  "full_text": "Step 3: attach the bracket..."
}

Enhancements over raw Docling output
-------------------------------------
1. Ligature fix       : "Flash i ng" → "Flashing" (3-pass regex, universal)
2. Table images       : each table rendered as PNG for CLIP embedding
3. Table summaries    : natural-language description for semantic search
4. Page screenshots   : full-page 150 DPI PNG for layout-aware retrieval
5. Noise filter       : figures smaller than MIN_FIG_PX are discarded
6. OCR flag           : --ocr enables Docling OCR for scanned documents

Every run always re-parses all PDFs from scratch.

Output layout
-------------
{output_dir}/
  {doc_id}_p{NN:02d}.json
  figures/
    {doc_id}_p{NN:02d}_fig_{K:02d}.png     extracted figure crops
    {doc_id}_p{NN:02d}_tbl_{K:02d}.png     rendered table images
    {doc_id}_p{NN:02d}_page.png            full-page screenshot

Usage
-----
    uv run python scripts/parse_documents.py --input data/altumint/
    uv run python scripts/parse_documents.py --input data/acme/manual.pdf
    uv run python scripts/parse_documents.py --input data/acme/ --output data/acme/parsed/
    uv run python scripts/parse_documents.py --input data/altumint/ --ocr
"""

import argparse
import base64
import io
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF — used for full-page screenshots
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_FIG_PX = 50   # discard figure crops smaller than this in either dimension


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def fix_ligatures(text: str) -> str:
    """Repair ligature-split text produced by certain PDF font encodings.

    Some PDFs encode fi/fl/ffi ligatures in a way that produces spurious
    spaces: "Flash i ng" → "Flashing", "Electron i cs" → "Electronics".
    Applied 3 times to handle double splits like "K i tt i ng" → "Kitting".
    """
    for _ in range(3):
        text = re.sub(r"(\w) i (\w)", r"\1i\2", text)
    return re.sub(r"  +", " ", text).strip()


def slugify(text: str) -> str:
    """Convert a filename stem into a clean doc_id slug.

    "Altumint-TM001(02) FLVM Assembly Instructions-240326"
        → "altumint_tm001_02_flvm_assembly_instructions"
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    # Strip trailing date-like suffixes: _YYYYMMDD or _YYMMDD_HHMMSS
    text = re.sub(r"_\d{6,8}(_\d{6})?$", "", text)
    return text


# ---------------------------------------------------------------------------
# Table utilities
# ---------------------------------------------------------------------------

def cells_to_table(cells: list[dict], num_rows: int, num_cols: int) -> dict[str, Any]:
    """Convert Docling table_cells list to {headers, rows, markdown}."""
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


def generate_table_summary(
    headers: list[str], rows: list[list[str]], page_no: int, source_file: str
) -> str:
    """Natural-language description of a table for semantic search."""
    n_rows = len(rows)
    n_cols = len(headers) if headers else (len(rows[0]) if rows else 0)
    col_str = ", ".join(h for h in headers[:6] if h)
    if n_cols > 6:
        col_str += f", ... ({n_cols} total)"

    summary = (
        f"Table on page {page_no} of {source_file} "
        f"with {n_rows} rows and {n_cols} columns."
    )
    if col_str:
        summary += f" Columns: {col_str}."
    if rows and headers:
        sample = ", ".join(
            f"{h}: {v}" for h, v in zip(headers[:4], rows[0][:4]) if h and v
        )
        if sample:
            summary += f" Sample: {sample}."
    return summary


def render_table_image(headers: list[str], rows: list[list[str]]) -> Image.Image | None:
    """Render a table as a PIL Image using matplotlib.

    The image is used for CLIP embedding so tables are findable by visual
    queries (e.g. 'show me the wire length table').
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        if not rows:
            return None

        n_cols = len(headers) if headers else len(rows[0])
        fig_w  = min(14, max(4, n_cols * 1.8))
        fig_h  = min(10, max(2, len(rows) * 0.4 + 1.2))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        tbl = ax.table(
            cellText=rows,
            colLabels=headers if headers else None,
            loc="center",
            cellLoc="left",
        )
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
# Figure utilities
# ---------------------------------------------------------------------------

def decode_figure_image(uri: str) -> Image.Image | None:
    """Decode a base64 data-URI PNG from Docling's picture item."""
    if not uri.startswith("data:image"):
        return None
    try:
        _, b64 = uri.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception:
        return None


def render_page_screenshot(pdf_path: Path, page_no: int, dpi: int = 150) -> Image.Image:
    """Render a full PDF page as a PIL Image using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pix = doc[page_no - 1].get_pixmap(dpi=dpi)   # page_no is 1-indexed
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


# ---------------------------------------------------------------------------
# Per-page extraction
# ---------------------------------------------------------------------------

def extract_page(doc_dict: dict[str, Any], page_no: int) -> dict[str, Any]:
    """Extract raw text blocks, table data, and figure metadata for one page."""

    text_blocks: list[dict] = []
    tables:      list[dict] = []
    figures:     list[dict] = []

    # --- Text ---
    for item in doc_dict.get("texts", []):
        for prov in item.get("prov", []):
            if prov.get("page_no") != page_no:
                continue
            text = fix_ligatures(item.get("text", "").strip())
            if text:
                text_blocks.append({
                    "text":  text,
                    "label": item.get("label", "text"),
                    "bbox":  prov.get("bbox", {}),
                })

    # --- Tables ---
    for item in doc_dict.get("tables", []):
        for prov in item.get("prov", []):
            if prov.get("page_no") != page_no:
                continue
            data   = item.get("data", {})
            n_rows = data.get("num_rows", 0)
            n_cols = data.get("num_cols", 0)
            if n_rows > 0 and n_cols > 0:
                tables.append(cells_to_table(data.get("table_cells", []), n_rows, n_cols))

    # --- Figures ---
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
# Single PDF parser
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: Path, output_dir: Path, ocr: bool) -> int:
    """Parse one PDF and write per-page JSON + figures. Returns pages written."""

    doc_id  = slugify(pdf_path.stem)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [parse] {pdf_path.name}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure      = True
    pipeline_options.do_ocr                  = ocr
    pipeline_options.images_scale            = 2.0
    pipeline_options.generate_page_images    = True
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result   = converter.convert(str(pdf_path))
    doc_dict = result.document.export_to_dict()

    page_numbers  = sorted(int(k) for k in doc_dict.get("pages", {"1": {}}).keys())
    pages_written = 0

    for page_no in page_numbers:
        raw     = extract_page(doc_dict, page_no)
        page_id = f"{doc_id}_p{page_no:02d}"

        # ---- Figure crops (from Docling) --------------------------------
        figures_out: list[dict] = []
        for k, fig in enumerate(raw["figures"]):
            w, h = fig["width"], fig["height"]

            # Discard noise: tiny icons below minimum size threshold
            if w < MIN_FIG_PX or h < MIN_FIG_PX:
                continue

            img = decode_figure_image(fig["uri"])
            if img is None:
                continue

            fig_path = fig_dir / f"{page_id}_fig_{k:02d}.png"
            img.save(fig_path)
            figures_out.append({
                "path":    str(fig_path),
                "width":   w,
                "height":  h,
                "bbox":    fig["bbox"],
                "caption": fig["caption"],
                "label":   "figure",
            })

        # ---- Page screenshot --------------------------------------------
        try:
            screenshot = render_page_screenshot(pdf_path, page_no)
            ss_path    = fig_dir / f"{page_id}_page.png"
            screenshot.save(ss_path)
            figures_out.append({
                "path":    str(ss_path),
                "width":   screenshot.width,
                "height":  screenshot.height,
                "bbox":    {},
                "caption": f"Full page screenshot — page {page_no} of {pdf_path.name}",
                "label":   "page_screenshot",
            })
        except Exception as exc:
            print(f"    WARNING: page screenshot failed for {page_id}: {exc}")

        # ---- Tables: add summary + rendered image -----------------------
        tables_out: list[dict] = []
        for k, tbl in enumerate(raw["tables"]):
            tbl["summary"] = generate_table_summary(
                tbl["headers"], tbl["rows"], page_no, pdf_path.name
            )
            tbl_img = render_table_image(tbl["headers"], tbl["rows"])
            if tbl_img is not None:
                tbl_img_path = fig_dir / f"{page_id}_tbl_{k:02d}.png"
                tbl_img.save(tbl_img_path)
                tbl["image_path"] = str(tbl_img_path)
                # Also add table image to figures for CLIP indexing
                figures_out.append({
                    "path":    str(tbl_img_path),
                    "width":   tbl_img.width,
                    "height":  tbl_img.height,
                    "bbox":    {},
                    "caption": tbl["summary"],
                    "label":   "table_image",
                })
            else:
                tbl["image_path"] = None
            tables_out.append(tbl)

        # ---- Assemble final page JSON -----------------------------------
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

        out_path = output_dir / f"{page_id}.json"
        out_path.write_text(json.dumps(page_out, indent=2, ensure_ascii=False))
        pages_written += 1

    print(f"    → {pages_written} pages written to {output_dir}/")
    return pages_written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse PDFs into per-page structured JSON for multimodal RAG."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a PDF file or a directory of PDF files."
    )
    parser.add_argument(
        "--output",
        help=(
            "Output directory. Defaults to {input_dir}/parsed/ "
            "when --input is a directory, or {pdf_dir}/parsed/ for a single file."
        ),
    )
    parser.add_argument(
        "--ocr", action="store_true",
        help="Enable Docling OCR for scanned or image-based PDFs."
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            parser.error(f"--input must be a .pdf file or a directory, got: {input_path}")
        pdf_files      = [input_path]
        default_output = input_path.parent / "parsed"
    elif input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {input_path}")
            return
        default_output = input_path / "parsed"
    else:
        parser.error(f"--input path does not exist: {input_path}")

    output_dir = Path(args.output) if args.output else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Document Parser")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_dir}")
    print(f"  PDFs   : {len(pdf_files)}")
    print(f"  OCR    : {'on' if args.ocr else 'off'}")
    print("=" * 60)

    total_pages = 0
    for pdf_path in pdf_files:
        total_pages += parse_pdf(pdf_path, output_dir, ocr=args.ocr)

    print(f"\nDone. {total_pages} pages written to {output_dir}/")


if __name__ == "__main__":
    main()
