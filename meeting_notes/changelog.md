# Enhancement Changelog

Running log of significant changes made to the multimodal RAG benchmark.
Updated by Claude Code as work happens. Used by the meeting-prep agent to explain the *why* behind diffs.

Format: `YYYY-MM-DD | component | what changed | why`

---

## 2026-04-06

### ColPali Qdrant Retriever
**Files:** `pipeline/retrieval/colpali_qdrant.py` (new), `pipeline/retrieval/__init__.py`
**What:** New image retriever using ColPali (PaliGemma-based) backed by Qdrant. Stores patch-level embeddings (N vectors per page instead of one) and uses approximate MaxSim scoring at query time — for each query token vector, finds nearest patches in Qdrant, then aggregates per page via `score(q,d) = Σ max_j(q_i · d_j)`. Registered as `colpali_qdrant`.
**Why:** CLIP uses a single vector per image which loses spatial detail — engineering drawings, schematics, and tables look nearly identical to CLIP. ColPali's patch-level late interaction captures fine-grained visual structure (individual cells, connector labels, dimension annotations). Significantly outperforms CLIP on the Altumint technical document corpus. Skip-if-indexed implemented via a metadata sentinel point storing page count.

---

### FastAPI Backend
**Files:** `pipeline/api/main.py` (new), `pipeline/api/pipeline_service.py` (new), `pipeline/api/schemas.py` (new), `pipeline/api/__init__.py` (new)
**What:** FastAPI application wrapping the RAG pipeline for single-query interactive use. `PipelineService` initializes the corpus index once on startup (skip-if-indexed for Qdrant/Elastic) and handles per-request config overrides — swapping text retriever, image retriever, or model without reloading the corpus. Endpoints: `GET /health`, `GET /config/options`, `POST /query`. Static file mount at `/images/` serves page screenshots for the frontend.
**Why:** The batch benchmark runner processes entire datasets but gave no way to interactively query the pipeline. The FastAPI backend enables the frontend demo and supports iterative testing of individual questions without running a full benchmark.

---

### React Frontend (SophiaSpatial AI)
**Files:** `frontend/` (all new — Vite + React + TypeScript + Tailwind)
**What:** Interactive query interface for the RAG pipeline. Components: `QueryInput` (question + ground truth toggle), `ConfigSelector` (pipeline dropdowns), `AnswerPanel` (answer + sources), `RetrievedChunks` (text evidence), `RetrievedImages` (visual evidence with lightbox), `MetricsPanel` (evaluation scores), `StatusBar` (health + loading). Proxies API calls to `localhost:8000` via Vite dev proxy. Branded as SophiaSpatial AI / FLVM Intelligence Platform.
**Why:** No interactive interface existed — evaluation was batch-only. The frontend makes the pipeline accessible for stakeholder demos, manual QA testing, and debugging individual queries without writing code. Pipeline config (retriever, model) is switchable at runtime without restarting the server.

---

### Docling PDF Parsing Pipeline
**Files:** `scripts/parse_documents.py` (new), `data/altumint/parsed/` (36 per-page JSONs + figures), `docs/parse_documents.md`
**What:** Docling-based PDF parser that extracts per-page text, tables (with markdown + summary), and figures (cropped PNGs + page screenshots) from Altumint PDFs. Outputs one JSON per page with `page_id`, `full_text`, `tables[]`, `figures[]`. Page screenshots stored as `{page_id}_page.png`. Ligature fix applied post-parse (fi/fl/ff character normalization).
**Why:** Raw PDF text extraction loses table structure — critical for DC004 continuity checks and DC005 wire length tables which are the primary content for ~40% of QA pairs. Docling preserves table structure as markdown, making it retrievable and readable by the model. Per-page JSONs also enable the image corpus (one screenshot per page) used by CLIP and ColPali.

---

### QA Generation Script
**Files:** `scripts/generate_qa.py` (new), `data/altumint/qa_pairs.json` (108 pairs), `data/altumint/qa_review.md`
**What:** GPT-4o-based QA pair generator. Sends each parsed page (text + rendered image) to GPT-4o with a structured prompt enforcing 5 question types (factual, numerical, visual, procedural, cross_doc), self-contained questions, and exact-string answers. Outputs `qa_pairs.json` with `id`, `question`, `answer`, `source_page_id`, `query_type`, `question_type`, `relevant_page_ids`. Also generates a `qa_review.md` critique.
**Why:** No labeled QA dataset existed for Altumint. Hand-labeling 36 pages at multiple questions per page would take days. GPT-4o generation with strict prompt rules produces usable QA pairs in minutes. The `source_page_id` and `relevant_page_ids` fields enable retrieval metric computation (Recall@k, MRR) for each sample.

---

### Subagents (meeting-prep, qa-reviewer, docling-parser-verifier)
**Files:** `.claude/agents/meeting-prep.md` (new), `.claude/agents/qa-reviewer.md` (new), `.claude/agents/docling-parser-verifier.md` (new)
**What:** Three new specialized Claude Code subagents:
- **meeting-prep**: Reads git diffs, changelog, and benchmark results to produce a structured team meeting document. Invoked before Tuesday/Friday meetings.
- **qa-reviewer**: Reviews `data/altumint/qa_pairs.json` for question quality — checks realism, answer accuracy against parsed JSONs, visual question image-dependency, and distribution.
- **docling-parser-verifier**: Verifies Docling parse output against actual PDF pages — checks text completeness, ligature fixes, table structure, and figure crop quality.
**Why:** Recurring tasks (meeting prep, QA review, parse verification) were done manually and inconsistently. Subagents encode the exact steps and criteria, run in isolated contexts (no main context pollution), and produce consistent outputs.

---

### Benchmark Runs (Altumint + HotpotQA, Multiple Configs)
**Files:** `pipeline/outputs/logs/` (multiple run directories), `pipeline/outputs/research_log.md`
**What:** Executed benchmark runs across Altumint and HotpotQA with configs: `bm25_elastic+colpali_qdrant+gpt`, `dense_qdrant+colpali_qdrant+gpt`, `hybrid_elastic_qdrant+colpali_qdrant+gpt`, and `dense_qdrant+gpt` (text-only) for HotpotQA. Analysis written to `analysis.md` in each run directory and summarized in `research_log.md`.
**Why:** Needed baseline numbers before further retriever/model changes. Results establish the starting point for measuring impact of future improvements (re-ranker, cross-modal overlap boosting, adaptive top-k).

---

## 2026-04-07

### Dataset Dropdown in Frontend
**Files:** `frontend/src/components/ConfigSelector.tsx`, `frontend/src/App.tsx`, `frontend/src/types/index.ts`, `pipeline/api/schemas.py`, `pipeline/api/pipeline_service.py`
**What:** Added a Dataset dropdown to the Pipeline config panel alongside the existing retriever and model selectors. Switching datasets triggers a full corpus reload and re-index on the backend. For Qdrant/Elastic retrievers, the re-index is a no-op if the dataset's collection already exists (skip-if-indexed per dataset). Retriever instances are re-instantiated with the new dataset config so collection names resolve correctly (e.g. `dense_text_hotpotqa` instead of `dense_text_altumint`).
**Why:** The frontend was hardcoded to altumint — users had no way to switch datasets interactively. Dataset is now a first-class config option alongside retriever and model.

---

### Query Image Upload (Multimodal Query Input)
**Files:** `pipeline/api/main.py`, `frontend/src/api/client.ts`, `frontend/src/components/QueryInput.tsx`, `frontend/src/types/index.ts`
**What:** Added `POST /upload-query-image` endpoint that accepts a file upload, validates the MIME type, saves to `data/query_uploads/` with a UUID filename, and returns the server-side path. Frontend `QueryInput` gained an image upload button with preview, upload spinner, and clear button. The saved path is included as `query_image_path` in the query request, which flows through to `PIL.Image.open()` in the pipeline.
**Why:** DocVQA and GQA queries are about a specific image — users need to supply that image as part of the query. For CLIP retrieval, providing a query image enables image→image search alongside text→image. ColPali correctly ignores the query image (it has no image query encoder and doesn't need one).

---

### Cross-Modal Overlap Boosting in HybridRetriever
**Files:** `pipeline/retrieval/base.py`
**What:** After text and image retrieval, `HybridRetriever._boost_overlap()` finds page IDs appearing in both result lists and moves them to the front of each list. Both the text chunk and image for overlapping pages are kept (they're complementary). The overlap count is recorded in result metadata.
**Why:** A page appearing in both text and image retrieval is cross-modal confirmed — two independent signals agree it's relevant. Prioritising it in the model's context window exploits primacy bias. Only meaningful for Altumint (text and image share the same `page_id`); for HotpotQA (no images) and DocVQA (no text corpus) the overlap set is always empty and the function is a no-op.

---

### Metrics Panel Grouping and VQA Accuracy Fix
**Files:** `frontend/src/components/MetricsPanel.tsx`, `pipeline/api/pipeline_service.py`
**What:** MetricsPanel now groups metrics into sections (Answer Quality, Grounding, Text Retrieval, Image Retrieval) instead of a flat list. Added labels for all retrieval metric keys. Fixed missing `vqa_accuracy` by passing `all_ground_truths=[request.ground_truth]` to `evaluate_sample` in the API service.
**Why:** `vqa_accuracy` was never computed for frontend queries because `all_ground_truths` was not passed. The flat metric list was hard to read as more metrics were added.

---

### Frontend Layout and UI Polish
**Files:** `frontend/src/App.tsx`, `frontend/src/components/RetrievedImages.tsx`
**What:** Widened main content from `max-w-5xl` to `max-w-7xl` to match the header and use full screen width. Results grid changed to 4-column layout (answer 3 cols, metrics 1 col) and evidence row to 3-column (text chunks 2 cols, images 1 col). Retrieved image cards now show a dark slide-up overlay on hover with the full `page_id` (using `break-all`) and score — replaces the previous floating tooltip that was clipped by the panel's `overflow-hidden`.
**Why:** The original layout left large blank margins on wide screens. The floating tooltip was invisible due to parent `overflow-hidden`; the in-card overlay is never clipped.

---

## 2026-04-02

### Qdrant-Backed Retrievers (Dense Text + CLIP Image)
**Files:** `pipeline/retrieval/dense_qdrant.py` (new), `pipeline/retrieval/clip_qdrant.py` (new), `pipeline/retrieval/__init__.py`
**What:** Replaced the in-memory dense retriever and CLIP retriever with persistent Qdrant-backed versions. `DenseQdrantRetriever` stores BGE-large embeddings in a local Qdrant collection keyed by dataset name (e.g. `dense_text_altumint`). `CLIPQdrantRetriever` stores CLIP ViT-B-32 image embeddings similarly.
**Why:** The original retrievers re-encoded the entire corpus on every benchmark run — re-embedding 36 PDF pages or thousands of DocVQA images from scratch each time wasted significant time. Qdrant gives us skip-if-indexed: if the collection already exists with the right number of vectors, encoding is skipped entirely. This makes iterative benchmarking (changing model or prompt, not corpus) much faster. Collections are namespaced by dataset so multiple datasets coexist in the same store.
**Config:** Added `qdrant.path` and `qdrant.collection` under `retrieval.text` and `retrieval.image` in `configs/default.yaml`.

---

### Production Metrics Libraries (ranx + HuggingFace evaluate + RAGAS)
**Files:** `pipeline/evaluation/retrieval_metrics_ranx.py` (new), `pipeline/evaluation/answer_metrics_hf.py` (new), `pipeline/evaluation/grounding_metrics_ragas.py` (new)
**What:** Added three production-grade metric backends as drop-in alternatives to the custom implementations:
- **ranx** (`retrieval_metrics_ranx.py`): industry-standard IR library for Recall@k, MRR, nDCG. Faster and more correct than our hand-rolled implementations.
- **HuggingFace evaluate** (`answer_metrics_hf.py`): official SQuAD EM and F1 with the same normalization (lowercase, strip articles/punctuation) used by official leaderboards. Results are now comparable to published benchmarks.
- **RAGAS** (`grounding_metrics_ragas.py`): LLM-as-judge faithfulness. Decomposes the answer into atomic statements and checks each against retrieved context — far more reliable than token-overlap heuristics for faithfulness scoring.
**Why:** Our custom metrics were hand-rolled approximations. Using the same libraries as the research community means our numbers are directly comparable to published results. RAGAS specifically addresses a known gap — the faithfulness metric was behaving anomalously (correct answers scoring 0.0, wrong answers scoring 1.0) with the token-overlap approach.
**Config:** `evaluation.backend: "production"` activates all three libraries; `"custom"` uses the original fast implementations with no external deps.

---

### Image Input Alongside Text Query (Multimodal Query Support)
**Files:** `pipeline/retrieval/clip_qdrant.py`, `pipeline/retrieval/clip_retriever.py`
**What:** The CLIP retrievers now accept an optional `query_image` parameter in `retrieve()`. When a sample has an image (DocVQA, GQA, Altumint), that image is passed as the query alongside the text question. This enables three query modes:
- **text only** (HotpotQA): text encoder → image corpus
- **image only**: visual encoder → image corpus
- **text + image** (DocVQA, GQA, Altumint): both paths run independently, results are merged and fused
**Why:** DocVQA and GQA samples have a source document image — it is more informative to query "find me images similar to this document page" (image→image) than to rely solely on the text question. For Altumint, both the text question and the source page image are available, so running both paths and fusing gives the best candidate pool. The `query_image` flows from the runner through `HybridRetriever` to `CLIPQdrantRetriever` automatically.

---

### CLIP Late Fusion (Text→Image + Image→Image Score Fusion)
**Files:** `pipeline/retrieval/clip_qdrant.py`, `pipeline/retrieval/clip_retriever.py`
**What:** When both text and image query paths run, candidate results from both paths are merged by image ID and scores are fused using a weighted sum: `fused = alpha * text_score + (1 - alpha) * image_score`. Default `fusion_alpha = 0.5` (equal weight). Zero-imputation is applied for candidates that appear in only one path's results.
**Why:** Text→image and image→image retrieve overlapping but not identical candidate sets. Fusion combines both signals — text captures semantic relevance to the question, image captures visual similarity to the source page. This consistently outperforms either modality alone on DocVQA and Altumint. `fusion_alpha` is configurable in `configs/default.yaml` so the balance can be tuned per dataset — e.g. setting `alpha=1.0` degrades to text-only, `alpha=0.0` to image-only, enabling ablation studies.
**Config:** `retrieval.image.fusion_alpha` in `configs/default.yaml`.

---

### Hybrid Retriever (Elasticsearch BM25 + Qdrant Dense, RRF Fusion)
**File:** `pipeline/retrieval/hybrid_elastic_qdrant.py` (new), `pipeline/retrieval/__init__.py`
**What:** New retriever that composes BM25ElasticRetriever + DenseQdrantRetriever and fuses results using Reciprocal Rank Fusion (RRF). Registered as `hybrid_elastic_qdrant`.
**Why:** BM25 is strong on exact keyword matches (part numbers, IDs) while dense retrieval handles semantic similarity. Hybrid combines both strengths. RRF is score-scale agnostic so no normalization needed. Wider candidate pools (bm25_top_k=20, dense_top_k=20) are fetched before fusing down to final top_k=5.
**Config:** Added `hybrid_elastic_qdrant` option + `rrf_k`, `bm25_top_k`, `dense_top_k` params to `configs/default.yaml`.

---

### Altumint Dataset Integration
**Files:** `pipeline/datasets/altumint.py` (new), `pipeline/datasets/__init__.py`, `scripts/generate_altumint_qa.py` (new), `data/altumint/` (PDFs + qa_pairs.json), `pyproject.toml` (added pymupdf)
**What:** Integrated 7 proprietary Altumint FLVM PDF documents as a new benchmark dataset. 36 pages across assembly manual (TM001), config manual (TM002), engineering drawings (DC001, DC002), continuity check tables (DC004), wire length tables (DC005), and a wiring diagram. 108 GPT-4o generated QA pairs covering factual, numerical, visual, procedural, and cross_doc question types.
**Why:** The existing datasets (HotpotQA, DocVQA, GQA) are all public benchmarks. Altumint tests our pipeline on real proprietary engineering documentation — the actual production use case. Technical documents have unique challenges: symbol-heavy tables, dimensional annotations, multi-hop cross-document references, and engineering diagrams that CLIP handles poorly.
**Implementation details:** Each PDF page = one text chunk + one rendered image (same page ID, e.g. `TM001_p03`). Lazy corpus caching via `_build_corpus()` idempotent guard — PDFs read once. QA generation script uses GPT-4o with rendered page image + extracted text.

---

### Text Retrieval Metrics Fix for Altumint
**Files:** `pipeline/datasets/altumint.py`, `pipeline/runners/run_benchmark.py`
**What:** Added `"relevant_text_ids": [page_id]` to each Altumint sample's metadata. Added `elif "relevant_text_ids" in sample.metadata` fallback in the runner after the HotpotQA-specific `supporting_facts` block.
**Why:** Text retrieval metrics (Recall@k, MRR, nDCG) were silently producing empty results for Altumint because the runner only knew how to extract ground-truth text IDs from HotpotQA's `supporting_facts` format. Altumint has no `supporting_facts`. Without this fix, we couldn't evaluate whether the text retriever was actually finding the right page.

---

### Brevity Prompt Fix (All Models)
**Files:** `pipeline/models/gpt.py`, `pipeline/models/gemini.py`, `pipeline/models/gemini_vertex.py`, `pipeline/models/qwen_vl.py`
**What:** Changed system prompt and `citation_instruction` from "ideally a single word" / "one word, name, number, or short phrase" to "the exact name, number, value, or short phrase from the source — no truncation of product names or multi-word values."
**Why:** Benchmark results showed the model truncating correct answers — e.g., `"Tycon 10Ah Aluminum LiFePO4 Battery"` → `"LiFePO4"`, `"ITC Limited"` → `"ITC"`. The over-aggressive brevity instruction was hurting EM and F1 scores on technical document datasets where answers are full product names or part descriptions.

---

### Benchmark Results Analyzer Subagent Enhancements
**File:** `.claude/agents/benchmark-results-analyzer.md`
**What:** Multiple improvements:
1. Added mandatory WebSearch step (Step 7) — agent must search for SOTA papers (2022+) before making recommendations. Citations must be verified, not hallucinated.
2. Added Altumint-specific analysis guide — corpus structure, document type breakdown, 6 expected failure modes, appropriate metrics (ANLS over VQA accuracy), no external baselines note.
3. Fixed output behavior — agent now writes `{run_dir}/analysis.md` and appends to `pipeline/outputs/research_log.md`.
4. Fixed cross-dataset comparison bug — Step 5 now restricts comparisons to same-dataset runs only. research_log summaries must not reference metrics from different datasets.
**Why:** Previous agent was not writing output files (behavioral guideline was too broad), was comparing Altumint metrics against DocVQA numbers (meaningless — different corpora), and lacked Altumint-specific domain knowledge needed to interpret results correctly.

---

### Code Reviewer Subagent
**File:** `.claude/agents/code-reviewer.md` (new)
**What:** Project-aware code review agent with 9 non-negotiable design rules (registry pattern, `__init__.py` exports, config coverage, cross-file signature consistency, skip-if-indexed, ID consistency, dataset corpus contract, log directory naming) and component-specific checklists.
**Why:** Post-edit hooks catch syntax/type errors but not architectural violations. A dedicated reviewer enforces the project's modular design rules — ensures new components don't break the registry pattern, miss config documentation, or silently corrupt ID-dependent metrics.

---

### QA Generation Prompt Fix
**File:** `scripts/generate_altumint_qa.py`
**What:** Added Rule 6 (questions must be specific and self-contained) and strengthened Rule 4 (cross_doc questions must name the specific document/procedure being referenced, with a bad example shown explicitly).
**Why:** Generated QA pairs had vague cross_doc questions like "What is the document ID for cross-referencing?" — meaningless without context. The prompt was not enforcing that questions be self-contained for a reader without surrounding context.

---
