# Benchmark Run Analysis

## Run Summary

- **Directory:** `altumint_denseqdrant_clip_gpt_20260406_105313`
- **Dataset:** Altumint (proprietary FLVM technical documentation, 7 PDFs, 36 pages)
- **Retrieval:** text=dense_qdrant (BGE-large-en-v1.5, top_k=5), image=CLIP (ViT-B-32, top_k=5, fusion_alpha=0.5)
- **Model:** GPT-4o (temperature=0.0, max_tokens=512)
- **Samples analyzed:** 122 (full dataset -- first complete Altumint run)
- **Corpus:** Text chunks include table markdown + summaries alongside full_text; parsed without `--ocr` for clean text layer
- **ID bugs:** Both fixed (long-form Qdrant IDs, source_page_id for images)

---

## Metric Scorecard

| Metric | Score |
|---|---|
| **Text Recall@5** | 0.762 |
| **Text MRR** | 0.605 |
| **Text nDCG@5** | 0.644 |
| **Image Recall@5** | 0.620 |
| **Image MRR** | 0.386 |
| **Image nDCG@5** | 0.445 |
| **Exact Match** | 0.238 |
| **F1** | 0.529 |
| **Faithfulness** | 0.410 |
| **Attribution Accuracy** | 0.716 |
| **VQA Accuracy** | 0.079 (not meaningful for Altumint) |

---

## Key Findings

### 1. The "Answer:" prefix is the single largest drag on EM

51 of 122 samples (41.8%) have predictions starting with "Answer:". **Every single one has EM=0.0.** The remaining 71 samples without the prefix achieve EM=0.408. If the prefix were stripped, overall EM would roughly double. This is the same brevity prompt bug identified in prior runs but now confirmed at scale.

### 2. Text retrieval is solid; image retrieval is weak due to CLIP clustering

Text Recall@5=0.762 with MRR=0.605 -- the correct text page is in the top-5 roughly 76% of the time and is often rank 1-2. Image Recall@5 is lower at 0.620 with MRR=0.386, and a single page (`TM001_p09`) dominates image top-1 retrieval for **47 of 122 queries** (38.5%). CLIP cannot discriminate between assembly instruction photos.

### 3. Generation is the bottleneck, not retrieval

67 of 122 samples (54.9%) have text_recall@5=1.0 but EM=0.0 -- the correct page was retrieved but the model failed to produce the exact answer. This confirms the pattern from the 5-sample run: retrieval is ahead of generation quality.

### 4. Visual queries are substantially harder than text-only queries

| | Visual (n=50) | Text-only (n=72) |
|---|---|---|
| EM | 0.160 | 0.292 |
| F1 | 0.361 | 0.646 |
| Text Recall@5 | 0.640 | 0.847 |
| Faithfulness | 0.236 | 0.530 |
| Attribution Acc | 0.633 | 0.773 |

Visual queries underperform on every metric. F1 gap is nearly 2x (0.361 vs 0.646).

### 5. DC004 continuity tables are the weakest document type

DC004 achieves only tR@5=0.400 and faithfulness=0.100 across 10 samples. The dense retriever struggles with symbol-heavy table queries (natural language queries for "resistance" vs documents containing "OL", "0", "Omega" symbols). The model also frequently responds "I don't know" to DC004 questions when the correct page is not retrieved.

---

## Per-Document-Type Performance

| Document | Samples | EM | F1 | Text R@5 | Image R@5 | Faith. | Attr. |
|---|---|---|---|---|---|---|---|
| **Wiring_diagram** | 3 | 0.667 | 0.667 | 1.000 | 1.000 | 0.333 | 1.000 |
| **DC005_wire_lengths** | 5 | 0.400 | 0.667 | 0.800 | 1.000 | 0.400 | 0.800 |
| **DC002_hole_location** | 3 | 0.333 | 0.333 | 0.667 | 1.000 | 0.667 | 0.667 |
| **TM002_config** | 41 | 0.293 | 0.607 | 0.780 | 0.632 | 0.439 | 0.748 |
| **TM001_assembly** | 57 | 0.175 | 0.504 | 0.825 | 0.556 | 0.420 | 0.749 |
| **DC004_continuity** | 10 | 0.200 | 0.432 | 0.400 | N/A | 0.100 | 0.400 |
| **DC001_assembly_drawing** | 3 | 0.000 | 0.103 | 0.333 | 1.000 | 0.667 | 0.333 |

**Best:** The wiring diagram and DC005 wire length tables perform best. The wiring diagram text is long and keyword-rich (perfect text retrieval) and DC005 has well-structured table markdown.

**Worst:** DC001 (assembly drawing) has EM=0.000 and F1=0.103 across all 3 samples. Text retrieval is only 0.333 (the drawing is mostly visual with minimal text), and while CLIP retrieves it perfectly (iR@5=1.0), the model cannot extract the right information. DC004 has the worst text retrieval (0.400) and lowest faithfulness (0.100).

---

## Failure Pattern Analysis

### Pattern 1: "Answer:" prefix corrupts EM (51 samples, 41.8%)
The model prepends "Answer: " to 51 predictions. Every one scores EM=0.0. Examples:
- `altumint_0017`: Pred="Answer: 4", GT="4" -- F1=0.667 but EM=0.0
- `altumint_0024`: Pred="Answer: Using zip ties in top right of plate", GT="Using zip ties in top right of plate" -- F1=0.941 but EM=0.0

### Pattern 2: Yellow arrow questions fail completely (4 samples, all EM=0.0, F1=0.0)
All 4 "yellow arrow" questions (`altumint_0058`, `0070`, `0082`, `0118`) score EM=0.0 and F1=0.0. The model defaults to "Netgear switch" or "SmartSolar charge controller" regardless of what the arrow actually points to. This is a visual grounding failure -- the model cannot read spatial pointers in assembly photos.

### Pattern 3: CLIP collapses to TM001_p09 (47/122 queries)
Page `altumint_tm001_02_..._p09` (web relay installation photo) is retrieved as image top-1 for 38.5% of all queries. Average CLIP score spread (top-1 minus top-5) is only 0.0226 -- effectively random ranking. CLIP embeddings cannot differentiate between assembly instruction photos.

### Pattern 4: DC004 table queries miss retrieval (6/10 samples tR@5=0.0)
Dense embeddings (BGE-large) fail on DC004's symbol-dense tables. Queries use natural language ("resistance value for BATT 3 to B- connection") but the indexed text contains dense Omega/OL symbols in markdown tables. The retriever returns TM001 assembly pages instead.

### Pattern 5: Serial number / voltage reading from screenshots (visual hallucination)
- `altumint_0029`: GT="HQ235304FFFM", Pred="HQ2240X9Q5R" (misread serial number from photo)
- `altumint_0102`: GT="13.55V", Pred="0.01V" (misread voltage from app screenshot)
- `altumint_0083`: GT="192.168.0.90", Pred="192.168.0.150" (confused with a different IP on same page)

### Pattern 6: "I don't know" responses (4 samples)
Samples `altumint_0008`, `0012`, `0014`, `0105` return "I don't know" -- all are DC004/TM002 questions where the correct page was not retrieved.

### Pattern 7: Ambiguous questions across documents
- `altumint_0000` ("What is the document title?") and `altumint_0003` ("What is the title of the document?") are ambiguous without page context. The retriever returns the wrong document, and the model answers from whatever it sees.
- `altumint_0060` ("What is the first step mentioned in the procedure?") retrieves TM001_p01 (Step 1: drill holes) instead of TM001_p11 (Step 26: TURN OFF ALL BREAKERS), which is the correct page.

---

## Strengths

1. **Text retrieval is functional at scale.** tR@5=0.762 across 122 samples with the correct page often at rank 1 (MRR=0.605). This is a strong baseline given the 36-page corpus.

2. **Attribution accuracy is good.** 71.6% of model citations point to the correct source document. The model is generally citing what it reads from.

3. **Simple factual lookup works well.** Questions like wire lengths (420mm), IP addresses (192.168.0.150), part numbers (PJPOLEMNT14), and component names (1FT CAT5 cable) achieve EM=1.0 when retrieval succeeds.

4. **Table markdown indexing helps DC005.** The wire lengths table achieves tR@5=0.800 and F1=0.667 -- the table markdown + summary approach works for clean, simple tables.

5. **ID consistency is confirmed.** No ID mismatch artifacts observed across 122 samples. The long-form ID and source_page_id fixes are stable.

---

## Bottlenecks & Weaknesses

### 1. Generation quality (CRITICAL)
- EM=0.238 overall; even without the "Answer:" prefix, EM is only ~0.408
- 67 samples with perfect retrieval still get EM=0.0
- Visual questions: EM=0.160, F1=0.361

### 2. "Answer:" prefix (HIGH -- quick fix, high impact)
- 41.8% of samples have the prefix, and 100% of those score EM=0.0
- Stripping the prefix would likely raise overall EM to ~0.35-0.40

### 3. CLIP image retrieval (HIGH)
- ViT-B-32 cannot differentiate assembly photos -- TM001_p09 dominates 38.5% of queries
- Average top-1-to-top-5 score spread is 0.023 (near-random)
- Cross-modal agreement on visual queries: only 19/50 (38%) have both text AND image retrieving the correct page

### 4. DC004 continuity table retrieval (MEDIUM)
- Dense embeddings fail on symbol-heavy tables (tR@5=0.400)
- Natural language queries vs Omega/OL table content creates a vocabulary mismatch

### 5. Faithfulness metric unreliability (MEDIUM)
- 70/122 samples (57.4%) have faithfulness=0.0
- Some correct answers score faithfulness=0.0 while wrong answers score 1.0
- RAGAS faithfulness remains unreliable for this dataset

### 6. VQA accuracy is meaningless (LOW -- known)
- VQA accuracy=0.079 is an artifact of the multi-annotator protocol applied to single-annotation data
- Should be excluded from Altumint reporting

---

## Comparison to Prior Altumint Runs

| Metric | Run 3 (5 samples, post-fix) | This Run (122 samples) | Delta |
|---|---|---|---|
| Text Recall@5 | 1.000 | 0.762 | -0.238 |
| Text MRR | 0.733 | 0.605 | -0.128 |
| Image Recall@5 | 1.000 | 0.620 | -0.380 |
| EM | 0.200 | 0.238 | +0.038 |
| F1 | 0.456 | 0.529 | +0.073 |
| Attribution Acc | 0.800 | 0.716 | -0.084 |

The 5-sample runs had artificially high retrieval metrics (small sample, easy questions). At 122 samples, retrieval drops significantly -- especially image recall (-0.380), confirming CLIP limitations at scale. EM and F1 actually improved slightly, suggesting the larger QA set has more answerable factual questions. Attribution accuracy held reasonably well.

---

## Recommendations (Prioritized)

### 1. Fix the "Answer:" prefix in the model prompt (HIGH impact, LOW effort)
Strip or prevent the "Answer:" prefix. This single change would raise EM from 0.238 to an estimated 0.35-0.40 -- the largest single metric improvement available. 51/122 samples are affected.

### 2. Add the ANLS metric (HIGH impact, LOW effort)
ANLS would capture partial string matches that EM misses ("Answer: OL" vs "OL", "Ø1.375" vs "1.375 inches", "4" vs "Four (4)"). For a technical document dataset, ANLS is the standard evaluation metric. The current EM/F1 combination underestimates true answer quality.

### 3. Replace CLIP (ViT-B-32) with ColPali for image retrieval (HIGH impact, MEDIUM effort)
CLIP's average score spread of 0.023 across top-5 means it effectively cannot rank document pages. ColPali, trained specifically for document page retrieval using late interaction (ColBERT-style), would address:
- Assembly photo clustering (TM001_p09 dominating 38.5% of queries)
- Engineering drawing discrimination (DC001 vs DC002 vs wiring diagram)
- Screenshot/UI understanding (TM002 configuration pages)

### 4. Add BM25 or hybrid retrieval for DC004 continuity tables (MEDIUM impact, MEDIUM effort)
Dense embeddings (BGE-large) fail on DC004 because natural language queries ("resistance value for BATT 3") don't match the Omega/OL symbol vocabulary in the indexed tables. BM25 would match on exact terms like "BATT 3", "B-", "PV CB". The hybrid_elastic_qdrant retriever already exists in the codebase -- try it for the next run.

### 5. Improve visual grounding for spatial pointer questions (MEDIUM impact, HIGH effort)
Yellow arrow questions fail completely (4/4 at EM=0.0, F1=0.0). The model cannot ground spatial pointers in assembly photos. This likely requires a VLM with stronger spatial reasoning (GPT-4o should handle this but may need prompt engineering to attend to visual annotations).

### 6. Add question-type metadata to QA pairs (LOW effort, enables analysis)
Tagging each sample with `question_type` (factual, numerical, visual, procedural, cross_doc) would enable automated per-type breakdowns. Currently this analysis relies on manual pattern matching.

---

## Anomalies / Flags

1. **CLIP clustering on TM001_p09:** This single page is image top-1 for 47/122 (38.5%) of queries. This is a severe retrieval bias -- the CLIP embedding for this page is a "centroid" that matches nearly everything. This artificially inflates image recall for questions about TM001_p09 while destroying it for all other pages.

2. **Faithfulness metric inversion:** 57.4% of samples score faithfulness=0.0, including many with correct answers and correct citations. The RAGAS faithfulness metric appears to systematically undercount for Altumint. Do not rely on this metric for Altumint analysis.

3. **Ambiguous questions in QA set:** Several questions like "What is the document title?" (altumint_0000, 0003) are ambiguous without specifying which document. These will always fail retrieval unless the question is tied to a specific page.

4. **No cross-modal consistency metric reported.** The config lists `cross_modal_consistency` but it does not appear in metrics.json. This may be a silent evaluation gap.

5. **Duplicate/near-duplicate questions:** Questions about "What component is indicated by the yellow arrow" appear 4 times for different pages but the model gives nearly identical wrong answers. Similarly, "What is the document code for the FLVM Electronics Enclosure?" appears for both DC001 (altumint_0001) and DC002 (altumint_0004).

---

## SOTA Research Recommendations

| Weakness Observed | Recommended Approach | Expected Impact | Citation |
|---|---|---|---|
| CLIP cannot discriminate document pages (score spread=0.023, TM001_p09 dominates 38.5% of queries) | **ColPali** -- vision language model trained for document page retrieval with late interaction (ColBERT-style). Handles layout, tables, figures natively. | Would replace CLIP entirely for document image retrieval. ViDoRe benchmark shows strong gains over CLIP+OCR pipelines on visually complex documents. | Faysse et al. (2024). "ColPali: Efficient Document Retrieval with Vision Language Models." arXiv:2407.01449 |
| DC004 table queries fail dense retrieval (tR@5=0.400) | **TableRAG** -- SQL-based RAG framework for heterogeneous documents with tables. Decomposes queries, retrieves via schema/cell matching, generates SQL for precise table lookups. | Would specifically address the vocabulary mismatch between natural language queries and symbol-dense table content. | Presented at EMNLP 2025. "TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning." arXiv:2506.10380 |
| EM/F1 underestimate answer quality on technical strings ("Ø1.375" vs "1.375 inches", "Answer: OL" vs "OL") | **ANLS*** -- universal document processing metric using normalized Levenshtein similarity, extended to handle dictionaries, lists, and structured outputs. | Would provide a more forgiving and appropriate metric for technical document QA where minor formatting differences should not count as failures. | Lichtenwalter et al. (2024). "ANLS* -- A Universal Document Processing Metric for Generative Large Language Models." arXiv:2402.03848 |

**Note:** Altumint is a proprietary dataset -- there are no published SOTA baselines. All comparisons are relative across our own runs.

Full citations:
- Faysse, M., Sibille, H., et al. (2024). "ColPali: Efficient Document Retrieval with Vision Language Models." arXiv:2407.01449. [Link](https://arxiv.org/abs/2407.01449)
- TableRAG (2025). "TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning." EMNLP 2025. arXiv:2506.10380. [Link](https://arxiv.org/abs/2506.10380)
- Lichtenwalter, R., et al. (2024). "ANLS* -- A Universal Document Processing Metric for Generative Large Language Models." arXiv:2402.03848. [Link](https://arxiv.org/abs/2402.03848)
