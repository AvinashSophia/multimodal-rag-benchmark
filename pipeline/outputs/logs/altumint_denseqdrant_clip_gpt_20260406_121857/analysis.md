# Benchmark Run Analysis

## Run Summary

| Field | Value |
|---|---|
| Directory | `altumint_denseqdrant_clip_gpt_20260406_121857` |
| Dataset | Altumint (proprietary FLVM technical documentation) |
| Retrieval | text=dense_qdrant (BAAI/bge-large-en-v1.5), image=CLIP (ViT-B-32, fusion_alpha=0.5), top_k=5 |
| Model | GPT-4o (temperature=0.0, max_tokens=512) |
| Samples | 120 (down from 122 in prior run -- 2 samples dropped from regenerated QA set) |
| Changes from prior run | (1) Re-parsed PDFs without --ocr for clean text, (2) Regenerated QA from field technician perspective, (3) Fixed "Answer:" prefix bug in gpt.py, (4) Qdrant collections deleted and rebuilt fresh |

## Metric Scorecard

### Aggregate Metrics -- Current vs Previous Run

| Metric | Current (120 samples) | Previous (122 samples) | Delta | Direction |
|---|---|---|---|---|
| **text_recall@5** | 0.817 | 0.762 | +0.055 | Improved |
| **text_mrr** | 0.630 | 0.605 | +0.025 | Improved |
| **text_ndcg@5** | 0.677 | 0.644 | +0.033 | Improved |
| **image_recall@5** | 0.667 | 0.620 | +0.047 | Improved |
| **image_mrr** | 0.374 | 0.386 | -0.012 | Slight regression |
| **image_ndcg@5** | 0.447 | 0.445 | +0.002 | Flat |
| **exact_match** | **0.500** | **0.238** | **+0.262** | **Major improvement** |
| **f1** | **0.654** | **0.529** | **+0.125** | **Major improvement** |
| **faithfulness** | 0.484 | 0.410 | +0.074 | Improved |
| **attribution_accuracy** | 0.729 | 0.716 | +0.013 | Slight improvement |
| **vqa_accuracy** | 0.167 | 0.079 | +0.088 | Improved (but still unreliable for Altumint) |

### Per-Document-Type Breakdown

| Doc Type | n | EM | F1 | text_recall@5 | Notes |
|---|---|---|---|---|---|
| DC005 wire lengths | 5 | **0.800** | **0.800** | 1.000 | Best performer; numerical lookups work well |
| TM002 config | 38 | **0.684** | **0.769** | 0.763 | IP/password/config questions strongly improved |
| TM001 assembly | 58 | 0.414 | 0.612 | 0.862 | Largest group; visual questions drag down EM |
| DC002 hole location | 3 | 0.333 | 0.368 | 1.000 | Small sample; dimensional annotation difficulty |
| DC001 assembly drawing | 3 | 0.333 | 0.333 | 0.000 | Text retrieval always fails (visual-only doc) |
| DC004 continuity | 10 | 0.300 | 0.567 | 0.800 | Up from 0.200 EM, 0.400 tR@5 in prior run |
| Wiring diagram | 3 | 0.333 | 0.651 | 1.000 | Model reads wiring text but struggles with specifics |

### Question-Type Performance

| Question Type | n | EM Rate |
|---|---|---|
| Config/IP/password/port | 25 | **80%** |
| Non-visual (text-grounded) | 101 | 53% |
| Visual (image-dependent) | 19 | 32% |
| Continuity table | 10 | 20% |

---

## Key Findings

### 1. "Answer:" prefix fix delivered the largest single improvement

EM jumped from 0.238 to 0.500 -- a 110% relative improvement. In the previous run, 51/122 samples (41.8%) had the "Answer:" prefix prepended by the system prompt bug, all scoring EM=0.0. The current run has **zero** samples with the prefix. This confirms the prior analysis that predicted EM would rise to approximately 0.40 after the fix; the actual 0.50 exceeds that estimate because the QA improvements provided additional lift.

### 2. QA regeneration significantly improved question quality

115 out of 120 questions were rewritten from a field technician perspective. The new questions are:
- **More specific**: "What wire gauge is used for BATT+ SW?" instead of "What is the document title?"
- **More actionable**: "What is the first step in assembling the FLVM electronics enclosure?" instead of "What is the document code?"
- **Better grounded**: Questions target specific facts on specific pages rather than generic metadata

This resulted in F1 improving from 0.529 to 0.654 (+0.125), reflecting both better question targeting and more precise ground truth answers.

### 3. Retrieval improved across the board

Text recall@5 rose from 0.762 to 0.817 (+5.5 percentage points). Two factors contribute:
- **Clean text parsing** (no --ocr) removes OCR noise that degraded dense embeddings
- **Better questions** that align more naturally with the vocabulary in the parsed text chunks

DC004 continuity tables saw the most dramatic retrieval improvement: text_recall@5 jumped from 0.400 to 0.800. The cleaner text parsing preserves table structure better, making dense retrieval more effective on these pages.

### 4. CLIP clustering persists as the primary image retrieval bottleneck

TM001_p09 remains the image top-1 result for 43/120 samples (35.8%, down slightly from 38.5%). The average CLIP score spread across top-5 images is only 0.0234 -- effectively random selection. Image MRR actually regressed slightly (0.386 to 0.374). CLIP cannot meaningfully distinguish between technical document pages.

### 5. Faithfulness metric remains unreliable

58/120 samples score faithfulness=0.0, while 54 score 1.0, with only 8 in between. This binary distribution, combined with correct answers frequently scoring 0.0, confirms the RAGAS faithfulness metric is unreliable for this dataset type.

---

## Failure Pattern Analysis

### Pattern 1: Continuity table interpretation errors (10 samples, EM=30%)

The model consistently misreads the dense DC004 continuity check tables. The tables have a complex FROM/TO structure where the correct cell depends on matching both the source device/point and the destination device/point. Common errors:
- **altumint_0001, 0006**: Expected "0 Ohms" for LOAD CB top-to-bottom screw, model answers "OL" (reading the wrong row)
- **altumint_0012, 0014**: Same pattern -- correct page retrieved but wrong cell read from the table
- The "OL" vs "0" distinction is critical (open-loop vs connected), and the model confuses adjacent rows

### Pattern 2: Visual-only questions with no text evidence (19 samples, EM=32%)

Questions requiring reading from photos, screenshots, or diagrams fail when the answer is only in the image:
- **altumint_0003**: "What is the diameter of the mounting holes?" -- GT="5 mm" but model says "I don't know" despite having the correct page retrieved via text. The dimension is only visible in the engineering drawing.
- **altumint_0026**: "What is the visible label on the vent?" -- GT="IP67", model says "I don't know"
- **altumint_0074**: "What is the orientation of the 1' NPT 90-degree fitting?" -- model cannot describe spatial orientation from image

### Pattern 3: Serial number / MAC address hallucination (5 samples)

The model fabricates specific identifiers rather than admitting ignorance:
- **altumint_0029**: GT="HQ23504UFFM", Pred="HQ2240X9QX"
- **altumint_0100**: GT="HQ2530FMDQ9", Pred="SmartSolar HQ2530U4FFM"
- **altumint_0101**: GT="HQ231504UFM", Pred="HQ2246X9"

These appear to be cases where the model has partial context (it recognizes the format) but generates plausible-looking but incorrect serial numbers.

### Pattern 4: "I don't know" responses (5 samples)

Five samples produce explicit refusals despite 4 of 5 having text_recall@5=1.0. The model receives the correct context but cannot extract the answer, typically because the answer requires reading from an image (dimension annotations, labels on photos, MAC addresses on screenshots).

### Pattern 5: Partial match / verbosity mismatch (many samples, F1 > 0 but EM = 0)

Many samples have close but not exact answers:
- "OL" vs "OL Ohms" (unit omission)
- "18 AWG 90C" vs "18 AWG" (over-specification)
- "To the left." vs "The CAT5 ports point to the left." (phrasing difference)
- "Four" vs "Four (2)" (including quantity notation)

These would be captured by ANLS which is still not implemented.

---

## Strengths

1. **Configuration/network questions are near-solved**: 80% EM on IP addresses, passwords, port numbers, and settings. The TM002 config manual is well-parsed and GPT-4o excels at extracting structured data from text.

2. **Wire length lookups work well**: DC005 achieves 0.800 EM. The table markdown parsing is clean enough for the model to find the correct row and extract the value.

3. **Text retrieval is robust**: 82% recall@5 across the full dataset, with MRR=0.630 indicating the correct page is frequently at rank 1 or 2.

4. **Attribution is strong**: 72.9% of model-cited sources match the ground truth relevant source. The model is generally honest about where it finds information.

5. **No prefix contamination**: The "Answer:" prefix bug is fully resolved. Zero samples affected.

---

## Bottlenecks and Weaknesses

### 1. CLIP image retrieval is near-random (CRITICAL)

With a score spread of 0.0234 and one page dominating 36% of all queries, CLIP ViT-B-32 provides almost no useful signal for document page retrieval. This is the single largest drag on multimodal performance.

### 2. Table cell disambiguation (HIGH)

DC004 continuity tables contain dense grids where the correct answer depends on precise row-column intersection. The model frequently reads the wrong cell, even when the correct page is retrieved.

### 3. Missing ANLS metric (MEDIUM)

Many near-correct answers score EM=0.0 due to minor string differences ("OL" vs "OL Ohms", "0" vs "0 Ohms", "positive distribution block" vs "+ distribution block"). ANLS would provide a more accurate picture of answer quality and likely show scores of 0.70-0.75 rather than 0.50 EM.

### 4. Visual-only evidence extraction (MEDIUM)

When answers exist only in images (dimensions, labels, orientations), the model frequently fails or refuses. This affects ~16% of samples.

### 5. Serial number hallucination (LOW but concerning)

The model generates plausible-looking but wrong serial numbers rather than saying "I don't know". This is a faithfulness/safety concern for production use.

---

## Recommendations (Prioritized)

### 1. Implement ANLS metric (highest ROI, low effort)

The gap between EM (0.500) and F1 (0.654) confirms many near-miss answers. ANLS with a threshold of 0.5 would properly credit answers like "OL" for ground truth "OL Ohms" and "0" for "0 Ohms". This is an evaluation improvement, not a pipeline change, and is already tracked as a pending task. Use the `anls` PyPI package or implement directly with normalized Levenshtein distance.

### 2. Replace CLIP with ColPali for image retrieval (highest pipeline impact)

ColPali uses late-interaction matching on PaliGemma vision patches, producing multi-vector embeddings that understand document layout. It would replace the near-random CLIP scoring with layout-aware retrieval that can distinguish engineering drawings from assembly photos from configuration screenshots. This is already tracked as a pending task.

### 3. Add hybrid (BM25 + dense) retrieval for DC004 continuity tables

DC004 tables contain specific symbols (OL, 0, B+, B-) and device names (LOAD CB, PV CB, Solar Controller) that BM25 would match exactly. The hybrid_elastic_qdrant retriever is already implemented -- running with it on this dataset would likely push DC004 text_recall@5 above 0.90.

### 4. Improve table cell extraction in prompts

For continuity table questions, the model needs guidance to identify the correct FROM-TO pair before reading the expected value. A targeted prompt addition like "When reading tables, first identify the exact row and column headers that match the question before extracting the cell value" could reduce misreads.

### 5. Add a re-ranker for text retrieval

Text MRR is 0.630, meaning the correct page is often at rank 2-5 rather than rank 1. A cross-encoder re-ranker (e.g., bge-reranker-v2-m3) applied to the top-20 dense results could push MRR above 0.80 and improve downstream answer quality. This is already tracked as a pending task.

---

## Anomalies / Flags

1. **Sample count decreased**: 120 vs 122 in the prior run. Two samples were dropped during QA regeneration. Not a data integrity issue, but worth noting for trend tracking.

2. **No per-sample image retrieval metrics**: The `metrics` dict per sample does not contain `image_mrr` or `image_recall@5` keys. Image retrieval metrics are only computed at the aggregate level. This makes it impossible to correlate image retrieval quality with answer quality at the sample level.

3. **DC001 text recall is 0.000 for all 3 samples**: DC001 is a single-page assembly drawing with minimal text. All 3 questions about it have `relevant_sources` pointing to DC001 but the dense retriever never surfaces it. These questions can only be answered from the image, confirming the visual-only bottleneck.

4. **Duplicate relevant_sources**: Several samples list the same source twice in `relevant_sources` (e.g., `altumint_0002` has `dc001...p01` twice). This is likely a QA generation artifact, not a pipeline bug.

5. **Faithfulness binary distribution**: 54 samples at 1.0, 58 at 0.0, only 8 in between. The RAGAS faithfulness metric continues to be unreliable for technical documentation with specialized vocabulary. Do not use for decision-making.

6. **VQA accuracy = 0.167**: This metric uses a multi-annotator protocol (min of 3 annotators) that is meaningless for Altumint's single-annotation ground truth. Ignore this metric.

---

## Comparative Analysis: Current vs Previous (Same Dataset)

| Aspect | Previous (105313) | Current (121857) | Assessment |
|---|---|---|---|
| QA quality | Generic metadata questions | Field technician perspective | Major upgrade |
| Text parsing | Without --ocr (same) | Without --ocr (same) | Same |
| "Answer:" prefix | 51/122 affected (41.8%) | 0/120 affected | Fixed |
| Qdrant index | Potentially stale | Freshly rebuilt | Clean |
| EM | 0.238 | 0.500 | +110% relative |
| F1 | 0.529 | 0.654 | +24% relative |
| text_recall@5 | 0.762 | 0.817 | +7% relative |
| DC004 tR@5 | 0.400 | 0.800 | +100% relative |
| CLIP clustering | 38.5% dominated by TM001_p09 | 35.8% dominated by TM001_p09 | Marginal change |

**Overall assessment**: This run represents a substantial improvement over the previous baseline. The three changes (prefix fix, QA regeneration, fresh index) collectively raised EM by 26.2 percentage points. The prefix fix alone accounts for roughly half of that gain, with improved question quality and fresh indexing accounting for the rest. The remaining gap to higher performance is primarily in (a) CLIP replacement, (b) table cell disambiguation, and (c) ANLS implementation for fairer scoring.

---

## SOTA Research Recommendations

| Weakness Observed | Recommended Approach | Expected Impact | Citation |
|---|---|---|---|
| CLIP near-random on document pages (score spread 0.023) | ColPali: late-interaction visual retrieval using PaliGemma vision patches | Replace CLIP entirely; ColPali natively understands document layout, tables, and text in images | Faysse et al. (2024) |
| Continuity table cell misreads (DC004 EM=0.30) | Chain-of-Table: structured reasoning chains for table understanding | Decomposes table QA into sequential operations (filtering, sorting, aggregation) before answering | Wang et al. (2024) |
| EM penalizes near-correct answers ("OL" vs "OL Ohms") | ANLS* metric: universal document processing metric with Levenshtein-based soft matching | Would raise effective accuracy from 0.50 (EM) to estimated 0.70-0.75 | Van Landeghem et al. (2024) |

**Full citations:**
- Faysse, M., Sibille, H., et al. (2024). "ColPali: Efficient Document Retrieval with Vision Language Models." ICLR 2025. [arXiv:2407.01449](https://arxiv.org/abs/2407.01449)
- Wang, Z., Dong, H., Jia, R., Li, J., et al. (2024). "Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding." ICLR 2024. [arXiv:2401.04398](https://arxiv.org/abs/2401.04398)
- Van Landeghem, J., Tito, R., et al. (2024). "ANLS* -- A Universal Document Processing Metric for Generative Large Language Models." [arXiv:2402.03848](https://arxiv.org/abs/2402.03848)

**Note:** Altumint is a proprietary dataset -- no external SOTA baselines exist. All performance comparisons are relative to our own prior runs.
