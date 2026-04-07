# Benchmark Run Analysis

## Run Summary
- **Directory:** `altumint_denseqdrant_clip_gpt_20260406_083337`
- **Dataset:** Altumint (proprietary FLVM technical documentation, 7 PDFs, 36 pages)
- **Retrieval:** text=dense_qdrant (BGE-large-en-v1.5, top_k=5), image=CLIP (ViT-B-32, top_k=5, fusion_alpha=0.5)
- **Model:** GPT-4o (temperature=0.0, max_tokens=512)
- **Samples analyzed:** 5 of 123 total (max_samples=5)
- **QA mix:** 4 text queries, 1 visual query (altumint_0002 has a diagram crop as query image)
- **Date:** 2026-04-06

---

## Metric Scorecard

| Metric | Score | Prior Run (Apr 2) | Delta |
|---|---|---|---|
| **text_recall@5** | 0.000 | N/A (not computed) | -- |
| **text_mrr** | 0.000 | N/A (not computed) | -- |
| **text_ndcg@5** | 0.000 | N/A (not computed) | -- |
| **image_recall@5** | 0.000 | 1.000 | -1.000 |
| **image_mrr** | 0.000 | 1.000 | -1.000 |
| **image_ndcg@5** | 0.000 | 1.000 | -1.000 |
| **exact_match** | 0.000 | 0.600 | -0.600 |
| **f1** | 0.156 | 0.700 | -0.544 |
| **faithfulness** | 0.000 | 0.400 | -0.400 |
| **attribution_accuracy** | 0.533 | 0.733 | -0.200 |
| **vqa_accuracy** | 0.000 | 0.200 | -0.200 |

**Verdict: Catastrophic regression across every metric.** All retrieval metrics are 0.0. EM dropped from 0.6 to 0.0. The root cause is an ID format mismatch between the text retriever index and the current dataset, compounded by a QA pair regeneration that changed the questions.

---

## CRITICAL FINDING: ID Format Mismatch (Text Retriever)

This is the single most important finding and explains the majority of the metric collapse.

### What happened

The Altumint parsed page JSONs use **long-form page IDs**:
```
altumint_dc001_01_flvm_electronics_enclosure_assembly_drawing_p01
altumint_tm001_02_flashing_light_video_monitor_assembly_instructions_p05
```

The QA ground truth (`relevant_text_ids`, `relevant_image_ids`) uses these same long-form IDs.

However, the **text retriever (Qdrant dense)** returns **short-form IDs** from a stale index:
```
DC001_p01, TM001_p05, DC002_p01, wiring_p01
```

This means `relevant_text_ids` (long format) will **never match** `retrieved_text_ids` (short format), causing text_recall@5 = 0.0 across all samples.

### Evidence per sample

| Sample | Retrieved text_ids (short) | Ground truth relevant_text_ids (long) | Match? |
|---|---|---|---|
| 0000 | DC001_p01, DC002_p01, DC005_p01, TM001_p01, TM002_p01 | altumint_dc001_01_...assembly_drawing_p01 | NO |
| 0001 | TM001_p05, wiring_p01, TM001_p06, DC004_p04, DC004_p03 | altumint_dc001_01_...assembly_drawing_p01 | NO |
| 0002 | TM001_p10, TM001_p12, DC002_p01, TM001_p08, wiring_p01 | altumint_dc001_01_...assembly_drawing_p01 | NO |
| 0003 | TM001_p15, TM001_p02, DC002_p01, DC005_p02, DC004_p05 | altumint_dc002_01_...hole_location_drawing_p01 | NO |
| 0004 | DC004_p01, DC002_p01, DC001_p01, TM001_p13, TM001_p07 | altumint_dc002_01_...hole_location_drawing_p01 | NO |

### Root cause

The Qdrant text collection `dense_text_altumint` was indexed during the prior run (April 2) using **short-form IDs** (e.g., `DC001_p01`). The Altumint dataset was subsequently re-parsed with Docling, which generated **long-form page IDs** (e.g., `altumint_dc001_01_flvm_electronics_enclosure_assembly_drawing_p01`). The skip-if-indexed optimization likely prevented the text index from being rebuilt, so the stale short-form IDs persisted.

### Impact

- All text retrieval metrics: **artificially 0.0** (format mismatch, not actual retrieval failure)
- The text retriever IS retrieving semantically relevant pages -- e.g., sample 0000 retrieves `DC001_p01` which IS the assembly drawing page -- but the ID string doesn't match the ground truth format.

### Fix required

**Re-index the Qdrant text collection** with the current long-form page IDs. Delete the existing `dense_text_altumint` collection and re-run the pipeline, or disable skip-if-indexed for this collection.

---

## Image Retrieval: Also Zero, But Different Cause

Unlike text retrieval (ID format mismatch), the **image retriever uses long-form IDs** that match the ground truth format. Yet image_recall@5 is still 0.0.

### Why image recall is zero

The ground truth `relevant_image_ids` for each sample points to the source page. The image retriever does not retrieve that page in the top-5 for most samples:

| Sample | Query type | Ground truth image ID | Retrieved in top-5? | Rank if found |
|---|---|---|---|---|
| 0000 | text | (no images in query, no image eval) | N/A | N/A |
| 0001 | text | (no images in query, no image eval) | N/A | N/A |
| 0002 | visual | altumint_dc001_01_...assembly_drawing_p01 | YES (rank 3) | 3 |
| 0003 | text | (no images in query, no image eval) | N/A | N/A |
| 0004 | text | (no images in query, no image eval) | N/A | N/A |

Wait -- sample 0002 does have the correct image at rank 3. But image_recall@5 = 0.0 for that sample. Let me check why.

Looking at sample 0002's ground truth more carefully:
- `relevant_sources` includes: `altumint_dc001_01_flvm_electronics_enclosure_assembly_drawing_p01` and `altumint_dc001_01_flvm_electronics_enclosure_assembly_drawing_p01_fig_00`
- But `relevant_image_ids` (line 116 of run_benchmark.py) comes from `sample.image_ids`, which is the **query image IDs**, not the ground truth page IDs.

For visual queries, `sample.image_ids` = `[img_path.stem]` (the cropped figure filename stem), which is something like `altumint_dc001_01_flvm_electronics_enclosure_assembly_drawing_p01_fig_00`. This is a **figure crop ID**, not the **page screenshot ID** indexed by CLIP. The retriever returns page-level image IDs, but the ground truth `relevant_image_ids` is a figure-level ID. These will never match.

For text queries (samples 0, 1, 3, 4), `sample.images = []`, so `relevant_image_ids` is empty and image retrieval eval is skipped entirely. This is correct behavior but means only 1 of 5 samples has image eval, and that one sample has a figure-vs-page ID mismatch.

### Second ID mismatch: figure crop IDs vs page screenshot IDs

This is a distinct bug from the text retriever issue:
- Ground truth `relevant_image_ids` for visual queries: `altumint_dc001_..._p01_fig_00` (figure crop)
- Retrieved image IDs: `altumint_dc001_..._p01` (page screenshot)
- These never match because the `_fig_00` suffix is appended to figure crops but the CLIP index stores page-level screenshots.

---

## Per-Sample Analysis

### Sample 0000 (text query, factual)
- **Q:** "What type of battery is used in the FLVM Electronics Enclosure?"
- **GT:** "Tycon 10Ah Aluminium LiFePO4 Battery"
- **Pred:** "Tycon TSBAT12-10L-WT Lithium Battery"
- **F1:** 0.444 | **EM:** 0.0
- **Analysis:** The model found a correct but differently-formatted answer. TM002_p01 mentions "Tycon TSBAT12-10L-WT Lithium Battery" which is the model number, not the descriptive name. The ground truth expects the descriptive label from DC001 (assembly drawing). Text retriever did NOT retrieve DC001_p01-equivalent in the right format. This is a retrieval-generation gap: the model answered from what it found (TM002_p01) rather than the intended source.

### Sample 0001 (text query, numerical)
- **Q:** "How many WAGO Distribution Blocks are included in the assembly?"
- **GT:** "4"
- **Pred:** "Two (2)"
- **F1:** 0.0 | **EM:** 0.0
- **Analysis:** The model cited TM001_p05 which says "Install two (2) 5-position WAGOs into one carrier and a single (1) 5-position WAGO into the second carrier." The text describes 3 individual WAGO connectors across 2 carriers, but the ground truth expects 4. The model misread the assembly instructions. The wiring diagram (wiring_p01) shows more WAGOs but the model relied on the assembly text. This is both a generation error and potentially an ambiguous ground truth.

### Sample 0002 (visual query, visual)
- **Q:** "Where is the POE Injector located in the electronics enclosure diagram?"
- **GT:** "The POE Injector is located beneath the modem in the electronics enclosure."
- **Pred:** "Answer: Beneath Modem"
- **F1:** 0.333 | **EM:** 0.0
- **Attribution:** 0.667 (model cited the correct source page)
- **Analysis:** The model gave a semantically correct but truncated answer. "Beneath Modem" captures the key fact but lacks the surrounding sentence structure. The "Answer:" prefix is a prompt formatting artifact. This is the **brevity prompt issue** documented in prior runs. With ANLS or a more forgiving metric, this would score much higher.

### Sample 0003 (text query, numerical)
- **Q:** "What is the diameter of the vent hole?"
- **GT:** "3.000"
- **Pred:** "Answer: 0.750"
- **F1:** 0.0 | **EM:** 0.0
- **Attribution:** 1.0 (correctly cited DC002_p01)
- **Analysis:** The model cited the correct source (hole location drawing) but read the wrong dimension. DC002_p01 has multiple hole dimensions (antenna hole = 0.750, vent hole = 3.000). The text chunk for DC002_p01 is nearly empty ("FLVM Electronics Enclosure - Hole Location Drawing.pdf") -- the dimensions are only visible in the IMAGE of the drawing. The model had to read the image but picked the wrong dimension. This is a **visual reasoning failure** -- the model confused the antenna hole (0.750) with the vent hole (3.000).

### Sample 0004 (text query, factual)
- **Q:** "Who checked the drawing?"
- **GT:** "Steven Hartig"
- **Pred:** "Answer: A"
- **F1:** 0.0 | **EM:** 0.0
- **Attribution:** 1.0 (correctly cited DC002_p01)
- **Analysis:** "Answer: A" is nonsensical. The ground truth "Steven Hartig" appears in the drawing's title block, which is only readable from the image. The text chunk for DC002 contains almost no text. The model likely tried to read the image but failed to locate the "CHECKED BY" field in the drawing's title block. This is a severe **visual reading failure** on an engineering drawing.

---

## Key Findings

1. **CRITICAL: Stale text retriever index with short-form IDs.** The Qdrant `dense_text_altumint` collection stores IDs from a prior dataset version (e.g., `DC001_p01`). The current dataset and ground truth use long-form IDs (e.g., `altumint_dc001_01_flvm_electronics_enclosure_assembly_drawing_p01`). This causes text_recall@5 = 0.0 as a measurement artifact, not an actual retrieval failure.

2. **CRITICAL: Figure crop ID vs page screenshot ID mismatch for visual queries.** The evaluator uses the query image's filename stem as `relevant_image_ids` (e.g., `..._p01_fig_00`), but the CLIP index stores page-level screenshot IDs (e.g., `..._p01`). These never match, causing image_recall@5 = 0.0 for visual queries.

3. **EM = 0.0 across all 5 samples.** Even where the model gives a semantically correct answer (sample 0002: "Beneath Modem" vs "The POE Injector is located beneath the modem..."), exact match fails due to brevity/formatting differences. ANLS metric is critically needed.

4. **Engineering drawing questions fail on visual reading.** Samples 0003 and 0004 require reading dimensions and title block fields from DC002's hole location drawing. GPT-4o misread the dimension (0.750 vs 3.000) and completely failed to extract the "CHECKED BY" field (answered "A" instead of "Steven Hartig").

5. **New QA pairs differ from prior run.** The April 2 run had different questions (e.g., "What component is located beneath the modem?"). The QA file was regenerated with the text/visual query type split (72 text, 51 visual). Direct metric comparison is confounded by both the ID bug and different questions.

---

## Failure Pattern Analysis

### Pattern 1: ID format mismatch (systemic, affects all samples)
- **Text retrieval:** Short IDs in index vs long IDs in ground truth. 5/5 samples affected.
- **Image retrieval:** Figure crop IDs vs page screenshot IDs. Affects all visual query samples.
- **Impact:** All retrieval metrics read as 0.0 even when the correct page IS retrieved.

### Pattern 2: Engineering drawing visual reading failures (samples 0003, 0004)
- Both questions target DC002 (hole location drawing) -- a technical engineering diagram.
- DC002's text chunk is nearly empty (just the document title). All information is in the image.
- GPT-4o struggles to: (a) distinguish between multiple annotated dimensions, (b) read title block metadata fields.
- This matches the known CLIP weakness on engineering drawings but is primarily a **generation** problem -- the image is delivered to the model but the model cannot read it accurately.

### Pattern 3: Brevity/formatting artifacts (samples 0002, 0003, 0004)
- All three predictions begin with "Answer:" prefix -- a prompt formatting artifact.
- Answers are over-compressed: "Beneath Modem" instead of a full sentence.
- The brevity prompt fix from the prior run analysis was flagged but may not have been applied.

### Pattern 4: Ambiguous ground truth (sample 0001)
- "4" WAGO distribution blocks is debatable based on the text content. The assembly instructions describe 3 WAGO connectors across 2 carriers. The answer may come from counting differently on the wiring diagram.

---

## Strengths

1. **Attribution accuracy is non-trivial (0.533).** Despite all the ID issues, the model correctly cites sources in 3/5 samples (0002, 0003, 0004 all score 1.0). The model's source attribution is working even when other metrics are broken.

2. **Text retriever returns semantically relevant pages.** Despite the ID format bug, examining the actual text chunks shows the retriever is pulling relevant content (e.g., TM002_p01 for battery question, TM001_p05 for WAGO question). The retrieval quality is masked by the measurement bug.

3. **Visual query pipeline is functional.** Sample 0002 demonstrates the full visual query flow: diagram crop -> CLIP query -> retrieve relevant pages -> model answers from image. The answer ("Beneath Modem") is semantically correct.

---

## Bottlenecks & Weaknesses (Prioritized)

1. **Stale Qdrant text index (BLOCKING)** -- All text retrieval metrics are invalid. Must re-index before any further analysis is meaningful.

2. **Figure-vs-page ID mismatch in evaluator (BLOCKING)** -- Visual query image recall will always be 0.0 until the evaluator maps figure crop IDs to their parent page IDs.

3. **GPT-4o visual reading on engineering drawings** -- The model cannot reliably read dimensional annotations or title block fields from technical drawings. This is a fundamental VLM limitation.

4. **Brevity prompt still active** -- The "Answer:" prefix and over-compressed responses depress EM and F1.

5. **No ANLS metric** -- For a dataset with answers like "3.000", "Steven Hartig", "180mm", ANLS is far more appropriate than EM/F1.

6. **Only 5 of 123 samples evaluated** -- With the ID bugs fixed, a full 123-sample run is needed for reliable metrics.

---

## Recommendations (Prioritized)

### P0: Fix ID consistency bugs before next run

1. **Delete and re-index `dense_text_altumint` Qdrant collection.** The stale short-form IDs must be replaced with the current long-form IDs. Remove the collection directory at `pipeline/outputs/qdrant_store/collection/dense_text_altumint/` and re-run.

2. **Fix figure-crop-to-page ID mapping in evaluator.** For visual queries, `relevant_image_ids` should be the **page ID** (e.g., `..._p01`), not the **figure crop ID** (e.g., `..._p01_fig_00`). Either:
   - (a) Strip the `_fig_XX` suffix when computing `relevant_image_ids`, or
   - (b) Set `relevant_image_ids` from `metadata["relevant_page_ids"]` instead of `sample.image_ids`.

### P1: Fix generation quality

3. **Remove "Answer:" prefix from prompt.** The model is prepending "Answer:" to every response, which hurts EM.

4. **Add ANLS metric.** This is the standard metric for document VQA and would correctly credit "Beneath Modem" as partially matching the full ground truth sentence.

5. **Fix brevity prompt.** Ensure the system prompt does not over-compress responses for questions expecting full sentences.

### P2: Improve engineering drawing understanding

6. **Evaluate ColPali as image retriever.** CLIP (ViT-B-32) is a general-purpose vision encoder. ColPali produces document-aware multi-vector embeddings that understand layout, text regions, and technical annotations. This would improve retrieval on DC001, DC002, and DC004.

7. **Consider a larger/better VLM for generation.** GPT-4o struggles with fine-grained technical drawing reading. GPT-4 Turbo with vision or Claude with vision may perform better on dimensional annotations.

### P3: Expand evaluation

8. **Run full 123-sample evaluation** once P0 bugs are fixed. The current 5-sample set is too small and skewed toward DC001/DC002 questions.

9. **Add per-question-type metric breakdowns.** With 5 question types (factual, numerical, visual, procedural, cross_doc), per-type metrics would reveal which categories need the most work.

---

## Anomalies / Flags

1. **Data integrity: stale Qdrant index (CRITICAL).** The `dense_text_altumint` collection contains short-form IDs from a prior dataset version. This silently corrupts all text retrieval metrics. The skip-if-indexed optimization prevented the index from being rebuilt when the dataset was re-parsed.

2. **Data integrity: figure-vs-page ID mismatch (CRITICAL).** The evaluator conflates figure crop IDs with page screenshot IDs for visual queries. This is a systematic bug that affects all visual query samples.

3. **Faithfulness = 0.0 across all samples.** This matches the RAGAS anomaly observed in all prior runs (DocVQA, Altumint Apr 2, HotpotQA). The RAGAS faithfulness metric is unreliable and should not be used for decision-making until debugged.

4. **VQA accuracy = 0.0.** Expected for Altumint -- the standard VQA multi-annotator protocol is meaningless for single-annotation proprietary datasets. Do not use as a quality indicator.

5. **Prior run comparison is confounded.** The April 2 run used different QA pairs (different questions, no text/visual split). Metric deltas in the scorecard above reflect BOTH the ID bugs AND the question changes. A clean comparison requires re-running the April 2 questions with the current index, or vice versa.

---

## Comparative Context (Same-Dataset Runs)

| Metric | Apr 2 (5 samples, old QA) | Apr 6 (5 samples, new QA) | Notes |
|---|---|---|---|
| text_recall@5 | N/A | 0.000 | Apr 2 had no text eval; Apr 6 has text eval but broken by ID mismatch |
| image_recall@5 | 1.000 | 0.000 | Apr 2 had short image IDs that matched; Apr 6 has figure-vs-page ID mismatch |
| exact_match | 0.600 | 0.000 | Different questions + ID bugs make comparison unreliable |
| f1 | 0.700 | 0.156 | Same caveat |
| attribution | 0.733 | 0.533 | Most reliable comparison -- attributions use model-cited sources |

**Key takeaway:** The April 2 run had a simpler ID scheme where everything matched. The dataset re-parse introduced long-form IDs that the text index was not rebuilt for. The April 6 run is not a valid comparison point until the ID bugs are fixed.

---

## SOTA Research Recommendations

| Weakness Observed | Recommended Approach | Why It Helps | Citation |
|---|---|---|---|
| CLIP fails on engineering drawings (DC001, DC002) | **ColPali** -- visual document retrieval using VLM-based multi-vector embeddings | ColPali produces layout-aware page embeddings that understand text regions, tables, and technical annotations in documents, unlike CLIP which treats pages as natural images | Faysse et al. (2024) |
| GPT-4o cannot read dimensions/title blocks from technical drawings | **Vision-guided chunking + specialized OCR preprocessing** | Pre-extracting text from engineering drawings via OCR and injecting it into the text corpus would give the model structured text instead of relying on vision alone | Mumuni & Mumuni (2025) |
| Dense embeddings miss domain-specific technical terms | **Technical-Embeddings framework** for domain-adapted retrieval | Fine-tuned embeddings on technical documentation improve semantic match for domain jargon (WAGO, LiFePO4, NPT fittings) | arXiv:2509.04139 (2025) |

### Full citations

- Faysse, M., Sibille, H., Wu, T., Music, B., et al. (2024). "ColPali: Efficient Document Retrieval with Vision Language Models." arXiv:2407.01449. [https://arxiv.org/abs/2407.01449](https://arxiv.org/abs/2407.01449)
- Mumuni, A. & Mumuni, F. (2025). "Vision-Guided Chunking Is All You Need: Enhancing RAG with Multimodal Document Understanding." arXiv:2506.16035. [https://arxiv.org/abs/2506.16035](https://arxiv.org/abs/2506.16035)
- arXiv:2509.04139 (2025). "Enhancing Technical Documents Retrieval for RAG." [https://arxiv.org/abs/2509.04139](https://arxiv.org/abs/2509.04139)

**Note:** Altumint is proprietary -- there are no published external SOTA baselines. All comparisons are relative to our own prior runs.

---

*Analysis generated 2026-04-03 by benchmark-results-analyzer.*
