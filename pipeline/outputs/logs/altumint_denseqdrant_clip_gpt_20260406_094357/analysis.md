## Benchmark Run Analysis

### Run Summary
- **Directory:** `altumint_denseqdrant_clip_gpt_20260406_094357`
- **Dataset:** Altumint (proprietary FLVM technical documentation, 7 PDFs, 36 pages)
- **Retrieval:** text=dense_qdrant (BAAI/bge-large-en-v1.5, top_k=5), image=CLIP (ViT-B-32, top_k=5, fusion_alpha=0.5)
- **Model:** GPT-4o (temperature=0.0, max_tokens=512)
- **Samples analyzed:** 5
- **Previous run (comparison):** `altumint_denseqdrant_clip_gpt_20260406_083337` (same 5 QA pairs, same config, pre-bugfix)
- **Bugs fixed between runs:** (1) Stale Qdrant `dense_text_altumint` collection deleted and rebuilt with long-form page IDs; (2) Visual query `image_ids` now use `source_page_id` instead of figure filename stem.

---

### Metric Scorecard

| Metric | Previous (083337) | Current (094357) | Delta |
|---|---|---|---|
| **text_recall@5** | 0.000 | **1.000** | +1.000 |
| **text_mrr** | 0.000 | **0.733** | +0.733 |
| **text_ndcg@5** | 0.000 | **0.800** | +0.800 |
| **image_recall@5** | 0.000 | **1.000** | +1.000 |
| **image_mrr** | 0.000 | **0.333** | +0.333 |
| **image_ndcg@5** | 0.000 | **0.500** | +0.500 |
| **exact_match** | 0.000 | **0.200** | +0.200 |
| **f1** | 0.156 | **0.456** | +0.301 |
| **faithfulness** | 0.000 | **0.400** | +0.400 |
| **attribution_accuracy** | 0.533 | **0.800** | +0.267 |
| **vqa_accuracy** | 0.000 | **0.067** | +0.067 |

Every single metric improved. The ID fixes were the critical unblock.

---

### Detailed Per-Sample Breakdown

| Sample | Question | text_mrr | text_recall@5 | image_mrr | image_recall@5 | EM | F1 | Faithful | Attrib |
|---|---|---|---|---|---|---|---|---|---|
| 0000 | Battery type in FLVM? | 1.0 | 1.0 | -- | -- | **1.0** | **1.0** | 1.0 | 1.0 |
| 0001 | How many WAGO blocks? | 0.333 | 1.0 | -- | -- | 0.0 | 0.0 | 0.0 | 0.0 |
| 0002 | POE Injector location? | 0.333 | 1.0 | 0.333 | 1.0 | 0.0 | 0.615 | 1.0 | 1.0 |
| 0003 | Vent hole diameter? | 1.0 | 1.0 | -- | -- | 0.0 | 0.667 | 0.0 | 1.0 |
| 0004 | Who checked the drawing? | 1.0 | 1.0 | -- | -- | 0.0 | 0.0 | 0.0 | 1.0 |

Notes: image retrieval metrics only appear for sample 0002 (the visual/diagram question). Samples 0000, 0001, 0003, 0004 appear to be text-only queries with no image ground truth IDs.

---

### Key Findings

**1. ID Consistency Fixes Fully Unblocked Retrieval Metrics**

The previous run had text_recall@5=0.0 and image_recall@5=0.0 because retrieved IDs (`DC001_p01`) could never match ground truth IDs (`altumint_dc001_01_...`). After rebuilding the Qdrant index with long-form IDs and fixing image ID mapping, **text_recall@5 and image_recall@5 are both perfect at 1.0**. The correct page appears in the top-5 for every single sample across both modalities.

**2. Text Retrieval Ranking is Strong (MRR=0.733, nDCG=0.800)**

3 of 5 samples have the correct text chunk ranked #1 (MRR=1.0). The two samples with MRR=0.333 (0001, 0002) have the correct page at rank 3 -- still retrieved, just not top-ranked. This is a solid baseline for a 36-page corpus with dense retrieval.

**3. Image Retrieval Ranking is Weaker (MRR=0.333, nDCG=0.500)**

Only 1 sample (0002) has image ground truth. CLIP placed the correct image (DC001 assembly drawing) at rank 3 out of 5, behind DC002 and the wiring diagram. This is the expected CLIP failure mode on engineering drawings: all technical line drawings cluster together in CLIP's embedding space. The scores are tightly bunched (0.515-0.550), confirming poor discriminability.

**4. Answer Quality: 1 Perfect, 2 Partial, 2 Failures**

- **altumint_0000 (EM=1.0, F1=1.0):** Perfect. "Tycon 10Ah Aluminium LiFePO4 Battery" matched exactly. The correct text chunk (DC001_p01) was ranked #1 with score 0.742.
- **altumint_0002 (EM=0.0, F1=0.615):** Partial match. Predicted "POE Injector (Beneath Modem)" vs ground truth "The POE Injector is located beneath the modem in the electronics enclosure." Correct semantically but EM fails on phrasing difference. The model extracted verbatim text rather than forming a full sentence.
- **altumint_0003 (EM=0.0, F1=0.667):** Partial match. Predicted "Answer: 3.000" vs ground truth "3.000". The F1 is non-zero because "3.000" is a substring. The "Answer:" prefix is the brevity prompt artifact -- it inflates the token count and breaks exact match.
- **altumint_0001 (EM=0.0, F1=0.0):** Complete failure. Predicted "3" (WAGO blocks) vs ground truth "4". The model read from TM001_p05 which describes installing WAGOs into carriers (2 into one carrier + 1 into another = 3 WAGOs), but the ground truth comes from DC001 which says "QTY 4" distribution blocks. The model used the wrong source. DC001 was retrieved at rank 3, but the model preferred the more procedurally detailed TM001_p05 at rank 1.
- **altumint_0004 (EM=0.0, F1=0.0):** Complete failure. Predicted "A" vs ground truth "Steven Hartig". The text chunk contains "APPROVED CHECKED DRAWN Steven Hartig" and also "SIZE A". GPT-4o appears to have misread the title block layout, confusing the SIZE field value "A" with the CHECKED field. This is a known limitation: OCR-extracted text from engineering drawings loses spatial layout, making title block field associations ambiguous.

**5. Faithfulness Metric Remains Unreliable**

Faithfulness scores 1.0 for samples 0000 and 0002 (both correct) but 0.0 for 0003 (which gave the correct answer "3.000" from the correct source). The RAGAS faithfulness metric appears to be inconsistently evaluating factual consistency -- the same anomaly observed in the Apr 2 run and in DocVQA.

**6. Attribution Accuracy is Strong (0.800)**

4 of 5 samples have perfect attribution (1.0). The model correctly cites the source page it used. The only failure is sample 0001 where the model cited TM001_p05 but the ground truth relevant source is DC001_p01. This is actually a source selection error, not an attribution error per se -- the model faithfully cited what it used, but it used the wrong source.

---

### Comparison to Previous Altumint Runs

| Run | Date | text_recall@5 | text_mrr | image_recall@5 | EM | F1 | Attribution |
|---|---|---|---|---|---|---|---|
| 20260402_154718 | Apr 2 | N/A (not computed) | N/A | 1.0 | 0.6 | 0.7 | 0.733 |
| 20260406_083337 | Apr 6 (pre-fix) | 0.0 | 0.0 | 0.0 | 0.0 | 0.156 | 0.533 |
| **20260406_094357** | **Apr 6 (post-fix)** | **1.0** | **0.733** | **1.0** | **0.200** | **0.456** | **0.800** |

**Important caveats on cross-run comparison:**
- The Apr 2 run used a different set of 5 QA pairs, so EM/F1/attribution are not directly comparable.
- The Apr 2 run did not compute text retrieval metrics (no supporting_facts populated).
- Image recall was 1.0 on Apr 2 (different QA pairs) and is 1.0 now (only 1 visual query in this batch).
- The current run's EM=0.200 and F1=0.456 reflect the harder question mix (WAGO count, title block parsing) rather than regression.

---

### Failure Pattern Analysis

**Pattern 1: "Answer:" prefix artifact (2/5 samples)**
Samples 0003 and 0004 prepend "Answer:" to their responses. Sample 0001 also prepends it. This is the brevity prompt issue previously documented. While it only breaks EM for sample 0003 (where the actual answer is correct), it adds noise to all F1 calculations.

**Pattern 2: Wrong source selection despite correct retrieval (1/5 samples)**
Sample 0001 retrieves DC001 (correct source, "QTY 4") at rank 3 but GPT-4o uses TM001_p05 (rank 1, which describes 3 individual WAGOs in carriers). The model prefers the more detailed/relevant-seeming text chunk over the authoritative summary. This is a generation-side source selection problem, not a retrieval problem.

**Pattern 3: Title block field confusion (1/5 samples)**
Sample 0004 fails because OCR-extracted text from DC002 loses the spatial relationship between "CHECKED" and "Steven Hartig" in the title block. The model sees "SIZE A" and "CHECKED DRAWN Steven Hartig" as flat text and guesses "A". This is a fundamental limitation of text-only retrieval for engineering drawing metadata.

**Pattern 4: CLIP clustering on technical drawings (observed in image scores)**
For sample 0002, the top-5 image scores range from 0.507 to 0.550 -- a spread of only 0.043. DC002, wiring diagram, and DC001 all score within 0.035 of each other. CLIP cannot discriminate between different technical line drawings.

---

### Strengths

1. **Perfect retrieval recall** -- text_recall@5=1.0 and image_recall@5=1.0. The correct page always appears in the top-5 for both modalities.
2. **Strong text ranking** -- MRR=0.733 means the correct page is usually at or near rank 1.
3. **Strong attribution** -- 4/5 samples have the model correctly citing its source.
4. **ID consistency validated** -- The two bug fixes fully resolved the collapsed metrics from the previous run. Long-form IDs now match across Qdrant index, CLIP index, ground truth, and evaluation.
5. **Dense retrieval handles Altumint's small corpus well** -- BGE-large embeddings produce clear score separation between relevant and irrelevant pages (e.g., 0.742 vs 0.620 for sample 0000).

---

### Bottlenecks and Weaknesses

1. **Generation quality is the primary bottleneck** -- Retrieval is near-perfect but EM=0.200 and F1=0.456. The model has the right context but produces wrong or partial answers in 4/5 samples.
2. **Brevity prompt "Answer:" prefix** -- Documented pending fix. Directly breaks EM for at least 1 sample.
3. **Title block / spatial layout understanding** -- Flat OCR text loses the spatial structure of engineering drawing title blocks. GPT-4o cannot reliably parse CHECKED/DRAWN/APPROVED fields from linearized text.
4. **Source selection in generation** -- When multiple retrieved chunks contain related but different information (e.g., TM001 assembly steps vs DC001 summary), GPT-4o sometimes picks the wrong one.
5. **CLIP image ranking on technical drawings** -- MRR=0.333 for the one visual query. All engineering drawings cluster together in CLIP space.
6. **ANLS metric missing** -- EM is overly strict for Altumint's technical answers. "Answer: 3.000" vs "3.000" should score high on ANLS but scores 0.0 on EM.
7. **Only 5 samples** -- Too few to draw statistical conclusions. The full 123-sample (72 text + 51 visual) evaluation is needed.
8. **VQA accuracy meaningless** -- Scores 0.067 but is not applicable to Altumint (single annotation, proprietary dataset).
9. **Faithfulness metric unreliable** -- RAGAS faithfulness inconsistently scores correct answers as 0.0.

---

### Recommendations (Prioritized)

1. **Fix the "Answer:" brevity prompt prefix** (high impact, low effort) -- This is a known pending task. Removing the prefix will immediately improve EM by at least 1 sample (0003) and clean up F1 scores across the board.

2. **Add ANLS metric** (high impact, medium effort) -- Known pending task. ANLS would correctly score sample 0003 ("Answer: 3.000" vs "3.000") and sample 0002 ("POE Injector (Beneath Modem)" vs full sentence) much higher than EM. For Altumint's technical answers with minor formatting differences, ANLS is the most appropriate metric. Consider the extended ANLS* variant (arXiv:2402.03848) which handles structured outputs.

3. **Run full 123-sample evaluation** (high impact, medium effort) -- 5 samples are insufficient to characterize the pipeline. The full set includes 72 text queries and 51 visual queries, which will surface systematic patterns in CLIP failures on DC001/DC002/wiring and text retrieval failures on DC004's symbol-heavy tables.

4. **Improve prompt for source selection** (medium impact, low effort) -- Sample 0001 shows the model using TM001 (procedural detail) over DC001 (authoritative summary). Adding prompt guidance like "prefer summary/specification documents over procedural steps for quantity questions" could help.

5. **Implement ColPali for image retrieval** (high impact, high effort) -- Known pending task. ColPali (Faysse et al., 2024) uses late-interaction multi-vector embeddings from a VLM, producing layout-aware page embeddings that would dramatically outperform CLIP on engineering drawings. CLIP's inability to discriminate between DC001, DC002, and the wiring diagram is a fundamental limitation of its natural-image training.

6. **Add cross-encoder re-ranker** (medium impact, medium effort) -- Known pending task. A re-ranker on the text retrieval side could push the correct page from rank 3 to rank 1 for samples 0001 and 0002, improving MRR from 0.733 toward 1.0 and potentially improving source selection in generation.

7. **Debug RAGAS faithfulness** (medium impact, medium effort) -- The faithfulness metric remains broken across multiple runs and datasets. Either diagnose the RAGAS backend issue or replace with a simpler LLM-judge faithfulness check.

---

### Anomalies / Flags

1. **No cross-modal consistency metric reported.** The config includes `cross_modal_consistency` in multimodal_metrics, but it does not appear in any per-sample or aggregate metrics. This metric may not be implemented or may require conditions not met by these samples.

2. **Image retrieval metrics only computed for 1 of 5 samples.** Only sample 0002 has `image_mrr`/`image_recall@5`/`image_ndcg@5`. The other 4 samples lack image ground truth IDs (they appear to be text-only queries). The aggregate image_recall@5=1.0 and image_mrr=0.333 are based on a single data point.

3. **Sample 0004 empty text chunks.** Three of the five retrieved text chunks for sample 0004 are empty strings (from DC004 pages). These are continuity check table pages with dense symbol content that may not have been properly OCR-extracted or chunked. The Qdrant scores for these empty chunks (0.462-0.488) are in the same range as meaningful chunks, suggesting the embeddings were computed on metadata or headers rather than content.

4. **Duplicate relevant_sources in sample 0002.** The `relevant_sources` list contains `altumint_dc001_01_flvm_electronics_enclosure_assembly_drawing_p01` twice. This does not affect metrics but suggests a data quality issue in the QA pair definition.

---

### SOTA Research Recommendations

| Weakness Observed | Recommended Approach | Expected Impact | Citation |
|---|---|---|---|
| CLIP fails to discriminate engineering drawings (image MRR=0.333, scores within 0.035 spread) | **ColPali** -- VLM-based visual retrieval with late interaction; layout-aware page embeddings | Would replace CLIP entirely for document page retrieval; shown to dramatically outperform text+OCR pipelines on visually complex documents | Faysse et al. (2024) |
| EM too strict for technical answers with minor format differences | **ANLS*** -- edit-distance-based metric tolerating minor string differences, extended for structured outputs | Would correctly credit "Answer: 3.000" vs "3.000" and partial sentence matches | Jaume et al. (2024) |
| Title block field confusion from flat OCR text (sample 0004) | **LayoutLMv3** -- pre-trained with unified text+image masking, preserves spatial layout relationships | Could parse CHECKED/DRAWN/APPROVED fields by understanding 2D position, not just token sequence | Huang et al. (2022) |

**Full citations:**
- Faysse, M., Sibille, H., Wu, T., et al. (2024). "ColPali: Efficient Document Retrieval with Vision Language Models." ICLR 2025. [arXiv:2407.01449](https://arxiv.org/abs/2407.01449)
- Jaume, G., Auer, C., Katti, A. R., et al. (2024). "ANLS* -- A Universal Document Processing Metric for Generative Large Language Models." [arXiv:2402.03848](https://arxiv.org/abs/2402.03848)
- Huang, Y., Lv, T., Cui, L., Lu, Y., Wei, F. (2022). "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking." ACM MM 2022. [arXiv:2204.08387](https://arxiv.org/abs/2204.08387)
