# Benchmark Run Analysis

**Run:** `altumint_denseqdrant_clip_gpt_20260402_154718`
**Date:** 2026-04-02
**Analyst:** benchmark-results-analyzer

---

## 1. Run Summary

| Field | Value |
|-------|-------|
| Directory | `pipeline/outputs/logs/altumint_denseqdrant_clip_gpt_20260402_154718/` |
| Dataset | Altumint FLVM (proprietary, 7 PDFs, 36 pages) |
| Text Retrieval | `dense_qdrant` (BAAI/bge-large-en-v1.5), top_k=5 |
| Image Retrieval | `clip` (ViT-B-32), top_k=5, fusion_alpha=0.5 |
| Model | gpt-4o, temperature=0.0, max_tokens=512 |
| Samples analyzed | 5 (of 108 total QA pairs; `max_samples: 5`) |
| Evaluation backend | production |

---

## 2. Metric Scorecard

### Aggregated Metrics (from metrics.json)

| Category | Metric | Score | Status |
|----------|--------|-------|--------|
| **Retrieval (image)** | image_recall@5 | **1.000** | Excellent |
| | image_mrr | **1.000** | Excellent |
| | image_ndcg@5 | **1.000** | Excellent |
| **Retrieval (text)** | text_recall@5 | **MISSING** | Not computed |
| | text_mrr | **MISSING** | Not computed |
| | text_ndcg@5 | **MISSING** | Not computed |
| **Answer Quality** | exact_match | **0.600** | Moderate |
| | f1 | **0.700** | Moderate |
| **Grounding** | faithfulness | **0.400** | Weak |
| | attribution_accuracy | **0.733** | Moderate |
| **Multimodal** | vqa_accuracy | **0.200** | Weak |
| | cross_modal_consistency | **MISSING** | Not computed |

### Per-Sample Breakdown

| Sample | Question | EM | F1 | Faithful | Attrib | VQA | img_MRR |
|--------|----------|---:|---:|---------:|-------:|----:|--------:|
| altumint_0000 | What component is located beneath the modem? | 1.0 | 1.0 | 1.0 | 1.0 | 0.33 | 1.0 |
| altumint_0001 | What type of battery is used in the enclosure? | 0.0 | 0.33 | 0.0 | 0.67 | 0.0 | 1.0 |
| altumint_0002 | Which document should be referenced for more details on the assembly? | 0.0 | 0.17 | 1.0 | 0.0 | 0.0 | 1.0 |
| altumint_0003 | What is the diameter of the antenna hole? | 1.0 | 1.0 | 0.0 | 1.0 | 0.33 | 1.0 |
| altumint_0004 | Where is the conduit hole located? | 1.0 | 1.0 | 0.0 | 1.0 | 0.33 | 1.0 |

---

## 3. Key Findings

### Finding 1: Perfect image retrieval -- CLIP perfectly identifies source pages

All 5 samples achieved image_recall@5=1.0, image_mrr=1.0, and image_ndcg@5=1.0. The source page image was always ranked #1 by the CLIP retriever. This is remarkable and likely attributable to the small corpus (36 pages) and visually distinctive page layouts across the 7 PDFs. The top image for every sample was always one of DC001_p01 or DC002_p01, which are engineering drawings with strong visual signatures.

### Finding 2: Text retrieval metrics are entirely missing -- a critical evaluation gap

No text retrieval metrics (recall, MRR, nDCG) were computed. Root cause: the Altumint dataset loader does not populate `supporting_facts` in sample metadata (see `pipeline/datasets/altumint.py` line 179). The benchmark runner at `pipeline/runners/run_benchmark.py` lines 102-108 only extracts `relevant_text_ids` from `supporting_facts.titles`. Since this key is absent for Altumint, `relevant_text_ids` is always empty, and the evaluator skips text retrieval metrics.

This means we cannot measure whether the dense (BGE) text retriever is finding the correct source page text. This is a significant blind spot given that Altumint uniquely indexes both text AND images from the same pages.

### Finding 3: Faithfulness scores are unreliable (0.4 average, 0/1 binary pattern)

Faithfulness shows a suspicious pattern: samples with correct short answers (0000, 0003, 0004) score 0.0, while the wrong-answer samples score 1.0 or 0.0 inconsistently. Sample 0003 answers "0.750" correctly from DC002_p01 but gets faithfulness=0.0, while sample 0002 gives a wrong document reference but gets faithfulness=1.0. This is the same RAGAS faithfulness anomaly observed in the DocVQA run -- the metric appears to measure whether claims can be found in retrieved text chunks, not whether the answer is actually correct. Short factual/numerical answers produce fewer "claims" for RAGAS to verify, leading to paradoxically low scores.

### Finding 4: VQA accuracy is artificially low (0.2 average) due to evaluation protocol mismatch

VQA accuracy uses the standard VQA protocol (min(#humans who agree / 3, 1)). Since Altumint has only 1 ground-truth answer per sample (`all_answers` always has exactly one entry), the maximum achievable VQA score per sample is 0.333 (1/3). Even for 3 out of 5 samples with perfect EM=1.0, VQA accuracy was only 0.333. This metric is structurally capped at 0.333 for single-annotation datasets and should not be used as a quality indicator for Altumint.

### Finding 5: Answer quality is strong for factual/numerical questions but fails on reference questions

Samples 0000 (factual), 0003 (numerical), 0004 (spatial) all achieved EM=1.0. The two failures:
- **altumint_0001**: Ground truth "Tycon 10Ah Aluminum LiFePO4 Battery" vs predicted "LiFePO4" -- the model gave a technically correct but truncated answer, losing the brand and capacity. This is the same brevity prompt problem documented in the DocVQA analysis.
- **altumint_0002**: Ground truth "FLVM Electronics Enclosure - Assembly Drawing.pdf" vs predicted "TM001(02) Flashing Light Video Monitor Assembly Instructions" -- the model cited the wrong document entirely. Notably, the correct answer (DC001_p01 text) was the #1 retrieved text chunk, but the model chose TM001_p01 instead. This is a **generation failure with correct retrieval** -- the model had the right evidence but selected the wrong one.

### Finding 6: Cross-modal consistency is not computed

This metric requires `text_only_answer` and `image_only_answer` from the model, but these fields are commented out in `run_benchmark.py` (line 137-138). For Altumint, where the same page is available as both text and image, this would be an especially valuable metric.

---

## 4. Failure Pattern Analysis

### Pattern A: Brevity truncation (1/5 samples, 20%)

Sample 0001 -- "LiFePO4" instead of "Tycon 10Ah Aluminum LiFePO4 Battery". The model's system prompt encourages concise answers, which causes it to strip brand names, quantities, and qualifiers. This is a known issue (documented in DocVQA analysis) and will recur for any question expecting a full product name, part number, or specification string.

### Pattern B: Wrong source selection despite correct retrieval (1/5 samples, 20%)

Sample 0002 -- The correct text chunk (DC001_p01 containing "Assembly Drawing.pdf") was retrieved at rank 1 with score 0.610, and the correct image (DC001_p01) was also rank 1. But the model cited TM001_p01 (rank 2, score 0.584) instead. The model appears to have interpreted "assembly" in the question and matched it to the more detailed assembly instructions document rather than the assembly drawing. This is a semantic disambiguation failure in the generation stage.

### Pattern C: Attribution vs source mismatch (sample 0002)

Sample 0002 has attribution_accuracy=0.0 because the model cited TM001_p01, but the evaluator's `relevant_sources` included the ground-truth page_id (DC001_p01 from `sample.image_ids`). The model's cited source does not intersect with relevant sources, hence 0.0.

---

## 5. Strengths

1. **Image retrieval is flawless.** CLIP with ViT-B-32 achieves perfect recall, MRR, and nDCG on this corpus. The engineering drawings and technical pages have sufficiently distinct visual signatures for CLIP to discriminate at top_k=5 over a 36-page corpus.

2. **Factual and numerical QA is strong.** 3 out of 5 samples achieve EM=1.0. The dense text retriever brings relevant context, and GPT-4o extracts precise values (e.g., "0.750" for antenna hole diameter, "Bottom" for conduit location).

3. **Citation behavior is generally correct.** In 4 of 5 samples, the model's cited sources match the sources it actually used. The model demonstrates reliable source attribution when it picks the right answer.

4. **Token efficiency is excellent.** Completion tokens range from 10-21 per sample (average ~15), with prompt tokens around 4,500. The model is not generating verbose explanations.

---

## 6. Bottlenecks and Weaknesses

### Bottleneck 1 (Critical): Missing text retrieval evaluation

Without `supporting_facts` metadata, we have no visibility into whether the BGE dense retriever is correctly ranking source pages by text. Given that Altumint has dual-indexed pages (text + image), measuring text retrieval quality is essential to understanding whether retrieval improvements should focus on the text side.

**Root cause:** `AltumintDataset.load()` does not populate `metadata["supporting_facts"]`. The `source_page_id` is stored but not in the format the runner expects.

### Bottleneck 2 (High): VQA metric is structurally broken for single-annotation datasets

VQA accuracy can never exceed 0.333 for Altumint samples. This metric should either be replaced with ANLS (which is already in the pending tasks list) or adjusted to use a different protocol for single-annotation datasets.

### Bottleneck 3 (Medium): Faithfulness metric is noisy and potentially miscalibrated

The RAGAS faithfulness implementation produces counterintuitive results where correct short answers score 0.0 and incorrect answers score 1.0. This was also observed in the DocVQA run. Until debugged, faithfulness should be treated as unreliable.

### Bottleneck 4 (Medium): Brevity prompt causes answer truncation

The system prompt encourages concise answers, which works well for factual/numerical questions but strips important qualifiers from specification-type answers (brand, capacity, material).

---

## 7. Recommendations (Prioritized)

### Priority 1: Add `supporting_facts` to Altumint dataset loader

Populate `metadata["supporting_facts"]` using the existing `source_page_id` field so that text retrieval metrics are computed. The fix is straightforward -- in `pipeline/datasets/altumint.py`, add to the metadata dict:

```python
"supporting_facts": {"titles": [page_id]}
```

This will unblock text_recall@5, text_mrr, and text_ndcg@5 computation for all Altumint runs.

### Priority 2: Replace VQA accuracy with ANLS for document QA datasets

As noted in the pending tasks, ANLS (Average Normalized Levenshtein Similarity) is the standard metric for DocVQA and would be appropriate for Altumint as well. ANLS handles partial matches naturally (e.g., "LiFePO4" vs "Tycon 10Ah Aluminum LiFePO4 Battery" would score ~0.3-0.4 rather than the misleading 0.0 from EM or 0.333 from VQA). This is a known pending task.

### Priority 3: Fix the brevity prompt for specification-type questions

The current system prompt should instruct the model to include full product names, part numbers, and specifications when the question asks about a specific component or document. A conditional prompt or question-type-aware prompting strategy would address this without changing behavior for factual/numerical questions.

### Priority 4: Debug RAGAS faithfulness implementation

Investigate why faithfulness=0.0 for correct short answers. The issue may be in how RAGAS decomposes short answers into claims. If a single-token answer (e.g., "0.750") generates zero extractable claims, faithfulness defaults to 0.0. This is a systemic issue affecting all datasets and should be prioritized.

### Priority 5: Run the full 108-sample evaluation

The current run used only 5 samples. Altumint has 108 QA pairs spanning 5 question types (factual, numerical, visual, procedural, cross_doc). The current sample is too small to draw conclusions about question-type-specific performance. Set `max_samples: null` or a higher value (e.g., 50) to get statistically meaningful results.

### Priority 6: Enable cross-modal consistency evaluation

Uncomment `text_only_answer` and `image_only_answer` in `run_benchmark.py` to enable the cross_modal_consistency metric. For Altumint, where both text and image come from the same page, this metric would reveal whether GPT-4o is leveraging visual evidence (drawings, diagrams) or relying solely on text.

### Priority 7: Consider ColPali for visual document retrieval

[ColPali (Faysse et al., arXiv:2407.01449, 2024)](https://arxiv.org/abs/2407.01449) is a Vision Language Model that produces multi-vector embeddings from document page images, using late interaction matching (ColBERT-style). It eliminates the need for separate OCR/text extraction and retrieval pipelines. On the ViDoRe benchmark, ColPali outperforms traditional text-based retrieval on visually rich documents. For Altumint's engineering drawings and wiring diagrams -- where spatial layout carries semantic meaning -- ColPali could outperform the current BGE+CLIP dual retrieval by natively understanding document structure.

[M3DocRAG (Cho et al., arXiv:2411.04952, 2024)](https://arxiv.org/abs/2411.04952) demonstrates a full multimodal RAG framework using ColPali for page retrieval + Qwen2-VL for generation, achieving state-of-the-art on multi-page document QA. This architecture is directly relevant to Altumint's multi-document setting and is worth benchmarking against the current BGE+CLIP+GPT-4o pipeline.

### Priority 8: Add a cross-encoder re-ranker stage

A [BGE cross-encoder re-ranker (e.g., bge-reranker-v2-m3)](https://huggingface.co/BAAI/bge-reranker-v2-m3) after the initial dense retrieval could help with cases like sample 0002, where the correct chunk was retrieved at rank 1 but the model selected the wrong one. Re-ranking with a cross-encoder scores each (query, passage) pair with full attention, producing more accurate relevance signals that could be passed as metadata to the generation model. This is a known pending task (re-ranker integration).

---

## 8. Anomalies and Flags

### Flag 1: No text retrieval metrics computed (DATA INTEGRITY)

This is NOT an ID consistency issue -- it is a metadata gap. The Altumint loader does not provide `supporting_facts`, so the evaluation framework correctly skips text retrieval metrics. However, this means every Altumint run to date has been evaluated without measuring text retrieval quality. Any conclusions about "retrieval performance" for Altumint are based solely on image retrieval.

### Flag 2: Image IDs are consistent across pipeline stages (VERIFIED)

Image IDs follow the `{DOC_ID}_p{page:02d}` pattern consistently:
- Dataset loader assigns `page_id` in `_build_corpus()` (line 125)
- Same IDs used for `sample.image_ids` (line 177)
- Retrieved `image_ids` in results.json match the corpus ID format
- Attribution `relevant_sources` correctly use the same IDs
- No evidence of ID corruption or mismatch

### Flag 3: Faithfulness scores contradict answer correctness

| Sample | EM | Faithfulness | Expected |
|--------|---:|------------:|----------|
| 0000 | 1.0 | 1.0 | Correct |
| 0001 | 0.0 | 0.0 | Ambiguous -- partial match |
| 0002 | 0.0 | 1.0 | **Contradictory** -- wrong answer, high faithfulness |
| 0003 | 1.0 | 0.0 | **Contradictory** -- correct answer, zero faithfulness |
| 0004 | 1.0 | 0.0 | **Contradictory** -- correct answer, zero faithfulness |

3 of 5 samples show contradictory faithfulness scores. This strongly suggests a systematic issue in the RAGAS faithfulness implementation for short answers.

### Flag 4: CLIP score clustering on DC001_p01 and DC002_p01

Across all 5 samples, DC001_p01 and DC002_p01 consistently appear as top-2 CLIP results regardless of the question. These are the engineering drawings which have distinctive visual layouts. This suggests CLIP may be biased toward visually complex pages rather than semantically relevant ones. The perfect retrieval metrics mask this issue because the source page happens to be one of these high-scoring pages for the 5 sampled questions. A full 108-sample run would reveal whether this clustering causes failures on questions sourced from text-heavy pages (TM001, TM002, DC004, DC005).

---

## 9. Cross-Run Comparison

| Metric | DocVQA (prior run) | Altumint (this run) | Delta |
|--------|---:|---:|---:|
| image_recall@5 | 1.000 | 1.000 | 0.000 |
| image_mrr | 0.767 | 1.000 | +0.233 |
| image_ndcg@5 | 0.826 | 1.000 | +0.174 |
| exact_match | 0.600 | 0.600 | 0.000 |
| f1 | 0.733 | 0.700 | -0.033 |
| attribution_accuracy | 0.800 | 0.733 | -0.067 |
| vqa_accuracy | 0.267 | 0.200 | -0.067 |

Altumint's image retrieval is significantly better (perfect MRR/nDCG vs DocVQA's 0.77/0.83), which makes sense given the smaller, more visually distinctive corpus (36 vs hundreds of pages). Answer quality (EM, F1) is nearly identical, reinforcing the prior finding that GPT-4o generation -- not retrieval -- is the shared bottleneck across datasets. Both runs share the same VQA accuracy issue (structurally capped by single annotations) and the same faithfulness anomaly.

---

## 10. Token Usage Summary

| Sample | Prompt tokens | Completion tokens | Total |
|--------|-----:|-----:|------:|
| altumint_0000 | 4,595 | 12 | 4,607 |
| altumint_0001 | 4,448 | 19 | 4,467 |
| altumint_0002 | 4,576 | 21 | 4,597 |
| altumint_0003 | 4,571 | 12 | 4,583 |
| altumint_0004 | 4,983 | 10 | 4,993 |
| **Total** | **23,173** | **74** | **23,247** |

Average prompt: ~4,635 tokens. Average completion: ~15 tokens. The model is extremely concise (by design). No cached prompt tokens were used.

---

## References

- [ColPali: Efficient Document Retrieval with Vision Language Models (Faysse et al., 2024)](https://arxiv.org/abs/2407.01449)
- [M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding (Cho et al., 2024)](https://arxiv.org/abs/2411.04952)
- [Retrieval-Augmented Generation for Large Language Models: A Survey (Gao et al., 2023)](https://arxiv.org/abs/2312.10997)
- [BGE Reranker v2 (BAAI)](https://huggingface.co/BAAI/bge-reranker-v2-m3)
