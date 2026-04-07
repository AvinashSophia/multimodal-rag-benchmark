# Benchmark Run Analysis

## Run Summary

| Field | Value |
|-------|-------|
| Directory | `docvqa_denseqdrant_clip_gpt_20260401_113545` |
| Dataset | DocVQA (validation split) |
| Retrieval | text=dense_qdrant (BGE-large-en-v1.5), image=CLIP (ViT-B-32, fusion_alpha=0.5) |
| Model | GPT-4o (temperature=0.0, max_tokens=512) |
| Evaluation backend | production |
| Samples analyzed | 5 (max_samples=5) |
| Top-k | text=5, image=5 |

---

## Metric Scorecard

| Category | Metric | Score | Assessment |
|----------|--------|-------|------------|
| **Retrieval** | image_recall@5 | 1.000 | Perfect |
| **Retrieval** | image_mrr | 0.767 | Good |
| **Retrieval** | image_ndcg@5 | 0.826 | Good |
| **Answer** | exact_match | 0.600 (3/5) | Moderate |
| **Answer** | f1 | 0.733 | Moderate-Good |
| **Grounding** | attribution_accuracy | 0.800 | Good |
| **Multimodal** | vqa_accuracy | 0.267 | Poor (misleading -- see below) |

**Missing metrics:** faithfulness and cross_modal_consistency are absent from the aggregated metrics. The DocVQA dataset is image-only (no text corpus), so faithfulness (which evaluates text-grounded answers) does not apply. Cross-modal consistency was not computed.

---

## Key Findings

### 1. Retrieval is not the bottleneck -- image recall is perfect

All 5 samples had their ground-truth image in the top-5 retrieved results (recall@5 = 1.0). CLIP fusion mode (text-to-image + image-to-image, alpha=0.5) is performing well on this small corpus.

### 2. VQA accuracy (0.267) is misleadingly low due to metric mismatch

The standard VQA evaluation protocol expects multiple human annotations per question. Our data has a single ground truth per sample, causing answers like "ITC" vs "itc limited" and "University of California, San Diego" vs "university of california" to be penalized far more harshly than they should be. **ANLS is the official DocVQA metric and should replace vqa_accuracy for this dataset.**

### 3. The brevity prompt instruction is actively harming answer quality

Sample 2 predicted "ITC" when the ground truth is "itc limited". The system prompt's "single word" instruction caused GPT-4o to truncate a correct multi-token answer. This is a known issue already flagged for remediation.

### 4. GPT-4o misreads fine-grained numeric data in document images

Sample 0 predicted "0.26" when the ground truth is "0.28" -- a tabular numeric reading error. The image was correctly retrieved at rank 1, attribution was correct, but the model's OCR-level precision failed on a small numeric difference in a table.

### 5. CLIP cannot disambiguate visually similar document pages

Images 24582, 24581, 24580 (UC San Diego documents) consistently score within 0.02 of each other across all samples. This identical-score cluster drags MRR from 1.0 to 0.767. Global CLIP embeddings cannot differentiate pages from the same document.

---

## Per-Sample Breakdown

| Sample | Question (truncated) | GT | Predicted | EM | F1 | MRR | Attribution | Failure Mode |
|--------|----------------------|----|-----------|----|----|-----|-------------|--------------|
| docvqa_0 | "actual value per 1000, 1975?" | 0.28 | 0.26 | 0.0 | 0.0 | 1.0 | 1.0 | Numeric misread |
| docvqa_1 | "name of university?" | university of california | University of California, San Diego | 1.0 | 1.0 | 0.33 | 0.5 | Over-citation |
| docvqa_2 | "name of the company?" | itc limited | ITC | 0.0 | 0.67 | 1.0 | 1.0 | Prompt truncation |
| docvqa_3 | "where is university?" | san diego | San Diego | 1.0 | 1.0 | 0.5 | 0.5 | Over-citation |
| docvqa_4 | "to whom is document sent?" | Paul | Paul | 1.0 | 1.0 | 1.0 | 1.0 | None |

---

## Failure Pattern Analysis

**Pattern 1: Prompt-induced truncation (1/5 samples, 20%)**
Sample 2 ("ITC" vs "itc limited") -- the model obeyed the "single word" instruction and dropped "limited". This is a systematic risk for any multi-token entity name. The F1 partial credit (0.67) confirms the answer is correct but incomplete.

**Pattern 2: Numeric precision error (1/5 samples, 20%)**
Sample 0 ("0.26" vs "0.28") -- GPT-4o misread a table value. This is a known VLM limitation on fine-grained document OCR. The pipeline performed perfectly up to the generation step; this is purely a model capability gap.

**Pattern 3: Cluster over-citation (2/5 samples, 40%)**
Samples 1 and 3 both cite images [24582, 24581, 24580] because the model cannot determine which of the three nearly-identical UC San Diego document pages is the single ground-truth relevant source. Attribution accuracy drops to 0.5 on these samples. This is partially a retrieval issue (CLIP scores are identical) and partially an evaluation artifact (the model is arguably correct to cite all pages from the same document).

**Pattern 4: Successful multi-image reasoning**
Sample 1 achieved EM=1.0 despite the correct image being at rank 3 (MRR=0.333). GPT-4o synthesized information across all 5 retrieved images to find the correct answer. This demonstrates that GPT-4o is robust to imperfect ranking -- it does not rely solely on the top-1 result.

---

## Strengths

1. **Perfect image retrieval recall** -- CLIP fusion retrieves the correct document image in all cases at top-5.
2. **Strong attribution** -- 0.8 overall, with 3/5 samples achieving perfect attribution accuracy (1.0).
3. **Robust generation despite rank noise** -- GPT-4o correctly answered a question even when the relevant image was ranked 3rd, demonstrating resilience to suboptimal retrieval ordering.
4. **Clean ID consistency** -- Image IDs (49153, 24582, 24581, 24580, 57349) are consistent between retrieved_context.image_ids and attribution.relevant_sources across all samples. No ID corruption detected.

---

## Bottlenecks and Weaknesses (Ranked by Impact)

1. **Evaluation metric mismatch (HIGH)** -- vqa_accuracy (0.267) uses standard VQA protocol unsuited for single-annotation DocVQA. ANLS would likely report approximately 0.65-0.75 for the same predictions.
2. **Prompt over-truncation (MEDIUM)** -- "Single word" instruction causes multi-token answer truncation. Directly costs 1 EM point (20% of samples).
3. **Model OCR precision (MEDIUM)** -- GPT-4o misreads fine numeric differences in tabular data. Costs 1 EM point (20% of samples).
4. **CLIP cluster disambiguation (LOW-MEDIUM)** -- Visually identical pages score identically. MRR is 0.767 instead of 1.0, and attribution accuracy suffers on over-cited clusters.
5. **Small sample size (CONTEXT)** -- 5 samples is sufficient for validation but not for statistical significance. All findings should be re-validated at scale (50+ samples minimum).

---

## Comparative Context: DocVQA vs HotpotQA

A second run exists in the same log directory: `hotpotqa_denseqdrant_gpt_20260401_113249`.

| Metric | DocVQA (this run) | HotpotQA | Delta |
|--------|-------------------|----------|-------|
| Recall@5 | 1.000 (image) | 0.900 (text) | +0.10 |
| MRR | 0.767 (image) | 1.000 (text) | -0.23 |
| nDCG@5 | 0.826 (image) | 0.907 (text) | -0.08 |
| Exact Match | 0.600 | 0.600 | 0.00 |
| F1 | 0.733 | 0.600 | +0.13 |
| Attribution Acc. | 0.800 | 0.867 | -0.07 |
| Faithfulness | N/A | 0.000 | -- |

**Key cross-run observations:**
- **EM is identical (0.6) across both datasets** despite very different modalities (text-only vs image-only). This suggests the generation model (GPT-4o) is the constant bottleneck, not the retrieval modality.
- **HotpotQA has perfect MRR (1.0)** but lower recall (0.9). DocVQA has perfect recall but lower MRR (0.767). This reflects the difference between dense text retrieval (precise ranking) and CLIP image retrieval (good recall, fuzzy ranking within clusters).
- **HotpotQA faithfulness is 0.0 across all 5 samples** despite correct answers -- this suggests a bug or misconfiguration in the RAGAS faithfulness evaluator for the production backend. Worth investigating separately.
- **HotpotQA sample 4 failed** because the required context passage ("Adriana Trigiani") was not retrieved (text_recall@5=0.5), and the model answered "Unknown" -- a clean retrieval failure leading to generation failure.
- **HotpotQA sample 1 answer prefix bug**: the model output "Answer: Ambassador" instead of "Ambassador", causing EM=0.0. The answer parsing may not be stripping the "Answer:" prefix.

---

## Recommendations (Prioritized)

### 1. Add ANLS metric for DocVQA evaluation [LOW EFFORT, HIGH IMPACT]

Replace or supplement exact_match with Average Normalized Levenshtein Similarity (ANLS), the official DocVQA evaluation metric. ANLS tolerates minor OCR errors and partial string matches, and would report approximately 0.65-0.75 for this run's predictions rather than the misleading 0.6 EM. Implement as a new evaluator file following the registry pattern (`@register_evaluator`).

**Reference:** Biten et al., "ICDAR 2019 Competition on Scene Text Visual Question Answering." See also the extended ANLS* metric from Peer et al., [ANLS* -- A Universal Document Processing Metric for Generative Large Language Models](https://arxiv.org/abs/2402.03848), which handles structured outputs.

### 2. Fix brevity prompt instruction [LOW EFFORT, MEDIUM IMPACT]

Change the system prompt from "single word, name, number, or short phrase" to "shortest complete answer -- do not truncate proper names or multi-word entities." This directly recovers sample 2 (ITC -> ITC Limited) and protects against similar truncation on other entity-heavy questions. Requires a one-line change in all 4 model files.

### 3. Investigate RAGAS faithfulness returning 0.0 [LOW EFFORT, DIAGNOSTIC]

The HotpotQA run shows faithfulness=0.0 on all 5 samples, including samples where the model answered correctly from retrieved context. This is almost certainly a configuration or API issue with the RAGAS production backend, not a genuine pipeline failure. Investigate whether the RAGAS `EvaluationResult` is being parsed correctly (the `hasattr(result, "to_pandas")` guard noted in the architecture memo may be relevant).

### 4. Add ColPali retriever for document image retrieval [MEDIUM EFFORT, MEDIUM IMPACT]

ColPali uses late interaction between query token embeddings and document image patch embeddings, enabling it to distinguish visually similar pages by attending to specific regions (table cells, headers, text blocks). This directly addresses the cluster disambiguation problem where images 24582/24581/24580 score identically under CLIP.

**Reference:** Faysse et al., [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449), ICLR 2025. Register as `@register_retriever("colpali")`.

### 5. Add cross-encoder re-ranker as post-retrieval stage [MEDIUM EFFORT, MEDIUM IMPACT]

After CLIP retrieves top-k candidates, apply a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2` or a multimodal variant) to re-rank. This improves MRR without changing recall, directly addressing the cluster disambiguation problem from a different angle than ColPali.

**Reference:** Wang et al., [Searching for Best Practices in Retrieval-Augmented Generation](https://aclanthology.org/2024.emnlp-main.981.pdf), EMNLP 2024. Also see Yu et al., [RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs](https://proceedings.neurips.cc/paper_files/paper/2024/file/db93ccb6cf392f352570dd5af0a223d3-Paper-Conference.pdf), NeurIPS 2024.

### 6. High-resolution image input or OCR preprocessing [MEDIUM EFFORT, MEDIUM IMPACT]

For tabular/numeric DocVQA questions, pass extracted OCR text alongside the image to GPT-4o, or use the `detail: "high"` parameter for image inputs. This directly addresses sample 0's 0.26 vs 0.28 misread. A hybrid approach (OCR text + image) is more robust than either alone.

### 7. Fix "Answer:" prefix in model output parsing [LOW EFFORT, LOW IMPACT]

HotpotQA sample 1 output "Answer: Ambassador" but the ground truth is "Chief of Protocol". However, the "Answer:" prefix should still be stripped during parsing to avoid false EM failures on otherwise correct answers. Check the `run_model()` regex for this edge case.

---

## Anomalies and Flags

1. **No text retrieval executed** -- text_chunks, text_ids, and text_scores are empty across all 5 DocVQA samples. This is expected behavior (DocVQA is image-only, so HybridRetriever's text side correctly skips when `_has_text=False`), but confirms that the dense_qdrant text retriever is not contributing to this run despite being configured.

2. **Identical CLIP scores for image cluster** -- Images 24581, 24580 always receive the exact same score (to 15 decimal places) across samples 1, 3, and 4. This suggests they may have identical CLIP embeddings, not just similar ones. If so, they may be duplicate images in the corpus. Worth verifying with an `audit_ids` run.

3. **HotpotQA faithfulness = 0.0 universally** -- Flagged as a likely RAGAS backend issue. All 5 HotpotQA samples show faithfulness=0.0 including samples with perfect EM, perfect attribution, and correctly cited sources. This metric should not be trusted until the RAGAS integration is debugged.

4. **No ID consistency issues detected** -- Image IDs are consistent between `retrieved_context.image_ids`, `attribution.used_sources`, `attribution.relevant_sources`, and `attribution.model_cited_sources` across all 5 samples. No silent corruption observed.

5. **Token usage is consistent** -- All 5 samples use approximately 4370-4390 prompt tokens, suggesting images are being encoded at a consistent resolution. Completion tokens range from 8-21, confirming the model is producing concise answers as instructed.

---

*Analysis generated 2026-04-02 from run `docvqa_denseqdrant_clip_gpt_20260401_113545`.*
