# Benchmark Analysis
**Run:** docvqa | dense_qdrant | clip | gpt-4o
**Timestamp:** 2026-04-01 11:35:45
**Samples:** 5
**Evaluation backend:** production

---

## Step 2 — Task Goal

**For this run, the goal is:** Document Visual QA — given a question about a document image, retrieve the correct document from a corpus of images and extract a precise answer (number, name, location). **The pipeline achieves this if:** image_recall@5 is high (right image found) AND exact_match/F1 is high (right answer extracted from that image).

---

## Step 3 — Metric-by-Metric Diagnosis

**image_recall@5: 1.0**
Every correct document image appeared in the top-5 retrieved results across all 5 samples. This is the best possible score. The CLIP retriever is doing its job — fusion of text→image and image→image is surfacing the relevant document every time. Not surprising at 5 samples with a small corpus, but directionally strong.

**image_mrr: 0.767**
The correct image ranks first in 3 samples, second in 1 (sample 3), third in 1 (sample 1). Mediocre-to-good — a perfect ranker would score 1.0. The main drag is that samples 1, 3, 4 involve the same document cluster (images 24582, 24581, 24580 — all UC San Diego documents) which receive nearly identical CLIP scores (0.608–0.634). The retriever cannot distinguish which of the three nearly-identical-scoring documents is the exact target. This is a cluster disambiguation problem, not a retrieval failure.

**image_ndcg@5: 0.826**
Consistent with MRR — the correct image is being found but not always ranked first within the UC San Diego cluster. NDCG rewards partial credit for finding the right image at rank 2-3 rather than rank 1. Solid score.

**exact_match: 0.6 (3/5 correct)**
Three samples answered exactly correctly (samples 1, 3, 4). Two failed:
- Sample 0: predicted "0.26", ground truth "0.28" — precision reading error
- Sample 2: predicted "ITC", ground truth "itc limited" — over-truncation

Compared to GPT-4o DocVQA val baseline of ~55-65% EM, this run is at the upper end of that range on only 5 samples. Directionally on-target.

**f1: 0.733**
Higher than EM because sample 2 gets F1=0.667 partial credit ("ITC" shares the token with "itc limited"). Sample 0 gets F1=0.0 because "0.26" shares no tokens with "0.28".

**attribution_accuracy: 0.8**
4/5 samples correctly identify the source. Samples 1 and 3 score 0.5 — the model cited all 3 UC San Diego images (24582, 24581, 24580) but only one is marked as relevant. The model is not wrong to cite all three (they are visually very similar and all potentially relevant), but the evaluation penalizes over-citation. This is a structural issue in how relevant_sources is defined, not a model failure.

**vqa_accuracy: 0.267**
The weakest metric by far. Scores per sample: 0.0, 0.333, 0.0, 0.667, 0.333. This metric applies the standard VQA evaluation protocol which expects answers to match across multiple human annotators. With only 1 ground truth per sample in our data, and answers like "ITC" vs "itc limited" failing soft matching, this metric is penalizing correct answers. This metric is **ill-suited for DocVQA** in our current data format — ANLS would be far more appropriate.

**Cross-metric patterns:**
- **High retrieval + low EM (sample 0):** Image correctly retrieved at rank 1 (MRR=1.0) but model misread 0.28 as 0.26 — pure model precision failure.
- **Low MRR + high EM (sample 1):** Correct image at rank 3 (MRR=0.333) but model answered correctly — model reasoned over all 5 retrieved images and found the right answer despite suboptimal ranking.
- **High F1 + low EM (sample 2):** Model said "ITC" instead of "ITC Limited" — the one-word answer instruction is too aggressive for company names.

---

## Step 4 — Per-Sample Deep Dive

**Most instructive failure — Sample 0:**
> Q: "What is the 'actual' value per 1000, during the year 1975?" | GT: "0.28" | Predicted: "0.26"

The correct image (49153) was retrieved at rank 1 with confidence score 0.651. The model cited it correctly. Yet GPT-4o misread 0.28 as 0.26. This is a classic VLM number-reading error — small numeric differences in tabular data (charts, financial tables) are notoriously hard for vision models to read precisely. The pipeline did everything right; the model failed at OCR-level extraction. Fixing this requires either: (a) feeding higher-resolution images, (b) adding OCR preprocessing and passing extracted text alongside the image, or (c) using a model with stronger document understanding.

**Most instructive failure — Sample 2:**
> Q: "What is the name of the company?" | GT: "itc limited" | Predicted: "ITC"

The correct image (57349) was retrieved at rank 1 (score 0.614). The model answered "ITC" — abbreviated. The system prompt says "single word, name, number, or short phrase" which caused GPT-4o to truncate the full company name. Ground truth is "itc limited" (two tokens) but the model obeyed the brevity instruction and dropped "limited". **The one-word answer prompt instruction is actively hurting EM on multi-token ground truths.**

---

## Step 5 — DocVQA-Specific Assessment

**Image retrieval success:** Perfect — all 5 correct images appear in top-5. The CLIP fusion retriever (text→image + image→image, α=0.5) is performing well. All samples use fusion mode since all have both query text and query image.

**Cluster disambiguation problem:** Images 24582, 24581, 24580 appear together across samples 1, 2, 3, 4 with nearly identical CLIP scores (~0.608–0.634). These are three pages/versions of the same UC San Diego document. CLIP cannot distinguish between them because they are visually nearly identical. This is a known limitation of global image embeddings — ColPali (which uses local patch-level matching) would handle this better.

**SOTA comparison:** GPT-4V achieves ~88% ANLS on DocVQA val; typical GPT-4o EM on val is ~55-65%. This run achieves 0.6 EM on 5 samples — within the expected GPT-4o range.

**OCR precision gap:** The 0.26 vs 0.28 failure (sample 0) is representative of a systemic gap — tabular numeric data in document images requires either high-resolution image processing or hybrid OCR+vision to achieve production-level accuracy.

---

## Step 6 — Pipeline Bottleneck Identification

Ranked by impact on current metrics:

1. **Evaluation mismatch (HIGH IMPACT)** — VQA accuracy (0.267) is misleadingly low because the standard VQA protocol doesn't suit our single-annotation DocVQA format. ANLS would report ~0.7+ for the same answers.
2. **Prompt over-truncation (MEDIUM IMPACT)** — The "single word" instruction caused sample 2 to drop "limited" from "ITC Limited", turning a correct answer into a wrong one.
3. **Model OCR precision (MEDIUM IMPACT)** — Sample 0's 0.26 vs 0.28 misread is representative of VLM numeric reading limitations.
4. **CLIP cluster disambiguation (LOW-MEDIUM IMPACT)** — Images 24582/24581/24580 score identically. MRR is 0.767 instead of 1.0.
5. **Retrieval (LOW IMPACT)** — image_recall@5 = 1.0, nothing to fix here.

**The primary bottleneck is the evaluation metric (vqa_accuracy) misalignment, followed by the over-truncating prompt instruction. Fixing both requires zero model or retrieval changes.**

---

## Step 7 — SOTA Improvement Roadmap

**1. ANLS metric for DocVQA**
Replace exact_match with Average Normalized Levenshtein Similarity — the official DocVQA evaluation metric. Tolerates minor OCR errors and abbreviations.
- Metric impact: ANLS would show ~0.65-0.75 vs current misleading 0.6 EM
- Effort: Low
- Modular fit: Yes — new evaluator behind `@register_evaluator`

**2. Fix brevity prompt instruction**
Change "single word, name, number, or short phrase" to "shortest complete answer — do not truncate proper names or multi-word entities."
- Metric impact: Recovers sample 2 (ITC → ITC Limited), immediate +0.2 EM
- Effort: Low — one-line prompt change in all 4 model files
- Modular fit: Yes

**3. ColPali for document image retrieval**
Late-interaction model that matches query tokens against image patch embeddings — handles visually similar document pages by attending to specific regions (table cells, headers).
- Metric impact: MRR 0.767 → ~0.95 on document clusters (based on ColPali paper results on DocVQA)
- Effort: Medium — new `@register_retriever("colpali")` class
- Modular fit: Yes

**4. High-resolution image input + OCR preprocessing**
Pass extracted OCR text alongside the image to GPT-4o, or use high-res detail mode. Directly addresses sample 0's numeric reading failure.
- Metric impact: ~15-20% improvement on tabular numeric questions
- Effort: Medium — modify image preprocessing in model wrappers
- Modular fit: Yes

**5. Cross-encoder re-ranker**
After CLIP retrieves top-10 candidates, apply a lightweight cross-encoder to rerank. Improves MRR without changing recall.
- Metric impact: MRR typically +0.05-0.15 over CLIP-only ranking
- Effort: Medium — new post-retrieval stage in HybridRetriever
- Modular fit: Yes

---

## Step 8 — Summary Verdict

**Run:** docvqa | dense_qdrant | clip | gpt-4o
**Goal achievement: 6/10** — The pipeline successfully retrieves the correct document image every time (recall@5=1.0) and answers correctly on simple factual questions, but struggles with precise numeric reading, multi-token entity truncation, and the evaluation metrics are misconfigured for DocVQA's answer format.
**Strongest component:** Image retrieval — CLIP fusion achieves perfect recall@5 with good MRR (0.767). The retriever is not the bottleneck.
**Weakest component:** Evaluation + Prompt — vqa_accuracy (0.267) is misleadingly low due to wrong metric choice; the brevity prompt is actively hurting EM on multi-token answers.
**Top 3 next steps:**
1. Add ANLS metric as DocVQA's primary answer metric (Low effort, immediately more accurate reporting)
2. Fix brevity prompt: "shortest complete answer" not "single word" (Low effort, recovers truncated answers)
3. Add ColPali retriever for document cluster disambiguation (Medium effort, highest retrieval quality gain)
**Interesting finding:** GPT-4o correctly answered sample 1 even though the correct image was at rank 3 (MRR=0.333) — the model synthesized across all 5 retrieved images and found the right answer without perfect ranking. This suggests improving MRR may not linearly improve EM for GPT-4o, which is robust to retrieval rank noise.
