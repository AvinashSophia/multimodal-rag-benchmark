# Multimodal RAG Benchmark -- Research Log

This file tracks analysis summaries across all benchmark runs for cross-run comparison.
Newest entries appear at the top.

---

## altumint_hybridelasticqdrant_colpaliqdrant_gpt_20260406_154229 -- 2026-04-06

**Run:** altumint | hybrid_elastic_qdrant (BM25 + BGE-large RRF) + ColPali v1.3-merged (MaxSim) | gpt-4o | 109 samples (66 text, 43 visual). ANLS metric now active.
**Key metrics:** text_recall@5=0.881 | text_mrr=0.674 | text_ndcg@5=0.725 | image_recall@5=0.860 | image_mrr=0.668 | image_ndcg@5=0.716 | exact_match=0.468 | f1=0.635 | anls=0.575 | attribution_accuracy=0.766 | faithfulness=0.491
**Verdict:** ColPali delivers massive image retrieval gains over CLIP baseline (image_recall +0.194, image_mrr +0.295, image_ndcg +0.270) and hybrid text retrieval improves over dense-only (+0.064 recall). However, EM slightly regressed (-0.032 vs CLIP baseline EM=0.500), with 12 regressions vs 5 improvements on 109 common samples. The CLIP clustering problem is resolved -- ColPali MaxSim scores show strong page discrimination. The bottleneck has shifted decisively from retrieval to generation: 47/109 (43%) samples retrieve correctly but answer wrong. Text-only EM=0.606 vs Visual EM=0.256 remains the largest performance gap. DC004 continuity table cell confusion persists (EM=0.333). ColPali fails on DC005 plain-text tables and wiring diagrams (ImgR@5=0.0 for both). 11 samples missing vs prior run (109 vs 120) -- needs investigation. Top priorities: (1) generation quality via structured prompts or reasoning models, (2) ANLS* for fairer evaluation, (3) cross-encoder re-ranker for DC004, (4) ColQwen2 upgrade for remaining image retrieval gaps.

-> Full analysis: altumint_hybridelasticqdrant_colpaliqdrant_gpt_20260406_154229/analysis.md

---

## altumint_denseqdrant_clip_gpt_20260406_121857 -- 2026-04-06

**Run:** altumint | dense_qdrant (BGE-large) + CLIP (ViT-B-32, fusion_alpha=0.5) | gpt-4o | 120 samples (re-parsed without --ocr, regenerated QA from field technician perspective, fixed "Answer:" prefix bug, fresh Qdrant rebuild)
**Key metrics:** text_recall@5=0.817 | text_mrr=0.630 | image_recall@5=0.667 | image_mrr=0.374 | exact_match=0.500 | f1=0.654 | attribution_accuracy=0.729 | faithfulness=0.484
**Verdict:** Major improvement over prior run (EM: 0.238 -> 0.500, F1: 0.529 -> 0.654). The "Answer:" prefix fix accounts for roughly half the EM gain (51 samples previously affected, now zero). Regenerated field-technician QA questions are more specific and actionable, contributing the other half. Text retrieval improved across all metrics (+5.5pp recall@5). DC004 continuity tables saw dramatic retrieval recovery (tR@5: 0.400 -> 0.800). Config/IP/password questions achieve 80% EM. CLIP clustering persists -- TM001_p09 is still image top-1 for 36% of queries with a score spread of only 0.023. Remaining bottlenecks: (1) CLIP replacement with ColPali, (2) table cell disambiguation for DC004, (3) ANLS metric to properly credit near-miss answers (estimated 0.70-0.75 ANLS vs 0.50 EM), (4) visual-only questions (32% EM vs 53% non-visual). Serial number hallucination observed in 5 samples.

-> Full analysis: [altumint_denseqdrant_clip_gpt_20260406_121857/analysis.md](pipeline/outputs/logs/altumint_denseqdrant_clip_gpt_20260406_121857/analysis.md)

---

## altumint_denseqdrant_clip_gpt_20260406_105313 -- 2026-04-06

**Run:** altumint | dense_qdrant (BGE-large) + CLIP (ViT-B-32, fusion_alpha=0.5) | gpt-4o | 122 samples (first full-dataset run; text corpus includes table markdown + summaries; both ID bugs fixed)
**Key metrics:** text_recall@5=0.762 | text_mrr=0.605 | image_recall@5=0.620 | image_mrr=0.386 | exact_match=0.238 | f1=0.529 | attribution_accuracy=0.716 | faithfulness=0.410
**Verdict:** First complete Altumint evaluation (122 samples). Retrieval is solid (tR@5=0.762) but image retrieval suffers from severe CLIP clustering -- TM001_p09 is image top-1 for 38.5% of all queries with an average score spread of only 0.023. The "Answer:" prefix remains the largest EM drag: 51/122 samples have the prefix, all scoring EM=0.0; without it, EM would be ~0.40. Generation is the clear bottleneck: 67 samples with perfect text retrieval still get EM=0.0. Visual queries (n=50) underperform text-only (n=72) on every metric (F1: 0.361 vs 0.646). DC004 continuity tables are the weakest doc type (tR@5=0.400, faithfulness=0.100) due to symbol-vocabulary mismatch with dense embeddings. Yellow arrow spatial pointer questions fail completely (4/4 at F1=0.0). Top priorities: fix "Answer:" prefix, add ANLS metric, replace CLIP with ColPali, add BM25/hybrid for DC004 tables.

-> Full analysis: [altumint_denseqdrant_clip_gpt_20260406_105313/analysis.md](pipeline/outputs/logs/altumint_denseqdrant_clip_gpt_20260406_105313/analysis.md)

---

## altumint_denseqdrant_clip_gpt_20260406_094357 -- 2026-04-06

**Run:** altumint | dense_qdrant (BGE-large) + CLIP (ViT-B-32, fusion_alpha=0.5) | gpt-4o | 5 samples (post-bugfix: rebuilt Qdrant index with long-form IDs, fixed image_ids to use source_page_id)
**Key metrics:** text_recall@5=1.0 | text_mrr=0.733 | text_ndcg@5=0.800 | image_recall@5=1.0 | image_mrr=0.333 | exact_match=0.200 | f1=0.456 | attribution_accuracy=0.800 | faithfulness=0.400
**Verdict:** Both ID consistency bugs fixed -- retrieval metrics fully recovered (text_recall and image_recall both 1.0, up from 0.0). Text ranking is strong (MRR=0.733, 3/5 at rank 1). Image ranking weak (MRR=0.333) due to CLIP clustering on engineering drawings. Generation is now the clear bottleneck: EM=0.200 with 1 perfect answer, 2 partial matches (brevity prefix, phrasing difference), and 2 complete failures (wrong source selection for WAGO count, title block field confusion on DC002). Brevity "Answer:" prefix and missing ANLS metric are the two highest-priority fixes. Attribution accuracy strong at 0.800 (4/5 correct citations). RAGAS faithfulness remains unreliable. Only 1 of 5 samples had image ground truth, so image metrics are based on a single data point.

-> Full analysis: [altumint_denseqdrant_clip_gpt_20260406_094357/analysis.md](pipeline/outputs/logs/altumint_denseqdrant_clip_gpt_20260406_094357/analysis.md)

---

## altumint_denseqdrant_clip_gpt_20260406_083337 -- 2026-04-06

**Run:** altumint | dense_qdrant (BGE-large) + CLIP (ViT-B-32, fusion_alpha=0.5) | gpt-4o | 5 samples (new QA pairs with text/visual query split)
**Key metrics:** text_recall@5=0.0 | image_recall@5=0.0 | exact_match=0.0 | f1=0.156 | attribution_accuracy=0.533 | faithfulness=0.0
**Verdict:** Catastrophic regression from the Apr 2 run, caused by two ID consistency bugs: (1) the Qdrant text index retains stale short-form IDs (e.g., `DC001_p01`) from prior indexing while ground truth now uses long-form IDs (e.g., `altumint_dc001_01_...`), making all text retrieval metrics artificially 0.0; (2) visual query `relevant_image_ids` are set to figure crop IDs (`..._p01_fig_00`) but CLIP indexes page screenshot IDs (`..._p01`), zeroing image recall. EM=0.0 across all 5 samples despite one semantically correct answer ("Beneath Modem" for POE Injector location). GPT-4o failed to read engineering drawing dimensions (0.750 vs 3.000) and title block fields ("A" vs "Steven Hartig"). Blocking fix: delete and re-index `dense_text_altumint` Qdrant collection; fix figure-to-page ID mapping in evaluator.

-> Full analysis: [altumint_denseqdrant_clip_gpt_20260406_083337/analysis.md](pipeline/outputs/logs/altumint_denseqdrant_clip_gpt_20260406_083337/analysis.md)

---

## altumint_denseqdrant_clip_gpt_20260402_154718 -- 2026-04-02

**Run:** altumint | dense_qdrant (BGE-large) + CLIP (ViT-B-32, fusion_alpha=0.5) | gpt-4o | 5 samples
**Key metrics:** image_recall@5=1.0 | image_mrr=1.0 | image_ndcg@5=1.0 | exact_match=0.6 | f1=0.7 | attribution_accuracy=0.733 | vqa_accuracy=0.2 | faithfulness=0.4
**Verdict:** Image retrieval is perfect across all 5 samples (MRR=1.0), outperforming DocVQA's 0.767 MRR thanks to Altumint's smaller, visually distinctive 36-page corpus. However, text retrieval metrics are entirely missing -- a critical evaluation gap caused by the Altumint dataset loader not populating `supporting_facts` metadata. Answer quality (EM=0.6) matches DocVQA exactly, confirming GPT-4o generation as the shared bottleneck. Two failure modes: (1) brevity truncation ("LiFePO4" vs full spec), the same prompt issue seen in DocVQA; (2) wrong source selection despite correct retrieval (model chose TM001 over DC001 for an assembly drawing question). VQA accuracy is structurally capped at 0.333 per sample due to single annotations. Faithfulness shows the same RAGAS anomaly -- correct answers score 0.0 while wrong answers score 1.0. CLIP shows clustering bias toward DC001/DC002 engineering drawings. Top priorities: add `supporting_facts` to Altumint loader, run full 108-sample evaluation, add ANLS metric, fix brevity prompt.

-> Full analysis: [altumint_denseqdrant_clip_gpt_20260402_154718/analysis.md](pipeline/outputs/logs/altumint_denseqdrant_clip_gpt_20260402_154718/analysis.md)

---

## docvqa_denseqdrant_clip_gpt_20260401_113545 -- 2026-04-02 (re-analysis)

**Run:** docvqa | dense_qdrant + CLIP (ViT-B-32, fusion_alpha=0.5) | gpt-4o | 5 samples
**Key metrics:** image_recall@5=1.0 | image_mrr=0.767 | image_ndcg@5=0.826 | exact_match=0.6 | f1=0.733 | attribution_accuracy=0.8 | vqa_accuracy=0.267
**Verdict:** Image retrieval is strong (perfect recall, good ranking), but answer quality is constrained by three independent issues: (1) vqa_accuracy is misleadingly low because the standard VQA protocol is wrong for single-annotation DocVQA -- ANLS would report ~0.65-0.75; (2) the "single word" system prompt truncates multi-token entities (ITC vs ITC Limited); (3) GPT-4o misreads fine numeric data in tables (0.26 vs 0.28). Cross-run comparison with HotpotQA shows identical EM (0.6) across both modalities, suggesting GPT-4o generation is the shared bottleneck, not retrieval. CLIP has a cluster disambiguation problem -- three UC San Diego pages score identically -- addressable by ColPali (Faysse et al., arXiv:2407.01449) or cross-encoder re-ranking. HotpotQA's faithfulness=0.0 across all samples is anomalous and likely a RAGAS backend bug. Top priorities: add ANLS metric, fix brevity prompt, debug RAGAS faithfulness.

-> Full analysis: [docvqa_denseqdrant_clip_gpt_20260401_113545/analysis.md](pipeline/outputs/logs/docvqa_denseqdrant_clip_gpt_20260401_113545/analysis.md)

---
