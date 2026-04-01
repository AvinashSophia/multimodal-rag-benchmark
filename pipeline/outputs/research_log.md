# Multimodal RAG Benchmark — Research Log

This file tracks analysis summaries across all benchmark runs for cross-run comparison.
Newest entries appear at the top.

---

## docvqa_denseqdrant_clip_gpt_20260401_113545 — 2026-04-01

**Run:** docvqa | dense_qdrant | clip | gpt-4o
**Goal achievement:** 6/10 — Pipeline retrieves the correct image every time but struggles with numeric precision reading, prompt over-truncation on multi-token answers, and misconfigured evaluation metrics.
**Strongest component:** Image retrieval — CLIP fusion (α=0.5) achieves perfect recall@5=1.0 with MRR=0.767.
**Weakest component:** Evaluation + Prompt — vqa_accuracy (0.267) is misleadingly low due to wrong metric (VQA protocol vs ANLS); brevity prompt truncates multi-token answers.
**Primary bottleneck:** Evaluation metric mismatch (vqa_accuracy not suited for DocVQA single-annotation format) + prompt over-truncation causing "ITC" instead of "ITC Limited".
**Top 3 next steps:**
1. Add ANLS metric as DocVQA's primary answer metric (Low effort, immediately more accurate reporting)
2. Fix brevity prompt: "shortest complete answer" not "single word" (Low effort, recovers truncated answers)
3. Add ColPali retriever for document cluster disambiguation (Medium effort, highest retrieval quality gain)
**Interesting finding:** GPT-4o correctly answered sample 1 despite the correct image being at rank 3 (MRR=0.333) — the model synthesized across all 5 retrieved images and found the right answer. This suggests MRR improvements may not linearly improve EM for GPT-4o, which is robust to retrieval rank noise.
**Key metrics:** image_recall@5=1.0 | image_mrr=0.767 | exact_match=0.6 | f1=0.733 | attribution_accuracy=0.8 | vqa_accuracy=0.267

→ Full analysis: [docvqa_denseqdrant_clip_gpt_20260401_113545/analysis.md](pipeline/outputs/logs/docvqa_denseqdrant_clip_gpt_20260401_113545/analysis.md)

---
