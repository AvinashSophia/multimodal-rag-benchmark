---
name: Altumint dataset baseline metrics and known issues
description: Altumint benchmark runs -- current best retrieval is Apr 6 154229 (ColPali+Hybrid); best EM remains Apr 6 121857 (CLIP+Dense, EM=0.500). Generation is now the bottleneck, not retrieval.
type: project
---

## Latest Run: altumint_hybridelasticqdrant_colpaliqdrant_gpt_20260406_154229 (109 samples, 2026-04-06)

Config: hybrid_elastic_qdrant (BM25 + BGE-large RRF) + ColPali v1.3-merged (MaxSim) + GPT-4o. ANLS metric now active.

**Aggregate metrics:** text_recall@5=0.881 | text_mrr=0.674 | text_ndcg@5=0.725 | image_recall@5=0.860 | image_mrr=0.668 | image_ndcg@5=0.716 | EM=0.468 | F1=0.635 | ANLS=0.575 | attribution_accuracy=0.766 | faithfulness=0.491

**Per-document-type (ranked by EM):**
- DC005_wire_lengths (n=5): EM=0.800, F1=0.800, tR@5=1.000, iR@5=0.000
- TM002_config (n=33): EM=0.667, F1=0.753, tR@5=0.848
- DC002_hole_location (n=2): EM=0.500, F1=0.500, tR@5=1.000
- TM001_assembly (n=56): EM=0.357, F1=0.595, tR@5=0.893, iR@5=0.926
- DC004_continuity (n=9): EM=0.333, F1=0.556, tR@5=1.000
- Wiring_diagram (n=3): EM=0.333, F1=0.333, tR@5=0.667, iR@5=0.000
- DC001_assembly_drawing (n=1): EM=0.000, tR@5=0.000, iR@5=1.000

**Text-only vs Visual split:** Text EM=0.606, Visual EM=0.256 (0.35 gap)

**ColPali vs CLIP delta (109 common samples):**
- Image Recall@5: +0.194, Image MRR: +0.295, Image nDCG@5: +0.270 (massive improvement)
- Text Recall@5: +0.064 (hybrid vs dense improvement)
- EM: -0.032, F1: -0.019 (slight regression despite better retrieval)
- 12 EM regressions vs 5 improvements on common samples

**Key finding:** ColPali solved the CLIP clustering problem but answer quality slightly regressed. 47/109 (43%) samples retrieve correctly but answer wrong -- generation is the bottleneck. DC004 table cell confusion (0 vs OL) and visual interpretation failures are top error categories.

**Sample count mismatch:** 109 vs 120 in prior run. 11 IDs missing (altumint_0000, 0001, 0004, 0013, 0022, 0029, 0085, 0097, 0100, 0101, plus one more). Needs investigation.

## Prior Baseline: altumint_denseqdrant_clip_gpt_20260406_121857 (120 samples)

Config: dense_qdrant (BGE-large) + CLIP (ViT-B-32) + GPT-4o. No ANLS.

**Aggregate:** EM=0.500 | F1=0.654 | tR@5=0.817 | iR@5=0.667 | iMRR=0.374

**How to apply:**
- Generation quality is now the #1 priority. Structured prompts for DC004, reasoning models, or chain-of-thought may help.
- ColQwen2 recommended as next image retriever upgrade (reported +5.3 nDCG@5 over ColPali on ViDoRe).
- ANLS* metric recommended to replace strict EM for final reporting.
- Investigate 11 missing samples before next run.
- Do not trust faithfulness or VQA accuracy for Altumint.
