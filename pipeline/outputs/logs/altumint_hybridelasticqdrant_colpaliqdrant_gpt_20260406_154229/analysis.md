## Benchmark Run Analysis

### Run Summary
- **Directory:** `altumint_hybridelasticqdrant_colpaliqdrant_gpt_20260406_154229`
- **Dataset:** Altumint (proprietary FLVM technical documentation, 7 PDFs, 36 pages)
- **Retrieval:** text=hybrid_elastic_qdrant (BM25 + BGE-large-en-v1.5, RRF k=60), image=colpali_qdrant (ColPali v1.3-merged, MaxSim scoring)
- **Model:** GPT-4o (temperature=0.0, max_tokens=512)
- **Samples analyzed:** 109
- **Date:** 2026-04-06
- **ANLS metric:** Active (first Altumint run with ANLS)

### Metric Scorecard

| Category | Metric | Score |
|---|---|---|
| **Retrieval (Text)** | Recall@5 | 0.881 |
| | MRR | 0.674 |
| | nDCG@5 | 0.725 |
| **Retrieval (Image)** | Recall@5 | 0.860 |
| | MRR | 0.668 |
| | nDCG@5 | 0.716 |
| **Answer Quality** | Exact Match | 0.468 |
| | F1 | 0.635 |
| | ANLS | 0.575 |
| **Grounding** | Faithfulness | 0.491 |
| | Attribution Accuracy | 0.766 |
| **Multimodal** | VQA Accuracy | 0.156 (not meaningful for Altumint) |

### Comparison with Prior Altumint Baseline (dense_qdrant + CLIP, 20260406_121857, 120 samples)

| Metric | CLIP+Dense Baseline | ColPali+Hybrid (this run) | Delta |
|---|---|---|---|
| text_recall@5 | 0.817 | 0.881 | **+0.064** |
| text_mrr | 0.630 | 0.674 | **+0.044** |
| text_ndcg@5 | 0.677 | 0.725 | **+0.048** |
| image_recall@5 | 0.667 | 0.860 | **+0.194** |
| image_mrr | 0.374 | 0.668 | **+0.295** |
| image_ndcg@5 | 0.447 | 0.716 | **+0.270** |
| exact_match | 0.500 | 0.468 | -0.032 |
| f1 | 0.654 | 0.635 | -0.019 |
| attribution_accuracy | 0.729 | 0.766 | **+0.037** |
| faithfulness | 0.484 | 0.491 | +0.007 |

### Key Findings

1. **ColPali delivers massive image retrieval improvement.** Image recall@5 jumped from 0.667 to 0.860 (+0.194), image MRR nearly doubled from 0.374 to 0.668, and image nDCG@5 rose from 0.447 to 0.716. The CLIP clustering problem (where TM001_p09 dominated as top-1 for 36% of queries) is resolved -- ColPali MaxSim scores now show strong page discrimination with scores ranging from 8.8 to 17.8 across retrieved pages.

2. **Hybrid text retrieval (BM25+Dense RRF) also improves.** Text recall@5 increased from 0.817 to 0.881 (+0.064), with MRR and nDCG both gaining ~0.05. The BM25 component helps with symbol-heavy DC004 tables where dense embeddings alone struggle.

3. **Answer quality slightly regressed despite better retrieval.** EM dropped from 0.500 to 0.468 (-0.032) and F1 from 0.654 to 0.635 (-0.019). This paradox -- better retrieval, worse answers -- indicates the generation stage is now the clear bottleneck. 48.6% of samples (53/109) have successful retrieval but EM=0.

4. **ANLS provides a fairer evaluation than EM.** ANLS=0.575 sits between EM=0.468 and F1=0.635, properly crediting near-miss answers. Many EM=0 failures are string-level mismatches, not semantic errors (e.g., "OL" vs "OL Ohm", "6A -> 13A -> 2A" vs "6A -> 13A -> 2A" with arrow encoding differences).

5. **11 samples are missing compared to prior run** (109 vs 120). Missing IDs: altumint_0000, 0001, 0004, 0013, 0022, 0029, 0085, 0097, 0100, 0101, plus one more. This needs investigation -- possible dataset loader regression.

### Per-Document-Type Breakdown

| Doc Type | N | EM | F1 | ANLS | tR@5 | iR@5 | tMRR | iMRR | Faith | Attr |
|---|---|---|---|---|---|---|---|---|---|---|
| DC005 wire lengths | 5 | 0.800 | 0.800 | 0.800 | 1.000 | 0.000 | 0.650 | 0.000 | 0.200 | 0.800 |
| TM002 config | 33 | 0.667 | 0.753 | 0.748 | 0.848 | 0.303 | 0.656 | 0.259 | 0.540 | 0.808 |
| DC002 hole location | 2 | 0.500 | 0.500 | 0.500 | 1.000 | 0.500 | 0.375 | 0.500 | 0.000 | 1.000 |
| TM001 assembly | 56 | 0.357 | 0.595 | 0.491 | 0.893 | 0.446 | 0.729 | 0.338 | 0.512 | 0.768 |
| DC004 continuity | 9 | 0.333 | 0.556 | 0.444 | 1.000 | 0.000 | 0.606 | 0.000 | 0.333 | 0.704 |
| Wiring diagram | 3 | 0.333 | 0.333 | 0.500 | 0.667 | 0.000 | 0.500 | 0.000 | 0.667 | 0.333 |
| DC001 assembly drawing | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.250 | 1.000 | 0.500 |

### Failure Pattern Analysis

**Pattern 1: Unit/symbol truncation (DC004 continuity tables)**
The model consistently drops the unit symbol from continuity readings. Ground truth "OL Ohm" becomes predicted "OL"; ground truth "0 Ohm" becomes "0". This accounts for most DC004 EM=0 failures (6/9 samples). The brevity prompt encourages short answers, which strips units. ANLS partially captures this (0.444 vs EM=0.333).

**Pattern 2: Over-verbose answers**
When retrieval is successful, GPT-4o often produces answers that are correct but include extra context. Examples:
- GT="four" -> Pred="QTY 4 - Tycon TSBAT12-10L-WT Lithium Batteries" (F1=0.29)
- GT="10-32" -> Pred="10-32 socket head cap screws" (F1=0.33)
- GT="Negative" -> Pred="The outside of the barrel jack is negative..." (F1=0.29)

**Pattern 3: Visual interpretation failures (TM001 assembly photos)**
Questions requiring reading assembly photos often fail. Examples:
- altumint_0041: "Which component is labeled SmartSolar..." -> EM=0, F1=0
- altumint_0053: "Which component is directly connected to the Netgear switch..." -> EM=0, F1=0
- altumint_0070: "What is the color of the wire connected to the bottom-left..." -> EM=0, F1=0

**Pattern 4: ColPali fails on text-only documents**
DC004 (continuity tables), DC005 (wire length tables), and the wiring diagram all have image_recall@5=0.000. ColPali is optimized for visually rich document pages; these plain-text tables and schematic drawings do not produce distinctive visual features. This is compensated by strong text retrieval (tR@5=1.000 for DC004 and DC005).

**Pattern 5: Both retrievers fail (5 samples)**
Five samples had text_recall@5=0 AND image_recall@5=0:
- altumint_0037: Phoenix Contact component labels (niche product names)
- altumint_0066: Wire label identification (visual detail)
- altumint_0083: Blue component label (visual detail + specific product)
- altumint_0087: Camera model name from configuration interface
- altumint_0117: Wire gauge (generic question, relevant page not retrieved)

### Strengths

1. **ColPali resolves the CLIP clustering problem.** MaxSim scoring produces discriminative scores (spread of 4-8 points between top-1 and top-5) compared to CLIP's near-identical scores (spread of 0.02).

2. **Hybrid text retrieval is robust.** 88.1% recall@5 with BM25+Dense RRF -- only 13/109 samples miss the correct text chunk entirely.

3. **Strong performance on configuration questions.** TM002 config achieves EM=0.667 and ANLS=0.748 -- IP addresses, passwords, settings, and port numbers are well-handled.

4. **Wire length lookup is nearly solved.** DC005 achieves EM=0.800 (4/5 correct) with perfect text recall.

5. **Attribution accuracy is high.** 0.766 overall, with TM002 at 0.808 -- the model correctly cites its sources in most cases.

### Bottlenecks & Weaknesses

1. **Generation quality is the #1 bottleneck.** 53/109 (48.6%) samples retrieve correctly but answer incorrectly. Retrieval improvements no longer translate to answer improvements.

2. **Visual question performance lags text.** TM001 assembly (heavily visual) achieves EM=0.357 vs TM002 config (mostly text) at EM=0.667 -- a 0.31 gap.

3. **Faithfulness metric is unreliable.** 51/109 samples score faithfulness=0, including many with correct answers and correct citations. This is a known RAGAS anomaly on this dataset. Do not use faithfulness for decision-making.

4. **VQA accuracy is not meaningful.** At 0.156, this metric reflects the single-annotation protocol limitation, not actual performance. Exclude from reporting.

5. **DC004 table cell disambiguation.** The model confuses "0 Ohm" with "OL Ohm" in continuity tables due to dense table layout where multiple rows look similar. The table extraction in text chunks uses markdown formatting that loses cell-to-cell spatial relationships.

6. **Sample count discrepancy.** 109 vs 120 in baseline -- 11 missing samples need investigation before this run can be considered a fair comparison.

### Recommendations (Prioritized)

1. **Improve generation prompts for structured answers (HIGH IMPACT).** The 48.6% retrieval-success/answer-failure rate is the biggest lever. Specifically:
   - Add unit-preservation instructions for DC004 ("always include the unit symbol")
   - Add brevity calibration ("answer with the exact value and unit, nothing more")
   - Consider few-shot examples for table lookups and numerical questions
   - Estimated impact: +0.05--0.10 EM based on the 17 partial-match failures alone

2. **Adopt ANLS as the primary answer metric (MEDIUM IMPACT).** ANLS=0.575 better reflects actual performance than EM=0.468. Many "failures" are trivial string differences. ANLS is already active in this run -- promote it to the headline metric for Altumint reporting.

3. **Investigate missing 11 samples (MEDIUM IMPACT).** The sample count dropped from 120 to 109 between runs. This could be a dataset loader issue or a filtering change. The missing samples may disproportionately affect one document type, skewing comparisons.

4. **Add a cross-encoder re-ranker for DC004 (MEDIUM IMPACT).** The continuity table pages are all retrieved (tR@5=1.0) but at varying ranks (MRR=0.606). A cross-encoder re-ranker could push the correct page to rank 1, helping the model focus on the right table section.

5. **Upgrade to ColQwen2 for image retrieval (LOW-MEDIUM IMPACT).** ColPali already achieves 0.860 recall@5 and 0.716 nDCG@5, so gains will be incremental. ColQwen2 reported +5.3 nDCG@5 over ColPali on ViDoRe V1. The main benefit would be on TM001 assembly photos (current iR@5=0.446).

6. **Do not trust faithfulness or VQA accuracy for Altumint.** Both metrics produce misleading results on this dataset. Faithfulness shows the known RAGAS inversion anomaly (correct answers scoring 0.0). VQA accuracy is structurally capped by single annotations.

### Anomalies / Flags

1. **Sample count mismatch:** 109 samples vs 120 in the prior dense+CLIP baseline. 11 IDs are missing. This must be resolved before making definitive cross-run comparisons.

2. **ColPali zero recall on text-heavy documents:** DC004, DC005, and wiring diagram all show image_recall@5=0.000. This is expected behavior (these are plain tables/schematics with minimal visual distinctiveness) but should be documented as a known ColPali limitation.

3. **Faithfulness inversion:** Multiple samples with correct answers and correct citations score faithfulness=0.0 (e.g., altumint_0005: EM=1.0, attribution=1.0, faithfulness=0.0). This is a known RAGAS anomaly.

4. **Duplicate relevant_sources:** Some samples list the same source_page_id twice in `relevant_sources` (e.g., altumint_0002 lists `dc001...p01` twice). This may inflate or deflate attribution_accuracy depending on how the evaluator handles deduplication.

5. **Arrow encoding mismatch:** altumint_0032 ground truth uses "->" but model predicts with Unicode arrows. These are semantically identical but produce EM=0 and F1=0.75. ANLS correctly handles this (0.69).

### SOTA Research Recommendations

| Weakness Observed | Recommended Approach | Expected Impact | Citation |
|---|---|---|---|
| ColPali image recall=0 on text-heavy tables (DC004/DC005) | ColQwen2 (Qwen2-VL backbone, better text-in-image understanding) | +5.3 nDCG@5 over ColPali on ViDoRe V1 | Faysse et al. (2024) |
| Generation failures on table cell lookup (DC004 EM=0.333) | Structured prompting with HTML table format + few-shot examples | Up to +15% accuracy on tabular tasks | Sui et al. (2024) |
| Re-ranking needed for DC004 multi-page tables | Cross-encoder re-ranker (BGE, Jina) in RAG pipeline | +59% MRR@5 reported on financial docs | Li et al. (2024) |

**Full citations:**

- Faysse, M., Fernandez, H., Music, S., Picot, M., Briand, E., & Elias, J. (2024). "ColPali: Efficient Document Retrieval with Vision Language Models." arXiv. [arXiv:2407.01449](https://arxiv.org/abs/2407.01449) -- ColQwen2 is trained with the same methodology on Qwen2-VL 2B backbone; model weights at `vidore/colqwen2-v1.0` on HuggingFace.

- Sui, Y., Zhou, M., Zhou, M., Han, S., & Zhang, D. (2024). "Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study." WSDM '24. [arXiv:2305.13062](https://arxiv.org/abs/2305.13062) -- HTML markup with format explanations achieves 65.43% overall accuracy on tabular tasks, outperforming plain-text and markdown formats.

- Mace, Q., Music, S., Faysse, M., Fernandez, H., & Briand, E. (2025). "ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval." arXiv. [arXiv:2505.17166](https://arxiv.org/abs/2505.17166) -- ColQwen2-v1.0 achieves nDCG@5=0.583 on ViDoRe V2; provides updated evaluation methodology for visual document retrieval.

---

*Note: Altumint is a proprietary dataset. There are no published external baselines. All comparisons are relative across our own pipeline runs.*
