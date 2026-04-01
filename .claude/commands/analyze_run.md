You are a research scientist specializing in multimodal RAG systems, information retrieval, and NLP evaluation. Your task is to perform a deep, rigorous post-run analysis of a benchmark result and save it directly to files — do NOT print the analysis to the console. Only print a short confirmation at the end.

## Step 1 — Load the run

If `$ARGUMENTS` is provided, use it as the run directory path.
Otherwise find the most recent run: list `pipeline/outputs/logs/`, pick the directory with the latest timestamp.

Read all three files from the run directory:
- `config.json` — what was configured (dataset, retriever, model, evaluation backend)
- `metrics.json` — aggregated metrics across all samples
- `results.json` — per-sample results (question, ground truth, predicted answer, retrieved context, metrics)

## Step 2 — Understand the task goal

Before analyzing numbers, state what this dataset+retriever+model combination is trying to achieve:

- **hotpotqa**: Multi-hop reasoning — the system must retrieve and chain evidence across multiple documents to answer questions that cannot be answered from a single passage. Success = high Recall@k AND high F1/EM together.
- **docvqa**: Document visual QA — the system must retrieve the correct document image and extract precise information (numbers, dates, names) from it. Success = correct image retrieved AND exact answer extracted.
- **gqa**: Visual scene understanding — the system must retrieve the correct scene image and perform spatial/relational reasoning. Success = correct image retrieved AND reasoning-based answer correct.

State clearly: "For this run, the goal is ___. The pipeline achieves this if ___."

## Step 3 — Metric-by-metric diagnosis

For each metric in `metrics.json`, reason through it using this pattern:

**[Metric Name]: [value]**
- What this number means in plain language for this specific task
- Is this good, mediocre, or poor? (reference: random baseline, typical published results for this dataset)
- What does this tell us about which pipeline stage is working or failing?
- What is the direct cause of this value based on the config and per-sample evidence in results.json?

Cover all five categories in order:
1. **Retrieval** (recall_at_k, mrr, ndcg) — Is the right evidence being found?
2. **Answer** (exact_match, f1) — Is the model producing correct answers?
3. **Grounding** (faithfulness, attribution_accuracy) — Is the model using the retrieved evidence or hallucinating?
4. **Multimodal** (vqa_accuracy, cross_modal_consistency) — Is the visual understanding working?
5. **Cross-metric patterns** — Look for telling combinations:
   - High retrieval + low answer = model failure despite good retrieval
   - Low retrieval + high answer = model is ignoring retrieval and using parametric knowledge
   - High F1 + low EM = model is partially correct but adding extra words
   - Low faithfulness + high F1 = model is hallucinating correct answers (dangerous in production)

## Step 4 — Per-sample deep dive

Read the individual samples in `results.json`. For each sample:
- Did the retriever find the relevant source? Check `retrieved_context.text_ids` / `image_ids` against `attribution.relevant_sources`
- Did the model answer correctly? Compare `predicted_answer` vs `ground_truth`
- Did the model cite sources? Check `attribution.used_sources`
- If the answer is wrong, diagnose: retrieval failure, model failure, or prompt failure?

Identify the 1-2 most instructive failure cases and explain exactly what went wrong in the pipeline for that sample.

## Step 5 — Dataset-specific pipeline assessment

### If dataset = hotpotqa:
- Multi-hop success rate: how many samples required chaining 2+ documents? Did the retriever find both?
- Check if supporting facts appear in `retrieved_context.text_ids` — if not, this is a retrieval bottleneck
- Assess whether the text retriever method (bm25 vs dense) is suited for multi-hop: BM25 struggles with implicit entity chains, dense retrieval handles semantic similarity better
- State-of-the-art for HotpotQA: IRCoT (Interleaved Retrieval with CoT) achieves ~60% F1 by iteratively retrieving; standard single-hop retrieval tops out ~45% F1. Where does this run sit?

### If dataset = docvqa:
- Image retrieval success: is the correct document image in the top-k? Check image_ids retrieved vs relevant_image_ids
- DocVQA requires OCR-level precision — EM scores below 0.5 are common even with correct image retrieval because models mis-read numbers/dates
- State-of-the-art for DocVQA: Hi-VT5 achieves ~70% ANLS; GPT-4V achieves ~88% ANLS on val. EM is stricter — typical GPT-4o EM on DocVQA val is ~55-65%. Where does this run sit?
- Fusion analysis: was text→image or image→image the dominant signal? Check `metadata.mode` in retrieved_context

### If dataset = gqa:
- GQA requires spatial/relational reasoning — even perfect image retrieval won't help if the model can't reason about object relationships
- State-of-the-art for GQA: mPLUG-Owl3 achieves ~65% accuracy; GPT-4V ~70%. Where does this run sit?
- Check if questions with spatial reasoning ("left of", "behind", "between") score lower than attribute questions ("what color") — this would confirm the model, not retrieval, is the bottleneck

## Step 6 — Pipeline bottleneck identification

Rank each pipeline stage by how much it is limiting performance:

1. **Data/Corpus quality** — Is the corpus missing information needed to answer? (e.g. DocVQA with no OCR text)
2. **Retrieval** — Is the relevant evidence being found in top-k?
3. **Model** — Given correct evidence, is the model producing correct answers?
4. **Prompt** — Is the prompt format causing the model to answer in the wrong format?
5. **Evaluation** — Are metrics accurately capturing correctness? (e.g. EM penalises "yes" vs "Yes")

State: "The primary bottleneck is ___ because ___. Fixing this would have the highest impact on metrics."

## Step 7 — SOTA improvement roadmap

Based on the diagnosed bottlenecks, suggest improvements ranked by expected impact vs implementation effort. For each suggestion:

**[Technique name]**
- What it is (1 sentence)
- Which metric it improves and by how much (cite a paper or known result if possible)
- Implementation effort: Low / Medium / High
- Fits our modular architecture: Yes/No (can it be added as a new retriever/model/evaluator?)

Consider techniques from these areas based on what's bottlenecking:

**Retrieval improvements:**
- HyDE (Hypothetical Document Embeddings) — generate a fake answer, embed it, retrieve similar docs
- IRCoT — iterative retrieval interleaved with chain-of-thought reasoning
- ColBERT / ColPali — late interaction models for fine-grained token-level matching
- Re-ranking with cross-encoder (e.g. ms-marco-MiniLM) after initial retrieval
- Query expansion / decomposition for multi-hop questions

**Model improvements:**
- Chain-of-thought prompting for multi-hop questions
- Few-shot examples in prompt for better answer format compliance
- Self-consistency (sample multiple answers, take majority vote)

**Evaluation improvements:**
- ANLS (Average Normalized Levenshtein Similarity) instead of EM for DocVQA
- LLM-as-judge for open-ended answers where EM/F1 is too strict
- BERTScore for semantic answer similarity

## Step 8 — Summary verdict

Produce a concise research summary using exactly this structure (this will be included in both saved files, not printed to console):

**Run:** {dataset} | {text_retriever} | {image_retriever} | {model}
**Goal achievement:** [0-10 score] — [one sentence verdict on whether the pipeline is achieving its task goal]
**Strongest component:** [which pipeline stage is working best and why]
**Weakest component:** [which pipeline stage is the bottleneck and why]
**Top 3 next steps:** [ranked list — highest expected impact first]
**Interesting finding:** [one non-obvious insight from the data that is worth investigating further]

## Step 9 — Save analysis to files

After completing the full analysis, save it to two places:

### 9a — Per-run analysis file

Write the complete analysis (all steps above, fully written out) to:
`{run_dir}/analysis.md`

Use this exact header at the top of the file:
```
# Benchmark Analysis
**Run:** {dataset} | {text_retriever} | {image_retriever} | {model}
**Timestamp:** {run_timestamp parsed from directory name}
**Samples:** {max_samples from config}
**Evaluation backend:** {backend from config}
```

### 9b — Central research log

Read `pipeline/outputs/research_log.md` if it exists. If it does not exist, create it with this header:
```
# Multimodal RAG Benchmark — Research Log

This file tracks analysis summaries across all benchmark runs for cross-run comparison.
Newest entries appear at the top.

---
```

Then prepend (add at the top, after the header) a new entry with this structure:
```
## {run_directory_name} — {date}

**Run:** {dataset} | {text_retriever} | {image_retriever} | {model}
**Goal achievement:** {score}/10 — {verdict}
**Strongest component:** {value}
**Weakest component:** {value}
**Primary bottleneck:** {value}
**Top 3 next steps:**
1. {step 1}
2. {step 2}
3. {step 3}
**Interesting finding:** {value}
**Key metrics:** {list the 4-5 most important metrics and their values for this dataset}

→ Full analysis: [{run_directory_name}/analysis.md]({run_dir}/analysis.md)

---
```

Save the updated `research_log.md`.

Do not print anything else to the console during the analysis. Once both files are saved, print only this confirmation:

```
Analysis complete.
  → {run_dir}/analysis.md
  → pipeline/outputs/research_log.md
```
