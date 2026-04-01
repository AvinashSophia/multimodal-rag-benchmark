Perform a thorough end-to-end audit of ID consistency across all pipeline stages. Silently corrupted IDs cause metrics (Recall@k, MRR, nDCG, Attribution Accuracy) to score zero even when retrieval is correct.

## Audit these five stages in order:

### Stage 1 — Dataset loading (`pipeline/datasets/`)
- Read each dataset loader (hotpotqa.py, docvqa.py, gqa.py).
- Find where `get_corpus()` returns `(corpus, corpus_ids)`. Confirm `corpus_ids` are unique, stable, and match the format used later in retrieval. Note the exact ID format (e.g. `hotpotqa_{idx}`, `chunk_{hash}`, etc.).

### Stage 2 — Retrieval index (`pipeline/retrieval/`)
- Read bm25.py, dense.py, clip_retriever.py.
- Confirm that `index(corpus, corpus_ids)` stores the `corpus_ids` directly.
- Confirm that `retrieve()` returns those same IDs in `RetrievalResult.text_ids` and `RetrievalResult.image_ids`. Check for any re-indexing, sorting, or deduplication that could shift ID–chunk alignment.

### Stage 3 — Model prompt (`pipeline/models/`)
- Read `_build_prompt()` in gpt.py, gemini.py, qwen_vl.py.
- Confirm IDs passed in are actually embedded in the prompt text (e.g. `[{text_ids[i]}]: {chunk}`).
- Check whether the model's `Sources: [id1, id2, ...]` response is parsed back into `ModelResult.sources`. **Known gap:** `ModelResult.sources` is currently always `[]` in gpt.py — flag this if unfixed.

### Stage 4 — Runner ID construction (`pipeline/runners/run_benchmark.py`)
- Check how `relevant_sources` is built for images: currently `f"image_{sample.id}_{i}"`. Confirm this format matches the image IDs returned in `RetrievalResult.image_ids` from clip_retriever. Mismatch here silently zeroes Attribution Accuracy for images.
- Check that `retrieved.text_ids` and `relevant_text_ids` passed to `evaluator.evaluate_sample()` are in the same format. Note: `relevant_text_ids` is currently hardcoded `None` — flag if so.

### Stage 5 — Evaluator (`pipeline/evaluation/`)
- Read retrieval_metrics.py and grounding_metrics.py.
- Confirm `recall_at_k`, `mrr`, `ndcg_at_k` compare IDs as strings using `set()` intersection — so format must match exactly (case, prefix, separator).
- Confirm `attribution_accuracy` compares `used_sources` (from `ModelResult.sources`) against `relevant_sources` — both must use the same ID format.

## Report format
For each stage, state:
1. What ID format is produced or expected
2. Whether it matches the adjacent stage
3. Any mismatch, gap, or hardcoded `None` found — with the exact file and line number

Propose the minimal code fix for each mismatch found. Do not refactor beyond what is needed to fix ID consistency.
