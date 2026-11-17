# Experiment 005 — Independent neural vectors and real FAISS IVF

**Status:** pre-result, preregistered for first pilot. Runs only on explicit
PR/dispatch; does not bundle any third-party documents or model weights.

## Hypothesis and scope

The O(kd) cluster-local residual revival showed a possible advantage on 96D
LSA document vectors at low expected scan fractions. We now test it on
**384-dimensional pretrained neural sentence embeddings**, using two
independent public corpora, rather than retuning against the old data.

Model: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
Apache-2.0. Record exact upstream SHA from Hugging Face and load the
immutable SHA. Produce L2-normalized float32 embeddings; no finetuning.
*Embedding construction time is excluded from the reported clustering
fit time and separately recorded.*

Corpora:
1. [PolyAI/BANKING77](https://huggingface.co/datasets/PolyAI/banking77),
   CC BY 4.0, official training/test splits; metadata records dataset SHA.
2. The scikit-learn [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)
   corpus, official train/test split with headers/footers/quotes removed;
   underlying source terms must be checked before any public redistribution.
   No raw text is included in this repository or an Actions artifact.

Duplicate/empty documents are excluded, training and queries are
disjoint, and each run records SHA-256 of both source-text selections and
resulting neural embeddings. Initial pilot: 1600 training vectors,
160 query vectors, k=64, 2 passes for Sven, 2 dataset seeds and 3
optimizer seeds, exact exhaustive query top-10 ground truth.

## Four arms, fixed in advance

- `sven_global`: existing shipped default, 2 epochs.
- `sven_local`: proposed residual direction with 4-batch observation
  cadence, 2 epochs (other settings unchanged).
- `faiss_kmeans_2`: native FAISS k-means, 2 Lloyd iterations.
- `faiss_kmeans_20`: conventional stronger 20-iteration reference
  (**not an equal-compute competitor**).

All centroids are injected directly into *the same*
`faiss.IndexIVFFlat(faiss.IndexFlatL2)` without FAISS retraining.
Train vectors are inserted with fixed ID order and no compression.
Compare recall@10 directly to `faiss.IndexFlatL2` exhaustive search.
Queries are fixed across all arms and no trained centroids see query
embeddings. Index full-list recall **must** be 100%, enforced by
an assertion.

For each nprobe=1/2/4/8/16/32/64 we report **actual FAISS search
recall**, median CPU query time over 3 warmup-separated repetitions,
inverted-list size and mean/max candidates scanned. Secondary metrics:
held-out nearest-center squared error, clustering fit milliseconds, and
split count. Equal expected scan operating points at 5/10/15/25% use
randomized nprobe interpolation; this is not a strict per-query cap.

Training costs are not normalized by algorithm iteration alone. Report
wall-clock *and* epochs/iterations, and do not claim comparisons with
Meta e-means or A100/H100 until actually run.

## Release and interpretation thresholds

This is a new, **unseen corpus** test; its first observed result must be
reported whether negative or positive. No parameter tuning against the
pilot datasets is allowed without separating and disclosing exploratory
versus final tests. Promotion to default requires a Pareto improvement
across independent corpora with no material recall or MSE regression,
including lower-search-budget performance and actual search latency.

This pilot is still CPU-only, small-scale and exploratory. A legitimate
paper requires stronger confidence intervals over independent corpora,
larger k and vector counts, multiple pretrained encoders, and comparison
to Meta e-means at equal compute/quality budgets.

Reproduction:
```bash
pip install -e ".[dev]"
pip install -r benchmarks/requirements-neural.txt
pytest -q tests/test_neural_faiss.py
python -m benchmarks.neural_faiss --dataset banking77 --output /tmp/banking.json
python -m benchmarks.neural_faiss --dataset 20news --output /tmp/news.json
```
