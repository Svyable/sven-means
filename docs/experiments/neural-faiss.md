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


## Initial pilot findings (GitHub Actions 44020029870, 2026-11-17 UTC)

The first unseen-embedding run **completed all three CI jobs successfully**:
synthetic FAISS invariant tests, independent BANKING77 and independent 20
Newsgroups neural embeddings. [CI and complete raw JSON artifacts](https://github.com/Svyable/sven-means/actions/runs/44020029870).

This is a CPU pilot, *not* a breakthrough claim against FAISS/Meta e-means.
The model SHA used in that run is
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (Apache-2.0);
BANKING77 dataset SHA is
`1846984189f2bc0ac51c6fc3a600fa9e1714e0c6` (CC BY 4.0).
Both revisions are now frozen in the benchmark source rather than
re-resolved from live `main`. Vectors have 384 dimensions and L2-unit norm.

Results below compare the *same 2-epoch Sven local* against the
*same 2-epoch Sven global default*; each corpus has two independently
drawn document subsets, each evaluated with three centroid seeds
(**six paired fits per corpus**, not six independent corpora).
Positive MSE improvement means smaller squared error; positive FAISS
recall change means more exact top-10 neighbors found by real IVF-Flat
search at the *same expected fraction of candidate vectors scanned*.

| Corpus | Sven local MSE improvement | Real FAISS recall@10 gain at 10% expected scan | Local/global CPU fit time |
|---|---:|---:|---:|
| 20 Newsgroups, pretrained 384D vectors | **+3.86%** (6/6 wins) | **+13.71 percentage points** (6/6 wins) | ~1.18× |
| BANKING77, pretrained 384D vectors | **−0.43%** (2/6 wins) | **−0.24 percentage points** | ~1.18× |

**20 Newsgroups cannot be compared at 5% expected scans** in the chosen
full-list nprobe grid: the global baseline's smallest observable mean
scan fraction is ~6–7%. Values below that are **unavailable, not zero**.

### A much stronger classical baseline

On *20 Newsgroups*, the native FAISS **two-iteration** k-means arm
improved held-out MSE by ~4.15% and scan-matched recall@10 by ~18.12
percentage points relative to Sven global, while training in ~0.18×
the CPU time. FAISS's 20-iteration arm yielded even lower MSE and still
higher recall, at a different compute budget. Thus Sven local is **not**
the best method here, despite improving substantially over Sven global.

On BANKING77, Sven global had lower MSE than both two-iteration and
twenty-iteration FAISS k-means on average; Sven local did not
consistently improve it. Report this cross-domain reversal rather
than claiming universal superiority.

### Mechanism worth testing, not yet established causally

In the 20 Newsgroups first data split, the mean fraction of training
vectors scanned by **one** probed IVF list was ~6.0% for Sven global,
~3.1% for Sven local, and ~1.9% for FAISS's two-iteration k-means.
The second split showed ~7.3%, ~3.1%, and ~2.2% respectively.
The large recall improvement at fixed scan work could thus partly
reflect a **more balanced inverted-list occupancy distribution**
rather than a better quantization direction by itself.

These pilot results motivate a separate cluster-occupancy diagnostic
with list-size dispersion, underfilled/overfull groups and scan
Pareto curves. Do not mistake *fixed nprobe* for fixed search work.
Do not tune parameters on these corpus splits and then report them
as fresh independent test data.

### Required to justify publication

- Compare with pinned Meta `faiss.contrib.e_means` at matched
  quality/time and with fixed seeding, not only FAISS Lloyd k-means.
- Expand to multiple independent pretrained encoders/corpora and
  at least 10 independent data splits.
- Test larger k, dataset sizes and actual GPU Triton/CUDA hardware,
  with memory peaks and full search latency/recall curves.
- Distinguish observations borrowed from Eyal Weiss (2026)
  distance-aware revival work, cited in Experiment 006, from any
  future independently validated contribution.
