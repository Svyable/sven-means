# Experiment 002 — Split-aware final-pass statistics

**Protocol registered before observing results.** No results are embedded into
the training code. The changed behavior is disabled by default.

## Hypothesis

During the final epoch, s-means accumulates per-cluster weighted sums and
counts for empirical-Bayes finalization. If a centroid splits during that
same epoch, its *identity* changes, but its stored weighted mean still
represents its pre-split assignment region. The resulting finalization may
pull the new child center back toward an obsolete parent location.

Candidate change: after each split in the last epoch, zero the final-pass
sums and counts of both the donor and the revived target; retain every
other centroid's evidence. Apply the same rule on native CPU, native GPU,
and Triton GPU paths. **This need not be universally better**: it discards
historical observations and can increase variance when few batches remain.

## Comparison

Control: `SMeansConfig(reset_pass_on_split=False)`.  
Candidate: `SMeansConfig(reset_pass_on_split=True)`.

Factor 2: global and local-direction revival (local mode remains CPU-only).
All other configuration, initialization seeds, data splits, and epochs are
identical. The CPU benchmark uses two independent dataset seeds and four
model seeds per data seed, for eight paired comparisons per revival policy.

- Synthetic rotated cluster mixture, 96D, 4,000 training rows.
- Synthetic skewed-frequency mixture, 96D, 4,000 training rows.
- Real scikit-learn handwritten digits (64 features), standardized using
  train-only statistics and split 75/25.
- Real scikit-learn breast-cancer observations (30 features), likewise
  train-only standardized.
- Real 20 Newsgroups text, downloaded from scikit-learn, transformed to
  96D LSA-style document vectors using TF-IDF and TruncatedSVD **fitted
  only on the training text**. The independent held-out documents are
  never fitted. This corpus is not a modern pretrained neural embedding.
  A separate CI job downloads the corpus to avoid bundling copyrighted
  content in this Git repository.

Metrics:
1. Mean held-out squared Euclidean distance to closest centroid.
2. Fit runtime, paired over identical data and model seeds.
3. **IVF candidate recall@10** at 1, 4, 8 probed centroid lists, using 96
   held-out queries: fraction of exact top-10 training neighbors whose
   centroid assignments belong to one of the query's nprobe closest lists.
   This is *not* a measurement of a production ANN index's recall/latency.
4. Number of splits and still-dead centroids.

The script emits complete per-seed rows, dataset hashes, dataset provenance,
and runtime versions into JSON artifacts. No speed or accuracy numbers
should be asserted until the CI artifacts and logs have been inspected.

## Decision rules

Do **not** promote the candidate to default solely because a synthetic
rotated dataset improves. Require consistent gains across at least the
real digits, cancer, and document-vector workloads, no material loss of
candidate recall at matched nprobe, and no unexplained large latency
or GPU regression. Retain non-winning code only as an explicitly
experimental option and document negative outcomes.

A second phase must test 10+ independent source datasets, higher-dimensional
pretrained embedding vectors, and a real FAISS IVF index with time-budget
and memory Pareto curves before framing a broader public claim.


## Stage A observed results: reset BOTH donor and revived target

**Source:** [GitHub Actions paired self-improvement run 43999981232](https://github.com/Svyable/sven-means/actions/runs/43999981232). Each dataset has 2 independent training/test splits and 4 optimizer seeds, giving **8 paired trials** per revival policy. All figures below are mean changes against the current *same-geometry* carry-through control, not comparisons to Meta e-means.

A negative MSE gain indicates worse held-out quantization error. Recall gains are **percentage points** in ideal coarse IVF candidate recall@10 with four centroid lists probed, not end-to-end FAISS recall/latency.

| Dataset | Global MSE gain (%) | Global recall gain (points) | Local MSE gain (%) | Local recall gain (points) |
|---|---:|---:|---:|---:|
| Synthetic rotated | −0.173 | +0.013 | −0.001 | +0.000 |
| Synthetic Zipf | −1.023 | +11.823 | −0.061 | −0.651 |
| Real digits (train-only standardized) | −0.042 | +0.013 | −0.102 | −0.013 |
| Real breast-cancer observations | −0.267 | +0.417 | −0.005 | +0.013 |
| Real 20 Newsgroups TF-IDF/SVD vectors | −0.093 | +1.224 | +0.009 | +0.078 |

**Decision:** the original hypothesis that resetting both indices improves mean held-out MSE is **not supported**. Global reset lost MSE on all five test families, with zero of eight MSE wins for both the rotated and Zipf cases. The Zipf candidate-recall increase is striking but is specific to a synthetic skewed distribution; the document-vector improvement is much smaller. Retain the policy **opt-in only**, do not claim a general accuracy win, and do not publish a production release of this experiment without GPU validation.

**New hypothesis generated after observing Stage A (must be tested separately):** resetting the donor discards legitimate final-pass history while resetting the revived target may remove genuinely stale evidence. Compare `target_only` against both carry-through and reset-both, on the same paired protocol, without changing the default. This is a post-hoc follow-up and **not** a preregistered Stage A outcome.

### Additional limitations

- The “20 Newsgroups embeddings” are 96D latent semantic analysis vectors, **not pretrained neural embeddings**.
- Candidate recall measures a coarse assignment property with exact baseline neighbors and fixed probe count; it does not account for index build time, scan cost, actual FAISS quantization configuration, or query latency.
- Numeric CPU timing depends on shared GitHub runners; full records, data hashes and environment versions are retained as workflow artifacts.
