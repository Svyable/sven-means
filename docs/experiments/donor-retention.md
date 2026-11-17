# Experiment 003 — Donor-history retention after splits

**Status:** post-hoc hypothesis registered after Stage A; exploratory, not
a broad claim of superior clustering quality. Main/default remains unchanged.

## What Stage A demonstrated

The [split pass invalidation study](split-pass-invalidation.md) showed that
resetting **both** donor and target final-pass statistics worsened mean
held-out MSE on all five tested families. Yet coarse IVF candidate recall
at `nprobe=4` improved by +11.823 percentage points for synthetic Zipf data
and +1.224 points for LSA-style document vectors. The same study reported
almost no runtime cost of clearing both arrays.

We now test whether **the donor's legitimate historical evidence should
survive a split**, while the revived target has its stale statistics cleared.

## Proposed strategy

For a split into `source` and `target` in the final epoch, let `H_s`
denote the source's accumulated weighted sum/count pair. Replace it by
`lambda * H_s`, while replacing the target's sum/count by zero. The
empirical-Bayes finalizer then combines those updated statistics with
the centroid's tracked online position. The source and target history
cannot be mixed accidentally.

The candidate retention factors are **chosen before Stage B measurement**:
- `carry`: no reset at all (current default).
- `both`: retention 0; reset both (reproduce Stage A).
- `half`: retention 0.5; preserve half of donor evidence and reset target.
- `target_only`: retention 1; preserve all donor evidence and reset target.

This is *not* equivalent to a valid reassignment of old source points to
the two new centers. It is a heuristic shrinkage of stale evidence.
The conditional SSE split-gain identity used elsewhere in this research
does not prove it is optimal.

## Paired study

Use the same five data families, seeds, `k=64`, two epochs and train-only
preprocessing as Stage A. The benchmark runs four methods in rotated
order, matching the exact train/test and model seed per pair. Each
dataset has two data seeds × four model seeds = **8 paired comparisons**
against the current `carry` control.

Metrics: held-out test MSE, IVF coarse *candidate* recall@10 at
`nprobe=1/4/8`, cluster occupancy coefficient of variation, fit latency,
and number of split/revived centroids. All raw rows and provenance are
published as CI artifacts. The second job obtains real 20 Newsgroups
text and embeds it with train-only TF-IDF and TruncatedSVD, **not
a pretrained embedding model**.

**Decision rule:** do not promote solely for a synthetic Zipf candidate-recall
win. For a Pareto improvement, require no consistent real-dataset MSE
regression, meaningful recall improvement across the real datasets at
fixed nprobe, and no material latency regression. If those conditions
are not met, retain this as a research knob with a negative result.

Run:
```bash
python benchmarks/donor_retention_study.py --dataset all-offline --output /tmp/retention-offline.json
python benchmarks/donor_retention_study.py --dataset news_lsa --output /tmp/retention-news.json
```

## Further validation required

Actual FAISS IVF retrieval and scan cost, pretrained embedding datasets,
independent source distributions, GPU correctness with TensorCore/Triton
and equal wall-clock comparisons. The candidate recall metric is
not an end-to-end ANN benchmark.
