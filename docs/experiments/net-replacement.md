# Experiment 011 — independent net replacement-cost gate (Issue #16)

**Status:** preregistered CPU-only exploratory feature; default unchanged.

A positive *donor-only* split score does not imply replacing an unrelated
low-count centroid helps globally. Recycling a rare isolated centroid
can destroy its group and increase the *full k-means* objective
dramatically. That mechanism motivated the independently published
[Eyal Weiss 2026 distance-aware guard](https://arxiv.org/abs/2611.18238).
Net replacement SSE and split–merge relocation itself are established
prior art (e.g. Hartigan-style moves). This is NOT a claim to have
invented full-objective acceptance.

## Design and what is actually independent

For `SMeansConfig(revive_acceptance="net_holdout")`, deterministically
randomly split the user's input before seeding:
- training-only: 90% of input, exclusively used by initialization,
  centroid EMA, donor proposals and final pass;
- reserved validation: 10% of input, **never used for centroid fit or
  donor proposal**. At least 32 rows are reserved.

Controls without a net gate are trained on **exactly the same
fit-only training subset** (not all original rows) for the benchmark.
All variants are evaluated on a completely separate held-out
test split. This is crucial: comparing a 90%-train net variant against
100%-train control would be confounded.

At each dead-cluster revival event, generate the same proposals as
the selected geometry mode, then consider at most four proposed
source/target pairs in deterministic proposal order. For each pair,
evaluate the row-level loss *before and after deleting BOTH old
source and old target centers and installing source +/- delta*:

```
  g_i = min_j ||x_i - c_j||²
      - min_j ||x_i - c'_j||²
  LCB = mean_i(g_i) - 1.645 * sample_std(g_i)/sqrt(m)
  accept iff LCB > 0
```

This is a one-sided *normal-inspired heuristic* threshold,
**NOT** a calibrated 95%-confidence guarantee. Validation rows are
reused across updates and proposals; even though held out of fit,
repeated adaptive decisions can overfit them. Heavy tails and
non-normal gains further break nominal interpretation. Report
the number of tested, accepted and rejected proposals, the
holdout size, test MSE, fit time and candidate recall.

## Efficient exact net-evaluator identity

For one proposal replacing two centers s and t, the nearest unaffected
center is guaranteed to be among the three closest *old* centers. Cache
their exact squared distances and indices on reserved validation.
Then for each validation row:

```
 new_best = min( nearest(old centers excluding s,t),
                 ||x-(c_s + delta)||², ||x-(c_s - delta)||² )
```

This is **exact** for any k>=2 (pad absent third center with infinity),
even if the source and target are the two closest; top-two alone is not
enough. Tests compare to fully recomputed k-center assignment
with duplicates, k=2/3, large global shifts and rare-island samples.

A full top-three computation costs O(m*k*d) once per check and after
each accepted proposal; each proposal evaluation costs O(m*d), scratch
O(m + tile_rows*tile_centers*d). The reserved holdout incurs an
O(m*d) data copy and the fit uses fewer rows; no scalable GPU
implementation yet. All these costs are included in CPU wall time.

## Preregistered paired comparison

Fixed default 10% holdout, LCB z=1.645, at most 4 proposals per check,
k=64, two fit epochs, local geometry uses historical covariance/RMS
with four-batch sketch cadence. Two source seeds × three model seeds.

Eight arms (two geometries x four gate modes):
- global (none, Weiss distance, net, Weiss+net);
- local directional (none, Weiss distance, net, Weiss+net).

Development sets: rare isolated Gaussian islands, Zipf skew mixture,
real digits, real breast-cancer features, 20 Newsgroups train-fitted LSA
vectors. These were **previously seen** and cannot be used for
confirmatory claims.

Metrics: test MSE, fit time including net-evaluation, split counts,
accepted/rejected proposal counts and ideal coarse IVF candidate
recall@10 at equal *expected* 5/10/25% database scanned. Candidate
proxy is not actual FAISS query latency. Save per-seed reproducible
source hashes and raw scan/recall curves in CI artifacts.

## Falsification criteria and next gate

Do not promote to default if net gate consistently rejects all
proposals while wasting data/time, harms real-data test MSE,
fails to preserve rare groups, or is not competitive with the
published Weiss control. If promising, freeze the selected approach
before trying on a **new** 384D pretrained corpus with actual FAISS
IVF-Flat, pinned Meta NumpyEmeans and FAISS Lloyd baselines.
Formal confidence coverage requires a separate statistical
procedure robust to sequential testing and adaptive reuse;
this first prototype does not provide it.

Neither split-merge net scoring, distance-aware protection nor
a basic Gaussian confidence bound is claimed original.
