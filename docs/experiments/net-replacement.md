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


## Initial paired evidence — 2026-11-17 UTC

[GitHub Actions net-replacement ablation](https://github.com/Svyable/sven-means/actions/runs/44173959142) **passed the reserved-holdout and exact top-three full-objective tests** (18 passed, 1 CUDA skip), plus paired offline/real-document studies. The full JSON contains all per-seed source hashes, acceptance/rejection decisions, candidate-recall scan curves and CPU fit times.

**Interpret these as improvements over the SAME geometry unguarded arm, on the EXACT SAME fit-only 90% input.** Positive MSE gain means smaller independent test SSE. Each workload has two data draws × three optimizer seeds (six paired fits).

| Dataset | Global + Weiss distance | Global + net LCB | Global + both | Local + Weiss distance | Local + net LCB | Local + both |
|---|---:|---:|---:|---:|---:|---:|
| Rare isolated synthetic | +18.34% | **−24.73%** | −24.73% | **+36.96%** | +15.99% | +24.00% |
| Skewed Zipf | +0.00% | −0.79% | −0.79% | +0.00% | −0.19% | −0.19% |
| Real digits | +5.12% | +2.74% | +2.74% | +4.47% | +2.86% | +3.98% |
| Real breast-cancer features | +0.16% | −0.81% | −0.65% | +0.00% | −0.06% | −0.06% |
| 20 Newsgroups LSA | +0.00% | **−3.50%** | −3.50% | +0.00% | −0.13% | −0.13% |

Global net LCB rejected **40/40** proposed replacements in mean paired rare/Zipf/LSA runs, so forbidding all revivals is worse than either the default or Weiss's distance guard. The net LCB sometimes accepted many *local* proposals, but lacked a reliable independent test advantage and incurred major overhead.

Median CPU fit-time ratios relative to the same-geometry unguarded control:
- On rare islands: global net **2.07×**, local net **3.81×**; local net+distance **4.12×**.
- On LSA documents: global net **2.25×**, local net **2.70×**.
- On digits: global net **1.57×**, local net **1.44×**.

The 10% candidate-proxy recall generally followed mixed results:
- On LSA, global net lost **4.87 percentage points** and local net lost **0.41 points** relative to corresponding unguarded methods.
- On rare islands the net local arm had nearly unchanged candidate recall despite its large MSE improvement; rare isolated points are too rare to dominate average recall.

**Decision: no default promotion, no superiority claim.** The simple heldout-normal-LCB surrogate is often too conservative, loses useful recycling, and is costly. The specific numerical results **falsify a general quality/compute benefit** for this configuration. The published distance-aware rule remains the preferable rare-island protection on tested CPU workloads. This finding is limited to development datasets; no real 384D neural/FAISS validation was justified after these negative results.

### Research follow-up, not post-hoc tuning

Explaining *why* the net LCB frequently rejects all global proposals might require modeling the **future effect** of centroid relocation after subsequent training, rather than its instantaneous net SSE. Sequential reuse also undermines nominal confidence. Any new method must be registered before evaluation and compared at equal training data and wall clock. No post-hoc lowering of 1.645 or increasing proposal limits is presented as a confirmatory rescue.
