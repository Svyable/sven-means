# Experiment 006 — Rare-cluster protection × local residual geometry

**Status:** experimental, CPU-only, opt-in and not claimed as original.

## Mandatory prior art

Eyal Weiss, *Small Is Not Dead: Distance-Aware Cluster Revival for e-means*,
arXiv:2611.18238 (submitted 2026-11-13),
https://arxiv.org/abs/2611.18238, already proposed and evaluated
distance-aware, age-gated protection against recycling low-usage centroids.
Our use of the distance criterion, nearest-center test, threshold=4,
and two-epoch abandonment is **an attributed adaptation, not a new method**.
The paper also points to earlier k-means-u utility concepts.

The new experiment asks only whether applying this independently published
protection rule to **Sven's local rank-one residual split proposal**
produces a useful interaction on data with rare, isolated groups.

## Rule

A low-usage centroid is an eligible removal target if and only if

```
  closest_other_centroid_L2_squared <= 4 * running_MSE
  OR batches_since_last_observed >= 2 * ceil(n / batch_size)
```

The running distortion EMA uses the *same current alpha* as the
running cluster mass, so `sum(distortion)/sum(usage)` estimates the
current squared error scale. After splitting, both children inherit
half the mass and distortion, and their last-seen steps reset.

Nearest-other-centroid distance is exact and computed with direct
coordinate differences in bounded row/column tiles; the algorithm
takes O(number_of_candidates * k * d) at each revival check and
therefore **does not scale to millions of centroids without an ANN
nearest-center search**. This is a CPU proof of concept only;
GPU requests fail explicitly.

## Preregistered four-arm ablation

1. `global`: Sven current unprotected global donor split (control).
2. `global_protected`: original global split with Weiss protection.
3. `local`: Sven rank-one local-direction split, without protection.
4. `local_protected`: **combination** of local direction + Weiss protection.

The dataset and model seeds, k, initialization, and epochs are paired.
Use two independent data draws × three optimizer seeds per dataset.

Development workloads: **rare isolated Gaussian islands**, skewed Zipf
synthetic data, real standardized handwritten digits and breast-cancer
feature vectors, and real 20 Newsgroups train-fitted LSA document vectors.
The rare-island generator explicitly supplies very few far-away points
per rare group. This is a targeted stress test, **not** independent
neural-embedding evidence or a reproduction of Weiss's dataset.

Outcomes:
- Held-out centroid MSE and split count
- Number of low-usage candidates protected at revival checks (counts
  repeated decisions, not distinct clusters)
- Approximate ideal coarse-IVF candidate recall@10 at equal **expected**
  5%, 10%, and 25% vector-scan budgets
- CPU training latency ratio and source/dataset SHA-256

## Criteria

A feature can be worthwhile even if it protects a rare group but lowers
average recall; publish the tradeoff. Promotion to the default requires
consistent multi-corpus improvements, and substantially better scaling of
candidate-nearest-center search. No invented H100, Meta-e-means, or
billion-scale performance claims.

The new [Experiment 005](neural-faiss.md) separately tests *unseen*
384D pretrained neural embeddings and a real FAISS IVF-Flat index.
**Do not tune the present experiment against those unseen results.**


## Pilot results — November 17, 2026 UTC

[GitHub Actions paired experiment 44020688391](https://github.com/Svyable/sven-means/actions/runs/44020688391) passed unit invariants and both offline/document datasets. Each method has two independently generated dataset draws and three training seeds (**6 paired fits per dataset**). All figures are *mean* holdout MSE improvement relative to Sven global unprotected (positive is better), not claims against Weiss's e-means implementation.

| Dataset | Global + distance protection | Local direction unprotected | Local + distance protection |
|---|---:|---:|---:|
| Rare isolated Gaussian islands | **+12.98%** | **−68.39%** | **+11.95%** |
| Skewed Zipf mixture | +0.00% | +0.17% | +0.17% |
| Real digits standardized | +0.03% | −0.92% | +0.74% |
| Real breast-cancer standardized | −0.39% | −1.62% | −1.43% |
| Real 20 Newsgroups LSA vectors | +0.00% | +0.99% | +0.99% |

**Interpretation:** on the deliberately rare-island workload, local splitting without protection is disastrous on average, but the distance guard eliminates most of that collapse. Nonetheless **global + distance guard has lower MSE than local + guard**: this is direct evidence *against* a claim that our direction innovation universally improves the published rare-protection method.

On the LSA document vectors, the distance guard never blocked a low-count centroid at any measured check (average number of guarded-candidate-check events = 0). Consequently the guard **did not improve accuracy** over the same unguarded method but increased wall-clock time: global protected ~1.09× global, local protected ~1.40× global versus unguarded local ~1.27×. This implementation uses an exact O(u k d) nearest-centroid search and has not attained Weiss's near-zero overhead on its own implementation.

The real breast-cancer dataset still regressed: −0.39% MSE for protected global and −1.43% for protected local. On rare islands the measured improvement is **not novel**; it corroborates the published mechanism on a different s-means implementation and dataset generator.

### Probes and uncertainty

- The six fits per dataset are **not six independent corpora**; there are two source seeds. Large point estimates on one constructed rare-island task are insufficient for a broad paper claim.
- The evaluation also archives ideal candidate recall at equal expected 5%, 10%, 25% vector-scan budgets and split counts. This is not real FAISS index recall or search latency.
- The unguarded local failure and local+guard rescue suggest a **safety precondition** for experimental local revival. But there is not enough real-world evidence or runtime efficiency to make the guard a default.
- Independent 384D pretrained embeddings and *actual* FAISS IVF search are being evaluated separately in Experiment 005. Do not tune to those test results.

**Decision:** leave the distance guard opt-in, clearly cite Weiss (2026), and do not assert an original breakthrough. To improve this combination, replace expensive exact small-candidate nearest-centroid computations, broaden real data and assess whether the guard avoids rare-class loss under actual workload distributions. A new contribution needs both an independent theoretical result and a reproducible quality–cost win over the published guard and our existing default.
