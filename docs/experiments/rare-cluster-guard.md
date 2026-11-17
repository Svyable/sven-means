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
