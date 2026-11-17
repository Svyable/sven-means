# Experimental rank-one, gain-gated revival

**Status:** opt-in and CPU-only. The established global-variance algorithm
remains the default. This is not a claim of better test MSE.

## Motivation

The upstream s-means split direction samples a global, feature-weighted
Rademacher vector for a selected high-distortion donor. That direction does
not know a cluster's *individual* residual covariance. Two clusters may
have identical per-feature spreads yet sharply different orientations.

Storing the full covariance matrix for every cluster costs
`O(k d²)`. Instead, each centroid tracks one normalized vector `q_j`
and two projected residual moments, at `O(kd)` storage.

For assigned point `x_i` and centroid `c_j`, let `r_i=x_i-c_j`
and `z_i=r_i·q_j`. With a batch of assignments we estimate

`g_j = E[r_i z_i | cluster j] = CovResidual_j q_j`.

The update normalizes `g_j` and blends it with the previous `q_j`.
This is a streaming, rank-one power-iteration approximation to the
leading direction of the (uncentered) local residual second moment,
not a full PCA decomposition. The vector can be noisy or stale.

## Exact conditional splitting identity

For a fixed unit vector `q` and radius `ρ >= 0`:

`min(||r - ρq||², ||r + ρq||²) = ||r||² + ρ² - 2ρ|r·q|`.

Therefore the change in donor-assigned SSE is

`gain_i = 2ρ|r_i·q| - ρ²`.

We maintain EMAs of `E|z|` and `E[z²]`; propose
`ρ = split_mult × sqrt(E[z²])` (default multiplier 0.55).
A donor is eligible only after at least
`local_min_observations` assignments and when the estimated
per-point gain exceeds `local_gain_floor`. Donors are ranked by
`usage × max(0, estimated per-point gain)`.

**This estimate is not a guaranteed improvement.** The labels, centers,
and vector all change during training; a local gain on old assignments
can be irrelevant to current or held-out points. Rejecting splits can
also leave dead centroids unrevived. After splitting we preserve the
direction but reset projected moment history for both children.

## Performance considerations

Each batch adds an `O(batch*d)` residual-projection operation and
`O(k*d)` scratch reduction. The existing global centroid-update logic
is unchanged for the default. GPU support is deliberately unavailable
until CUDA memory, kernel launch, and assignment correctness can be
tested on hardware. No claimed reduction in runtime or total allocation.

## How to falsify

```bash
python benchmarks/revival_study.py --kind all --n 6000 --d 32 --k 48 --epochs 2 --seeds 5 --output /tmp/revival.json
```

The study uses paired seeds and identical held-out data for both modes,
including rotated elongated clusters, isotropic clusters, Zipf weights,
large-coordinate offsets and two variants of the real digits dataset.
Evaluate wins **and** losses, median runtime ratio, split counts, and
whether the method actually revives dead clusters.

If the effect is absent or negative on these workloads, keep this as an
explicit experiment and reconsider its geometry, score, and early
stopping before attempting a GPU port. If promising, add independent
real embedding evaluations (FAISS/IVF recall, not MSE alone).
