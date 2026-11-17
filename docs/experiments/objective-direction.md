# Experiment 010 — objective-aligned direction for symmetric revival

**Status:** pre-result experimental CPU-only opt-in. All existing defaults
remain unchanged. The L1 principal-component gradient is **prior art**;
the proposed contribution to study is objective alignment in the
sven-means rank-one *revival* architecture, not inventing L1 PCA.

## Mathematical mismatch in the existing implementation

For residual r=x-c and a unit direction q, split into two symmetric
children c +/- rho*q. For a fixed set of donor assignments,

  min(||r-rho*q||², ||r+rho*q||²)
   = ||r||² + rho² - 2rho*|r.q|.

The *exact conditional expected* improvement is

  G(q,rho) = 2rho*E|r.q| - rho².

Optimizing over nonnegative rho yields rho*=E|r.q| and
G(q,rho*)=(E|r.q|)^2. So the direction should maximize
**E|r.q|**, a projected absolute-residual objective.

Our existing streaming direction update instead follows
E[r*(r.q)] = E[rr^T]q, the covariance power-iteration
direction maximizing **E[(r.q)²]**. This is only a surrogate:
a rare high-leverage residual can dominate the squared objective but
need not provide the best split gain.

At nonzero projection, the (sub)gradient of E|r.q| with respect to
q is **E[sign(r.q)*r]**. We add this streaming sign-gradient
variant in `DirectionalRevival`, with normalization and the same EMA
mixing as the covariance method, retaining the O(kd) state bound.
For `split_gain` only, projected absolute/squared moments are
recomputed along the **newly updated q** to avoid using stale
statistics in the proposed split radius. This costs an additional
O(batch*d) per sampled sketch update; wall time is measured.

The sign-gradient generalized-power idea belongs to earlier
**L1 PCA / robust subspace** research. We make no claim of a new
global optimizer, convex objective, guaranteed descent, or novel
L1-PCA theory. The same nonconvex and changing-assignment limitations
apply.

## Preregistered factorial ablation

All variants use the same data, seeds, seeding and two training epochs;
k=64, 4-batch sketch observation cadence, no Weiss distance guard.
We compare:

1. Sven global revival control.
2. Sven local **covariance direction + RMS-scaled radius**, existing
   default of the opt-in local branch (**primary research control**).
3. Local covariance direction + conditionally optimal mean-absolute radius.
4. Local **split-gain sign-gradient direction + RMS radius**.
5. Local **split-gain sign-gradient direction + mean-absolute radius**
   (**objective-aligned joint candidate**).

Two independent dataset seeds × four centroid seeds on previously used
*development* datasets: rotated and Zipf synthetic mixtures, real
standardized digits and breast-cancer features, rare-island stress
synthetic, and separate LSA document vectors.

Record held-out MSE, all split counts, CPU wall-clock fit time, exact
data-array SHA-256 and ideal IVF *candidate recall@10* at equal expected
5%,10%,25% scanned training vectors, not an actual FAISS index.
Any missing operating points must be labeled missing, not zero.

Unit tests directly verify that covariance and sign gradients point
in different directions on a controlled heavy-tailed residual batch,
and that the sign-gradient can improve the *actual* projected absolute
objective. Confirm end-to-end CPU determinism, zero-gradient stability,
sklearn exposure and old-default parity.

## Decision rule, failure modes, publication boundary

This was designed **after seeing** local revival's limited benefits and
some serious failures. These six datasets are *development workloads*,
not independent confirmatory evidence. A true novel algorithmic claim
would require a strong win against both our existing covariance mode
and *pinned Meta NumpyEmeans*, preferably on actual FAISS IVF recall
under equal expected scan fractions **and** matched wall-clock time.

A positive MSE or recall delta on one chosen distribution is
insufficient, and a real regression prevents default promotion.
If promising, freeze the chosen direction/radius strategy and run it
unchanged on at least one new, previously unused pretrained corpus,
including scale, k and GPU ablations. Clearly distinguish L1 PCA prior
art, Weiss (2026) rare-center protection, and Sven implementation.

Do not claim the sign-gradient is a globally optimal split direction,
that its EMA estimates are unbiased under nonstationary centroids, or
that one experimental synthetic result constitutes a paper.
