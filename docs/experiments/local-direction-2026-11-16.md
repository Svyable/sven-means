# Experiment 001 — local residual revival

**Date:** November 16, 2026 (CI completed November 17 UTC)
**Source:** [paired CI study and downloadable per-seed JSON](https://github.com/Svyable/sven-means/actions/runs/43986118087)
**Status:** exploratory; the existing global-revival policy remains the default.

## Design

Two epochs; five model seeds; 48 centroids; synthetic datasets with 6,000 training samples and independent held-out draws; a 75/25 train/test split of sklearn's real handwritten digits, with train-only scaling in the standardized variant. Exactly matched training data, initializer, epoch budget, and seed for global versus local revival. Positive gain means **lower held-out test MSE** for local revival.

## Paired mean held-out MSE improvement (%)

| Workload | Every batch / RMS | Every 4 / RMS | Every 4 / mean-abs | Every 8 / mean-abs |
|---|---:|---:|---:|---:|
| Rotated elongated | +12.55 | +11.86 | +11.86 | +8.96 |
| Isotropic | +0.16 | +0.12 | +0.09 | +0.12 |
| Zipf | −0.03 | −0.06 | +0.02 | +0.11 |
| Shifted | +0.05 | +0.07 | +0.10 | +0.12 |
| Digits raw | −0.32 | −0.42 | −0.40 | −1.15 |
| Digits standardized | +1.14 | +0.89 | +0.86 | +0.85 |

## Median fit time (local / global)

| Workload | Every batch / RMS | Every 4 / RMS | Every 4 / mean-abs | Every 8 / mean-abs |
|---|---:|---:|---:|---:|
| Rotated | 1.88× | 1.23× | 1.22× | 1.13× |
| Isotropic | 1.90× | 1.21× | 1.20× | 1.11× |
| Zipf | 1.84× | 1.24× | 1.24× | 1.13× |
| Shifted | 1.86× | 1.23× | 1.24× | 1.10× |
| Digits raw | 1.89× | 1.25× | 1.23× | 1.12× |
| Digits standardized | 1.87× | 1.25× | 1.24× | 1.12× |

## Decisions

- **No default promotion:** real-world performance is near neutral or mixed, including a small regression on raw digits. The large rotated gain is a useful diagnostic on a synthetic workload intentionally chosen to expose global-direction weaknesses.
- **Experimental observation stride defaults to four batches**: rotated gain is +11.86% at ~23% CPU runtime overhead, versus +12.55% with ~88% overhead every batch.
- **Keep `rms_scaled` as the experimental radius default.** The mathematically conditional-optimal `mean_abs` radius did not yield meaningful generalization gains here.
- **CPU-only.** No GPU/Triton implementation or billion-vector claim.
- All benchmark JSON is an Actions artifact and the script can regenerate it.

**Threats to validity:** five seeds over one dataset draw per synthetic family; modest dataset sizes; GitHub-hosted shared runner timing variability; mean improvements without confidence intervals; no FAISS retrieval recall, no high-dimensional real embeddings. The split-gain identity is exact only for fixed directions and assignments, not for the changing online model.

**Next falsification test:** multiple independent embedding datasets with local correlated geometry, ablations at equal wall-clock budgets, then downstream IVF/FAISS recall, latency, and memory analysis. Do not substitute this study for production validation.
