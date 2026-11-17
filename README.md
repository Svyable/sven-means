# sven-means

An experimental, MIT-licensed refactor of [s-means](https://github.com/suayptalha/s-means) in the `svenmeans` Python namespace. s-means builds on Meta FAIR's [e-means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means). **This is a derivative research implementation, not a claim that we have beaten either upstream on independent benchmarks.**

## What's different in this branch

1. **Bounded-workspace initialization.** The original subsampled distance-weighted initializer materializes `m × candidate_centroids` distances (with `m≈4k`). We tile centroid columns (default 128) and cap the coreset (default 16,384, never below `k`). Peak *distance-matrix* workspace is `O(m · tile_cols)` instead of `O(mk)`. The arithmetic is still `O(mkd)` and the coreset itself occupies `O(md)`; huge cluster counts still require distributed/sharded engineering.
2. **Avoid full-`n` permutation for small coresets.** When `m << n`, we sample and deduplicate a bounded index pool instead of building `randperm(n)`. A random permutation of the distinct pool prevents a bias toward lower row indices. Selected initial centroids are unique **row indices**, including for repeated data values.
3. **Numerically robust initialization and inference.** Distance calculations use a shared coordinate anchor before the squared-norm identity to reduce floating-point cancellation. `SMeansConfig(center_data=True)` optionally centers the full fit, with explicit extra `O(nd)` storage and preparation time.
4. **Centroids-only CPU fitting.** `smeans(..., compute_labels=False, return_info=True)` avoids the final `O(nkd)` assignment pass and correctly returns `labels=None`, `inertia=None`, `mse=None`.
5. **On-demand shuffled epochs (experimental).** Choose `shuffle_mode="affine"` (cheap modular bijection) or `"feistel"` (stronger mixing through six Feistel rounds and cycle-walking) to avoid `randperm(n)`. Both use `O(batch)` index memory; neither is a uniform random permutation. The default remains `"randperm"`. See [the design note](docs/feistel-shuffle.md).
6. **Preserved core clustering algorithm.** Distortion-aware revival, cosine decay, and final-pass refinement remain upstream s-means mechanisms. Their accuracy benefits are not newly claimed here.

## Install and use

```bash
python -m pip install -e ".[dev]"
pytest -q
```

```python
import torch
from svenmeans import SMeansConfig, smeans

x = torch.randn(20_000, 64)
result = smeans(
    x, n_clusters=256, nepochs=3, device="cpu",
    config=SMeansConfig(init_max_samples=8192, init_tile_cols=64,
                        shuffle_mode="affine"),  # opt-in, benchmark before use
    compute_labels=False, return_info=True,
)
centers = result.centroids
assert result.labels is None

# Opt in when coordinates have a large common offset:
stable = smeans(x + 100_000, 256, device="cpu",
                config=SMeansConfig(center_data=True))
```

A scikit-learn-compatible estimator is also available: `from svenmeans import SMeans`. CUDA/Triton code is inherited and adapted to use bounded initialization; CUDA performance and correctness require independent GPU CI and benchmarks before production use.

## Reproduce rather than believe

```bash
python benchmarks/compare.py --kind gaussian --n 10000 --d 32 --k 128 --seeds 3
python benchmarks/compare.py --kind shifted --n 10000 --d 32 --k 128 --center-data --seeds 3
```

To compare against the exact original s-means Python package, separately install upstream (distribution `s-means`, import `smeans`) and add `--baseline`. The runner uses *identical generated data and held-out evaluation* across methods, records elapsed fit times and test MSE, and emits JSON. It does not quietly attribute upstream's published results to this fork.

## Known limitations

- Affine and Feistel shuffling are bijections, **not uniformly random shuffles**. Both are experimental. Affine can introduce ordering patterns; Feistel can add costly cycle-walking synchronizations, especially on GPUs.
- Tiling reduces memory, **not the `O(mkd)` seeding arithmetic**. Large `k` needs approximate, hierarchical, or sharded initialization.
- Full-data centering costs an extra tensor copy, can affect runtime, and is disabled by default.
- This first refactor does **not** add a new cluster-splitting heuristic: that requires controlled evaluation on real embeddings, not just synthetic distributions.
- The optional CUDA/Triton route needs its own reproducible multi-GPU test matrix. No H100/billion-vector claims are made here.
- The original implementation and its authors retain attribution; see [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).

## Experimental cluster-local revival (CPU only)

`SMeansConfig(revive_geometry="local_direction")` enables an alternative
to the default global-variance perturbation. For each cluster, it streams
a *single* covariance-vector product `E[r (r·q)]` along with projected
absolute and squared residual moments, using `O(kd)` persistent state.
The proposal `c ± ρq` is chosen using the conditional SSE-gain proxy
`usage × max(0, 2ρ E|r·q| - ρ²)`. A cold/uninformative cluster is not
split under this policy.

The experimental sketch defaults to one update every four batches, based
on paired CPU ablations; the analytic mean-absolute radius is available
but did not consistently outperform RMS scaling. See the
[measured experiment log](docs/experiments/local-direction-2026-11-16.md).

This is an **experiment, not a recommended default**. The gain proxy is
in-sample, based on changing centroids and directions; it does not prove a
held-out improvement. No GPU/Triton implementation is claimed: requesting
this mode on CUDA raises `NotImplementedError`. See
[the mathematical design and limitations](docs/local-residual-revival.md)
and run `python benchmarks/revival_study.py --kind all --seeds 5`.


Run comparisons against upstream s-means and Meta e-means at equal wall clock and epochs; add real embedding/FAISS IVF recall benchmarks; explore localized residual-covariance splitting with ablations; establish GPU regression CI; only promote ideas that improve held-out quality without surprising memory/latency costs.
