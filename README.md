# sven-means

An experimental, MIT-licensed refactor of [s-means](https://github.com/suayptalha/s-means) in the `svenmeans` Python namespace. s-means builds on Meta FAIR's [e-means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means). **This is a derivative research implementation, not a claim that we have beaten either upstream on independent benchmarks.**

## What's different in this branch

1. **Bounded-workspace initialization.** The original subsampled distance-weighted initializer materializes `m × candidate_centroids` distances (with `m≈4k`). We tile centroid columns (default 128) and cap the coreset (default 16,384, never below `k`). Peak *distance-matrix* workspace is `O(m · tile_cols)` instead of `O(mk)`. The arithmetic is still `O(mkd)` and the coreset itself occupies `O(md)`; huge cluster counts still require distributed/sharded engineering.
2. **Avoid full-`n` permutation for small coresets.** When `m << n`, we sample and deduplicate a bounded index pool instead of building `randperm(n)`. A random permutation of the distinct pool prevents a bias toward lower row indices. Selected initial centroids are unique **row indices**, including for repeated data values.
3. **Numerically robust initialization and inference.** Distance calculations use a shared coordinate anchor before the squared-norm identity to reduce floating-point cancellation. `SMeansConfig(center_data=True)` optionally centers the full fit, with explicit extra `O(nd)` storage and preparation time.
4. **Centroids-only CPU fitting.** `smeans(..., compute_labels=False, return_info=True)` avoids the final `O(nkd)` assignment pass and correctly returns `labels=None`, `inertia=None`, `mse=None`.
5. **On-demand shuffled epochs (experimental).** `SMeansConfig(shuffle_mode="feistel")` uses a six-round Feistel permutation plus cycle-walking to visit every row exactly once without `randperm(n)`. This uses `O(batch)` index memory; it does **not** give a uniform random permutation and may trade speed or convergence behavior for memory. See [the design note](docs/feistel-shuffle.md).
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
                        shuffle_mode="feistel"),  # opt-in, benchmark before use
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

- Feistel shuffling is a bijection, **not a uniformly random shuffle**. It is a research option and may change clustering quality or add synchronizations on GPU.
- Tiling reduces memory, **not the `O(mkd)` seeding arithmetic**. Large `k` needs approximate, hierarchical, or sharded initialization.
- Full-data centering costs an extra tensor copy, can affect runtime, and is disabled by default.
- This first refactor does **not** add a new cluster-splitting heuristic: that requires controlled evaluation on real embeddings, not just synthetic distributions.
- The optional CUDA/Triton route needs its own reproducible multi-GPU test matrix. No H100/billion-vector claims are made here.
- The original implementation and its authors retain attribution; see [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).

## Research plan

Run comparisons against upstream s-means and Meta e-means at equal wall clock and epochs; add real embedding/FAISS IVF recall benchmarks; explore localized residual-covariance splitting with ablations; establish GPU regression CI; only promote ideas that improve held-out quality without surprising memory/latency costs.
