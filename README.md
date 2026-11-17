# sven-means

An experimental, MIT-licensed refactor of [s-means](https://github.com/suayptalha/s-means) in the `svenmeans` Python namespace. s-means builds on Meta FAIR's [e-means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means). **This is a derivative research implementation, not a claim that we have beaten either upstream on independent benchmarks.**

## What's different in sven-means

1. **Bounded-workspace initialization.** The original subsampled distance-weighted initializer materializes `m × candidate_centroids` distances (with `m≈4k`). We tile centroid columns (default 128) and cap the coreset (default 16,384, never below `k`). Peak *distance-matrix* workspace is `O(m · tile_cols)` instead of `O(mk)`. The arithmetic is still `O(mkd)` and the coreset itself occupies `O(md)`; huge cluster counts still require distributed/sharded engineering.
2. **Avoid full-`n` permutation for small coresets.** When `m << n`, we sample and deduplicate a bounded index pool instead of building `randperm(n)`. A random permutation of the distinct pool prevents a bias toward lower row indices. Selected initial centroids are unique **row indices**, including for repeated data values.
3. **Numerically robust initialization and inference.** Distance calculations use a shared coordinate anchor before the squared-norm identity to reduce floating-point cancellation. `SMeansConfig(center_data=True)` optionally centers the full fit, with explicit extra `O(nd)` storage and preparation time.
4. **Centroids-only CPU fitting.** `smeans(..., compute_labels=False, return_info=True)` avoids the final `O(nkd)` assignment pass and correctly returns `labels=None`, `inertia=None`, `mse=None`.
5. **On-demand shuffled epochs (experimental).** Choose `shuffle_mode="affine"` (cheap modular bijection) or `"feistel"` (stronger mixing through six Feistel rounds and cycle-walking) to avoid `randperm(n)`. Both use `O(batch)` index memory; neither is a uniform random permutation. The default remains `"randperm"`. See [the design note](docs/feistel-shuffle.md).
6. **Preserved default clustering algorithm.** Distortion-aware revival, cosine decay, and final-pass refinement remain upstream s-means mechanisms. A cluster-local residual-direction policy is separately available as a CPU-only experiment (below).

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

## Experimental split-aware finalization

During the final training pass, revived centroid indices can retain
assignment statistics accumulated **before their split**. The optional
`SMeansConfig(reset_pass_on_split=True)` invalidates only those stale
pass statistics at each split, so the final pass averages only subsequent
assignments for the changed centers. The unchanged algorithm remains
the default until the paired [self-improvement study](docs/experiments/split-pass-invalidation.md)
supports promotion. The option is wired to CPU, native CUDA, and Triton,
though GPU speed/accuracy requires separate hardware verification.

The release-candidate [benchmark](benchmarks/release_candidate.py) compares
both policies on identical data and seeds. It evaluates held-out MSE and
a limited IVF *candidate recall* proxy (not full FAISS search performance).

## Experimental donor-history retention

A follow-up to the split-statistics A/B study tests **fractional preservation
of the donor's final-pass evidence** while clearing the revived target's
statistics. `reset_pass_on_split=True, donor_pass_retention=0.5` retains
half of donor history; `1.0` retains all donor history (target-only reset);
`0.0` resets both (original experimental behavior). The default remains
`reset_pass_on_split=False`. See
[Experiment 003](docs/experiments/donor-retention.md) for the preregistered
comparison; it is not a generalized performance claim.

## Development experiment: train-only power-priced IVF routing

A separate [Experiment 007](docs/experiments/power-routing.md) tests
calibrated additive centroid prices for balancing inverted-list
occupancy while limiting extra *training assignment* distortion to
1%. The implementation uses an exact one-coordinate FAISS embedding
of a **known power diagram** and a bounded-error dual-price heuristic;
none of this mathematical prior art is claimed as Sven's invention.
Ordinary sven-means fitting, default centroids and routing are unchanged.

## Known limitations

- Affine and Feistel shuffling are bijections, **not uniformly random shuffles**. Both are experimental. Affine can introduce ordering patterns; Feistel can add costly cycle-walking synchronizations, especially on GPUs.
- Tiling reduces memory, **not the `O(mkd)` seeding arithmetic**. Large `k` needs approximate, hierarchical, or sharded initialization.
- Full-data centering costs an extra tensor copy, can affect runtime, and is disabled by default.
- The new cluster-local splitting policy is **experimental, CPU-only and opt-in**. A rotated synthetic workload benefits; real-data evidence remains mixed, so the global policy remains default.
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
