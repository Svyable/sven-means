"""Hyperparameter scaling rules, epoch budgeting, and subsampled D^1.5 seeding."""

import math
from dataclasses import dataclass
from typing import List

import torch


@dataclass(frozen=True)
class SMeansConfig:
    """Algorithmic hyperparameters for s-means clustering."""

    revive_thresh: float = 0.1
    revive_stop_frac: float = 0.85
    split_mult: float = 0.55
    donor_usage_w: float = 0.35
    eb_tau: float = 0.20
    pass_w_span: float = 0.35
    sse_decay: float = 0.85
    var_beta: float = 0.85
    init_rounds: int = 4
    init_power: float = 1.5
    init_max_samples: int = 16384
    init_tile_cols: int = 128
    center_data: bool = False
    assign_tile_rows: int = 512
    assign_tile_cols: int = 256


@dataclass(frozen=True)
class SMeansRecipe:
    """Dataset-dependent batch size, step-size, and dead-cluster schedule."""

    batch_size: int
    alpha0: float
    check_every: int
    unused_thresh: float
    revive_min_d: int
    init_usage: float


def round_pow2(val: float) -> int:
    val = max(1.0, float(val))
    return max(1, int(2 ** round(math.log2(val))))


def make_recipe(
    n: int,
    k: int,
    d: int,
    nepochs: float,
    revive_thresh: float = 0.1,
) -> SMeansRecipe:
    """Compute dimension-aware batch size and Robbins-Monro initial step rate."""
    law = "quality" if d <= 16 else "throughput"
    if law == "quality":
        c, bk, bn, bt = 0.00439175, 0.134749, 0.876473, 0.813344
        cap = max(1, n // 2)
    else:
        c, bk, bn, bt = 0.035134, 0.134749, 0.876473, 0.813344
        cap = max(1, round_pow2(n / 2.0))
    b = max(1, min(n, round_pow2(c * (k**bk) * (n**bn) * (nepochs**bt)), cap))
    alpha0 = max(
        1e-4,
        min(
            1.0,
            0.8244 * (k**-0.4617) * (b**0.9003) * (n**-0.4498) * (nepochs**-0.1057),
        ),
    )
    return SMeansRecipe(
        batch_size=int(b),
        alpha0=float(alpha0),
        check_every=5,
        unused_thresh=float(revive_thresh),
        revive_min_d=1,
        init_usage=2.0**-16,
    )


def compute_epoch_rows(n: int, nepochs: float) -> List[int]:
    """Split arbitrary fractional epoch budgets into whole and tail passes.

    For fractional budgets like T = 1.5, the partial pass runs first and a full
    epoch finishes at the end so final-pass averaging covers the whole dataset.
    """
    if nepochs <= 0:
        raise ValueError(f"nepochs must be positive, got {nepochs}")
    whole = math.floor(nepochs)
    rem = int(round((nepochs - whole) * n))
    if whole == 0:
        return [max(1, rem)]
    return [n] * (whole - 1) + ([rem] if rem > 0 else []) + [n]


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) % (2**31 - 1))
    return gen


def _sample_coreset_indices(n: int, m: int, rng: torch.Generator, device: torch.device) -> torch.Tensor:
    """Uniformly sample m unique row indices without an O(n) permutation when m << n.

    For small fractions, oversampled draws are deduplicated and a random m-subset
    is selected. The unique-pool permutation avoids bias from sorted unique IDs.
    """
    if m >= n // 2 or n <= 4096:
        return torch.randperm(n, generator=rng, device=device)[:m]
    pool = torch.unique(torch.randint(n, (2 * m,), generator=rng, device=device))
    while int(pool.numel()) < m:
        extra = torch.randint(n, (2 * (m - int(pool.numel())),), generator=rng, device=device)
        pool = torch.unique(torch.cat((pool, extra)))
    return pool[torch.randperm(int(pool.numel()), generator=rng, device=device)[:m]]


def _update_min_sqdist(
    x: torch.Tensor,
    x_norm: torch.Tensor,
    candidates: torch.Tensor,
    current: torch.Tensor,
    tile_cols: int,
) -> torch.Tensor:
    """Update point-to-nearest-center squared distances in bounded column tiles.

    Peak distance workspace is O(m * tile_cols), not O(m * k).
    Coordinates are already shifted to a common origin by the caller.
    """
    if tile_cols < 1:
        raise ValueError("init_tile_cols must be >= 1")
    for j in range(0, int(candidates.shape[0]), tile_cols):
        c = candidates[j : j + tile_cols]
        c_norm = c.square().sum(dim=1)
        # addmm avoids constructing a three-dimensional distance tensor.
        sq = torch.addmm(c_norm[None, :], x, c.t(), beta=1.0, alpha=-2.0)
        sq.add_(x_norm[:, None]).clamp_min_(0.0)
        current = torch.minimum(current, sq.amin(dim=1))
    return current


def subsampled_d15_init(
    x: torch.Tensor,
    k: int,
    seed: int,
    rounds: int = 4,
    power: float = 1.5,
    max_samples: int = 16384,
    tile_cols: int = 128,
) -> torch.Tensor:
    """Distance-weighted, unique seeding with bounded working memory.

    Uses a coreset of min(n, max(4*k, 512), max(max_samples, k)) points.
    Squared Euclidean distances are raised to `power`, preserving upstream's
    weighting semantics. Distance columns are tiled; no O(k**2) distance
    matrix is materialized. Work remains O(m*k*d), so this is not a
    constant-time initializer for enormous k.

    Numeric distances are computed after subtracting one common anchor,
    which avoids cancellation for datasets with large global offsets.
    """
    n = int(x.shape[0])
    if k < 1 or n < k:
        raise ValueError(f"Expected 1 <= k <= n, got k={k}, n={n}")
    if rounds < 2 or max_samples < 1 or tile_cols < 1 or power < 0:
        raise ValueError("rounds >= 2, max_samples >= 1, tile_cols >= 1, power >= 0 required")
    if n == k:
        return x.float().clone()

    rng = make_generator(x.device, seed)
    m = min(n, max(4 * k, 512), max(k, int(max_samples)))
    idx = _sample_coreset_indices(n, m, rng, x.device)
    xs = x[idx].float().contiguous()

    # Translation does not change distances; computing in centered coordinates
    # reduces float32 cancellation when the original data have a large offset.
    centered = (xs - xs[:1]).contiguous()
    x_norm = centered.square().sum(dim=1)

    nr = int(rounds)
    k0 = min(k, max(8, k // (nr * 2)))
    chosen = torch.arange(k0, device=x.device)
    taken = torch.zeros(m, dtype=torch.bool, device=x.device)
    taken[:k0] = True
    dmin = _update_min_sqdist(
        centered, x_norm, centered[:k0], torch.full((m,), float("inf"), device=x.device),
        int(tile_cols),
    )
    step_k = max(1, (k - k0 + nr - 2) // max(1, nr - 1))

    for _ in range(nr - 1):
        remaining = k - int(chosen.numel())
        if remaining <= 0:
            break
        cand_n = min(step_k, remaining)
        # Normalize before exponentiation, so very large distances cannot
        # overflow. Give zero-distance *unselected* points a tiny weight.
        scale = dmin.max().clamp_min(1e-30)
        weights = (dmin / scale).clamp_min_(0.0).pow_(power)
        weights = torch.where(taken, 0.0, weights + 1e-10)
        picked = torch.multinomial(weights, cand_n, replacement=False, generator=rng)
        taken[picked] = True
        chosen = torch.cat((chosen, picked))
        dmin = _update_min_sqdist(
            centered, x_norm, centered[picked], dmin, int(tile_cols)
        )

    if int(chosen.numel()) != k:
        raise RuntimeError("initializer did not generate k unique coreset indices")
    return xs[chosen].contiguous()
