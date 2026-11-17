"""Cache-tiled PyTorch CPU implementation of s-means clustering and assignment."""

import math
import time
from typing import Any, Dict, Tuple

import torch

from ._directional import DirectionalRevival
from ._finalization import invalidate_split_statistics
from ._rare_guard import distance_aware_removable
from ._net_guard import (
    reserve_validation,
    exact_top_three,
    net_split_gain_samples,
    lower_confidence_gate,
)
from ._recipe import (
    SMeansConfig,
    compute_epoch_rows,
    affine_epoch_indices,
    affine_epoch_key,
    feistel_epoch_indices,
    feistel_epoch_key,
    make_generator,
    make_recipe,
    subsampled_d15_init,
)


def tiled_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_norm: torch.Tensor | None = None,
    c_norm: torch.Tensor | None = None,
    c_t: torch.Tensor | None = None,
    tile_rows: int = 512,
    tile_cols: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute nearest centroid labels and squared Euclidean distances via L2/L3 cache tiles."""
    n = int(x.shape[0])
    k = int(centroids.shape[0])
    device = x.device
    if x_norm is None:
        x_norm = (x * x).sum(dim=1)
    if c_norm is None:
        c_norm = (centroids * centroids).sum(dim=1)
    if c_t is None:
        c_t = centroids.t().contiguous()

    labels = torch.empty((n,), dtype=torch.int64, device=device)
    sq_dists = torch.empty((n,), dtype=torch.float32, device=device)

    eff_r = max(1, min(int(tile_rows), n))
    eff_c = max(1, min(int(tile_cols), k))

    for i0 in range(0, n, eff_r):
        i1 = min(i0 + eff_r, n)
        xb = x[i0:i1]
        xbn = x_norm[i0:i1]
        best_val = None
        best_idx = None
        for j0 in range(0, k, eff_c):
            j1 = min(j0 + eff_c, k)
            tile = torch.addmm(c_norm[j0:j1], xb, c_t[:, j0:j1], alpha=-2.0, beta=1.0)
            vals, locs = tile.min(dim=1)
            if best_val is None:
                best_val = vals
                best_idx = locs + j0
            else:
                mask = vals < best_val
                best_val = torch.where(mask, vals, best_val)
                best_idx = torch.where(mask, locs + j0, best_idx)
        labels[i0:i1] = best_idx
        sq_dists[i0:i1] = (xbn + best_val).clamp_min_(0.0)

    return labels, sq_dists


def fit_cpu(
    x: torch.Tensor,
    k: int,
    nepochs: float = 3.0,
    seed: int = 0,
    cfg: SMeansConfig = SMeansConfig(),
    compute_labels: bool = True,
) -> Dict[str, Any]:
    """Fit s-means on CPU tensor `x` (shape `[n, d]`, float32)."""
    x_f = x.detach().to(dtype=torch.float32).contiguous()
    original_x = x_f
    n, d = int(x_f.shape[0]), int(x_f.shape[1])
    device = x_f.device
    net_enabled = cfg.revive_acceptance == "net_holdout"
    if cfg.revive_acceptance not in ("none", "net_holdout"):
        raise ValueError("revive_acceptance must be 'none' or 'net_holdout'")
    if (not math.isfinite(cfg.net_validation_fraction)
            or not 0.0 < cfg.net_validation_fraction <= 0.4):
        raise ValueError("net_validation_fraction must be in (0, 0.4]")
    if (not math.isfinite(cfg.net_lcb_z) or cfg.net_lcb_z < 0
            or not math.isfinite(cfg.net_min_gain) or cfg.net_min_gain < 0
            or cfg.net_max_proposals < 1):
        raise ValueError("net LCB z/gain must be finite nonnegative and max proposals >= 1")
    # Crucially, training never sees the reserved rows: not during seeding,
    # EMA updates, candidate selection, or final-pass centroid fitting.
    validation = None
    if net_enabled:
        x_f, validation = reserve_validation(
            x_f, seed, cfg.net_validation_fraction
        )
        if len(x_f) < k:
            raise ValueError("net validation leaves fewer than k fit points")
    n = len(x_f)
    x_norm = (x_f * x_f).sum(dim=1).contiguous()

    recipe = make_recipe(n, k, d, nepochs, revive_thresh=cfg.revive_thresh)
    batch_size = max(1, min(n, recipe.batch_size))
    alpha0 = recipe.alpha0
    check_every = recipe.check_every
    unused_thresh = recipe.unused_thresh
    if cfg.shuffle_mode not in ("randperm", "feistel", "affine"):
        raise ValueError("shuffle_mode must be 'randperm', 'feistel' or 'affine'")
    if cfg.revive_geometry not in ("global", "local_direction"):
        raise ValueError("revive_geometry must be 'global' or 'local_direction'")
    if cfg.revive_protection not in ("none", "distance"):
        raise ValueError("revive_protection must be 'none' or 'distance'")
    if cfg.revive_near_factor <= 0 or not math.isfinite(cfg.revive_near_factor):
        raise ValueError("revive_near_factor must be finite and > 0")
    if cfg.revive_abandon_epochs <= 0 or not math.isfinite(cfg.revive_abandon_epochs):
        raise ValueError("revive_abandon_epochs must be finite and > 0")
    if cfg.local_observe_every < 1:
        raise ValueError("local_observe_every must be >= 1")
    if cfg.local_direction_objective not in ("covariance", "split_gain"):
        raise ValueError("local_direction_objective must be 'covariance' or 'split_gain'")
    if cfg.local_radius_policy not in ("rms_scaled", "mean_abs"):
        raise ValueError("local_radius_policy must be 'rms_scaled' or 'mean_abs'")
    epoch_rows = compute_epoch_rows(n, nepochs)
    total_steps = sum((rows + batch_size - 1) // batch_size for rows in epoch_rows)

    t0 = time.perf_counter()

    centroids = subsampled_d15_init(
        x_f, k, seed=seed, rounds=cfg.init_rounds, power=cfg.init_power, max_samples=cfg.init_max_samples, tile_cols=cfg.init_tile_cols
    ).to(dtype=torch.float32, device=device).contiguous()
    usage = torch.full((k,), recipe.init_usage, dtype=torch.float32, device=device)
    sse_ema = torch.zeros((k,), dtype=torch.float32, device=device)
    feat_scale = torch.ones((d,), dtype=torch.float32, device=device)
    # Guard distortion is an alpha-synchronized EMA, comparable with usage.
    guarded = cfg.revive_protection == "distance"
    guard_distortion = (
        torch.zeros((k,), dtype=torch.float32, device=device) if guarded else None
    )
    guard_last_seen = (
        torch.zeros((k,), dtype=torch.int64, device=device) if guarded else None
    )
    guard_wait_steps = max(
        1, int(math.ceil(cfg.revive_abandon_epochs * math.ceil(n / batch_size)))
    )
    guarded_candidate_checks = 0
    net_proposed = 0
    net_accepted = 0
    net_rejected = 0
    net_gain_sum = 0.0
    net_lcb_sum = 0.0

    pass_sums = torch.zeros((k, d), dtype=torch.float64, device=device)
    pass_counts = torch.zeros((k,), dtype=torch.float64, device=device)
    batch_sums = torch.empty((k, d), dtype=torch.float32, device=device)
    batch_counts = torch.empty((k,), dtype=torch.float32, device=device)
    batch_sse = torch.empty((k,), dtype=torch.float32, device=device)

    directional = (
        DirectionalRevival(
            k, d, seed=seed, device=device, beta=cfg.local_beta,
            min_observations=cfg.local_min_observations,
            direction_objective=cfg.local_direction_objective,
        )
        if cfg.revive_geometry == "local_direction" else None
    )

    rev_gen = make_generator(device, seed)
    c_norm = (centroids * centroids).sum(dim=1)
    c_t = centroids.t().contiguous()
    dirty_c = False

    step = 0
    total_splits = 0
    revive_stop_step = int(cfg.revive_stop_frac * total_steps)
    n_ep = len(epoch_rows)

    for ep_idx, rows in enumerate(epoch_rows):
        if cfg.shuffle_mode == "randperm":
            perm = torch.randperm(
                n,
                generator=make_generator(device, (seed * 1000003 + ep_idx) % (2**32)),
                device=device,
            )
        else:
            perm = None
            epoch_key = (feistel_epoch_key(n, seed, ep_idx)
                         if cfg.shuffle_mode == "feistel" else
                         affine_epoch_key(n, seed, ep_idx))
        is_last_pass = ep_idx == n_ep - 1
        steps_in_ep = (rows + batch_size - 1) // batch_size

        for s_ep, begin in enumerate(range(0, rows, batch_size)):
            end = min(begin + batch_size, rows)
            if perm is not None:
                idx = perm[begin:end]
            elif cfg.shuffle_mode == "feistel":
                idx = feistel_epoch_indices(n, begin, end, epoch_key, device)
            else:
                idx = affine_epoch_indices(n, begin, end, epoch_key, device)
            xb = x_f[idx]
            xbn = x_norm[idx]
            b = int(xb.shape[0])

            if dirty_c:
                c_norm = (centroids * centroids).sum(dim=1)
                c_t = centroids.t().contiguous()
                dirty_c = False

            labels, sq_dists = tiled_assign(
                xb,
                centroids,
                x_norm=xbn,
                c_norm=c_norm,
                c_t=c_t,
                tile_rows=cfg.assign_tile_rows,
                tile_cols=cfg.assign_tile_cols,
            )

            batch_sums.zero_()
            batch_counts.zero_()
            batch_sse.zero_()
            batch_sums.index_add_(0, labels, xb)
            batch_counts.scatter_add_(0, labels, torch.ones((b,), dtype=torch.float32, device=device))
            batch_sse.scatter_add_(0, labels, sq_dists)

            if directional is not None and step % cfg.local_observe_every == 0:
                directional.observe(xb, labels, centroids, batch_counts)

            frac = min(step, total_steps) / max(1, total_steps)
            alpha = alpha0 * (0.06 + 0.94 * (0.5 * (1.0 + math.cos(math.pi * frac))) ** 0.92)
            pfrac = (s_ep + 0.5) / max(1, steps_in_ep)
            w_pass = (1.0 - cfg.pass_w_span) + 2.0 * cfg.pass_w_span * pfrac

            if is_last_pass:
                if w_pass == 1.0:
                    pass_sums.add_(batch_sums)
                    pass_counts.add_(batch_counts)
                else:
                    pass_sums.add_(batch_sums, alpha=w_pass)
                    pass_counts.add_(batch_counts, alpha=w_pass)

            # Online inter-cluster feature variance for anisotropic revival
            tot_c = float(batch_counts.sum().item())
            if tot_c > 0.0:
                batch_mu = batch_sums.sum(dim=0) / max(1.0, tot_c)
                cur_var = ((batch_sums - batch_counts[:, None] * batch_mu[None, :]).pow(2).sum(dim=0) / max(1.0, tot_c)).clamp_min(1e-12)
                feat_scale.mul_(cfg.var_beta).add_(cur_var, alpha=1.0 - cfg.var_beta)

            # EMA centroid and usage update
            touched = batch_counts > 0
            safe_c = torch.where(touched, batch_counts, 1.0)
            mean = batch_sums / safe_c[:, None]
            is_rep = alpha >= 1.0
            retained = torch.zeros_like(usage) if is_rep else usage * (1.0 - alpha)
            mass = retained + alpha * batch_counts
            w = alpha * batch_counts / torch.where(touched, mass, 1.0)
            upd = torch.where((w == 1.0)[:, None], mean, centroids + w[:, None] * (mean - centroids))
            centroids = torch.where(touched[:, None], upd, centroids)
            usage = torch.where(touched, mass, usage if is_rep else retained)
            sse_ema.mul_(cfg.sse_decay).add_(batch_sse, alpha=1.0 - cfg.sse_decay)
            if guarded:
                assert guard_distortion is not None and guard_last_seen is not None
                guard_distortion.mul_(1.0 - alpha).add_(batch_sse, alpha=alpha)
                guard_last_seen[touched] = step
            dirty_c = True

            # Revival has two controlled options:
            #   global: exact legacy distortion- and global-variance policy.
            #   local_direction: rank-one streaming residual geometry, scored
            #       by estimated per-donor gain from a symmetric pair split.
            if ((step + 1) % check_every == 0) and (step <= revive_stop_step):
                thresh = float(usage.sum().item()) * unused_thresh / max(1, k)
                unused = usage < thresh
                if guarded and bool(unused.any().item()):
                    assert guard_distortion is not None and guard_last_seen is not None
                    # Distance-aware candidate filtering adapts Weiss (2026),
                    # arXiv:2611.18238; it is not a sven-means invention.
                    mse_ema = guard_distortion.sum() / usage.sum().clamp_min(1e-12)
                    eligible = distance_aware_removable(
                        centroids, unused,
                        mse=float(mse_ema.item()),
                        near_factor=cfg.revive_near_factor,
                        last_seen=guard_last_seen,
                        step=step,
                        abandon_steps=guard_wait_steps,
                    )
                    guarded_candidate_checks += int((unused & ~eligible).sum().item())
                    targets_available = eligible
                else:
                    targets_available = unused
                # Protected low-use centers are neither sources nor targets.
                donors = (~unused) & (usage > 0.0)
                max_splits = min(int(targets_available.sum().item()), int(donors.sum().item()))
                if max_splits > 0:
                    if directional is not None:
                        src, delta = directional.propose(
                            donors, max_splits, usage, cfg.split_mult,
                            cfg.local_gain_floor,
                            cfg.local_radius_policy,
                        )
                        n_sp = int(src.numel())
                    else:
                        n_sp = max_splits
                        has_sse = float(sse_ema.max().item()) > 0.0
                        score_vec = (
                            sse_ema.clamp_min(1e-12) * (usage.clamp_min(1e-12) ** cfg.donor_usage_w)
                            if has_sse
                            else usage
                        )
                        u_rand = torch.rand((k,), dtype=usage.dtype, device=device, generator=rev_gen).clamp_min(
                            torch.finfo(usage.dtype).tiny
                        )
                        keys = torch.where(
                            donors, score_vec.clamp_min(1e-30).log() - (-u_rand.log()).log(), float("-inf")
                        )
                        src = torch.topk(keys, n_sp).indices
                        fvec = feat_scale.clamp_min(1e-10).sqrt()
                        fvec = fvec / fvec.norm().clamp_min(1e-8) * math.sqrt(float(d))
                        rad = cfg.split_mult * (
                            sse_ema[src] / usage[src].clamp_min(1e-6) / max(1, d)
                        ).clamp_min(1e-10).sqrt()[:, None]
                        radem = torch.where(
                            torch.rand((n_sp, d), device=device, generator=rev_gen) < 0.5,
                            -1.0,
                            1.0,
                        )
                        delta = rad * radem * fvec[None, :]

                    if n_sp > 0:
                        tgt = torch.nonzero(targets_available, as_tuple=True)[0][:n_sp]
                        if net_enabled:
                            assert validation is not None
                            # Proposals come from fit-only data. The *full
                            # nearest-centroid loss*, including the cost of
                            # deleting the target, is measured on reserved rows.
                            # Recompute top-3 after each accepted mutation.
                            accepted_src = []
                            accepted_tgt = []
                            accepted_delta = []
                            trials = min(n_sp, cfg.net_max_proposals)
                            top_dist, top_ids = exact_top_three(validation, centroids)
                            for i in range(trials):
                                ds = int(src[i].item())
                                dt = int(tgt[i].item())
                                per_row = net_split_gain_samples(
                                    validation, top_dist, top_ids, ds, dt,
                                    centroids[ds], delta[i],
                                )
                                ok, estimate, lcb = lower_confidence_gate(
                                    per_row, z=cfg.net_lcb_z,
                                    gain_floor=cfg.net_min_gain,
                                )
                                net_proposed += 1
                                if ok:
                                    accepted_src.append(ds)
                                    accepted_tgt.append(dt)
                                    accepted_delta.append(delta[i].clone())
                                    net_accepted += 1
                                    net_gain_sum += estimate
                                    net_lcb_sum += lcb
                                    # Temporary local copy updates ensure next
                                    # proposal is tested against prior accepts.
                                    original = centroids[ds].clone()
                                    centroids[ds] = original - delta[i]
                                    centroids[dt] = original + delta[i]
                                    top_dist, top_ids = exact_top_three(
                                        validation, centroids
                                    )
                                else:
                                    net_rejected += 1
                            if accepted_src:
                                src = torch.tensor(accepted_src, dtype=torch.int64, device=device)
                                tgt = torch.tensor(accepted_tgt, dtype=torch.int64, device=device)
                                delta = torch.stack(accepted_delta)
                                # Restore old positions so the existing split
                                # state/usage bookkeeping below runs unchanged.
                                # Positions for accepted sources have already
                                # been changed above; reconstruct old center:
                                # c_old = c_new_source + delta.
                                centroids[src] += delta
                                n_sp = len(accepted_src)
                            else:
                                n_sp = 0

                    if n_sp > 0:
                        dc = centroids[src].clone()
                        half = 0.5 * usage[src]
                        centroids[tgt] = dc + delta
                        centroids[src] = dc - delta
                        usage[tgt] = half
                        usage[src] = half
                        hsse = 0.5 * sse_ema[src]
                        sse_ema[tgt] = hsse
                        sse_ema[src] = hsse
                        if guarded:
                            assert guard_distortion is not None and guard_last_seen is not None
                            half_distortion = 0.5 * guard_distortion[src]
                            guard_distortion[tgt] = half_distortion
                            guard_distortion[src] = half_distortion
                            guard_last_seen[tgt] = step
                            guard_last_seen[src] = step
                        if directional is not None:
                            directional.on_split(src, tgt)
                        if is_last_pass and cfg.reset_pass_on_split:
                            # A post-split center must not inherit the donor's
                            # old single-center assignment mean.
                            invalidate_split_statistics(
                                pass_sums, pass_counts, src, tgt, cfg.donor_pass_retention
                            )
                        dirty_c = True
                        total_splits += n_sp

            step += 1

    # Empirical-Bayes finalization shrinking low-count pass means toward tracked EMA centroids
    tau = float(cfg.eb_tau)
    w_fin = (pass_counts / (pass_counts + tau)).float()
    pmean = (pass_sums / pass_counts.clamp_min(1.0)[:, None]).float()
    centroids = torch.where((pass_counts > 0)[:, None], w_fin[:, None] * pmean + (1.0 - w_fin[:, None]) * centroids, centroids)

    # Skip the full O(n*k*d) evaluation pass when only centroids are requested.
    # The original CPU implementation executed this pass even if labels were disabled.
    labels, inertia = None, None
    if compute_labels:
        all_norm = (original_x * original_x).sum(dim=1).contiguous() if net_enabled else x_norm
        labels, sq_dists = tiled_assign(
            original_x, centroids, x_norm=all_norm,
            tile_rows=cfg.assign_tile_rows, tile_cols=cfg.assign_tile_cols,
        )
        inertia = float(sq_dists.sum().item())
    t1 = time.perf_counter()

    return {
        "centroids": centroids,
        "labels": labels,
        "inertia": inertia,
        "time_s": t1 - t0,
        "n_iter": int(step),
        "dead_clusters": int((pass_counts <= 0.5).sum().item()),
        "total_splits": int(total_splits),
        "guarded_candidate_checks": int(guarded_candidate_checks),
        "net_proposed": int(net_proposed),
        "net_accepted": int(net_accepted),
        "net_rejected": int(net_rejected),
        "net_validation_rows": int(len(validation)) if validation is not None else 0,
        "net_mean_accepted_gain": net_gain_sum / max(1, net_accepted),
        "net_mean_accepted_lcb": net_lcb_sum / max(1, net_accepted),
        "backend": "cpu_tiled",
        "dtype": "float32",
    }
