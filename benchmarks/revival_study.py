"""Paired, held-out CPU ablation: global s-means revival vs local-direction experiment.

This is an *observational* study, never a hardcoded win. Test points are
created independently from training points, or held out from load_digits.
Same train/test arrays, epochs, seed, batch law, and initializer for both arms.

Example:
  python benchmarks/revival_study.py --k 48 --epochs 2 --seeds 5 --output /tmp/study.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from svenmeans import SMeansConfig, assign_clusters, smeans


def make_data(kind: str, n: int, d: int, k: int, seed: int):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    if kind in ("digits_raw", "digits_std"):
        digits = load_digits()
        X = digits.data.astype(np.float32, copy=False)
        tr, te = train_test_split(
            X, train_size=0.75, random_state=seed, shuffle=True
        )
        if kind == "digits_std":
            scaler = StandardScaler().fit(tr)
            tr = scaler.transform(tr)
            te = scaler.transform(te)
        return torch.from_numpy(np.ascontiguousarray(tr, dtype=np.float32)), torch.from_numpy(
            np.ascontiguousarray(te, dtype=np.float32)
        )

    groups = min(12, max(4, k // 4))
    centers = torch.randn((groups, d), generator=rng) * 7
    directions = torch.randn((groups, d), generator=rng)
    directions = torch.nn.functional.normalize(directions, dim=1)
    weights = torch.ones(groups)
    if kind == "zipf":
        weights = (torch.arange(groups, dtype=torch.float32) + 1).pow(-1.3)
    weights /= weights.sum()

    def draw(size):
        labels = torch.multinomial(weights, size, generator=rng, replacement=True)
        noise = torch.randn((size, d), generator=rng) * 0.7
        if kind == "rotated":
            scalar = torch.randn((size, 1), generator=rng) * 4.0
            noise = torch.randn((size, d), generator=rng) * 0.15
            noise += scalar * directions[labels]
        sample = centers[labels] + noise
        if kind == "shifted":
            sample += 100_000.0
        return sample.contiguous()

    return draw(n), draw(max(1000, n // 3))


def run_one(train, test, n_clusters: int, nepochs: float, seed: int,
            geometry: str, center_data: bool, observe_every: int):
    config = SMeansConfig(revive_geometry=geometry, center_data=center_data,
                          local_observe_every=observe_every)
    start = time.perf_counter()
    result = smeans(train, n_clusters, nepochs=nepochs, seed=seed,
                    device="cpu", config=config, compute_labels=False,
                    return_info=True)
    elapsed = (time.perf_counter() - start) * 1000
    with torch.no_grad():
        _, sq = assign_clusters(test, result.centroids)
        mse = float(sq.double().mean().item())
    return {
        "geometry": geometry,
        "seed": seed,
        "test_mse": mse,
        "fit_ms": elapsed,
        "total_splits": result.total_splits,
        "dead_clusters": result.dead_clusters,
    }


def evaluate(kind: str, args):
    train, test = make_data(kind, args.n, args.d, args.k, args.data_seed)
    arms = ["global", "local_direction"]
    rows = []
    for seed in range(args.seeds):
        # Alternate order to reduce one-sided cache/warmup effects.
        order = arms if seed % 2 == 0 else list(reversed(arms))
        for geometry in order:
            rows.append(run_one(
                train, test, args.k, args.epochs, seed, geometry,
                center_data=(kind == "shifted"),
                observe_every=args.observe_every,
            ))
    paired = []
    for seed in range(args.seeds):
        base = next(x for x in rows if x["seed"] == seed and x["geometry"] == "global")
        new = next(x for x in rows if x["seed"] == seed and x["geometry"] == "local_direction")
        paired.append({
            "seed": seed,
            "mse_gain_pct": 100 * (1.0 - new["test_mse"] / base["test_mse"]),
            "time_ratio": new["fit_ms"] / base["fit_ms"],
            "split_difference": new["total_splits"] - base["total_splits"],
        })
    return {
        "kind": kind,
        "n_train": int(train.shape[0]),
        "n_test": int(test.shape[0]),
        "d": int(train.shape[1]),
        "k": args.k,
        "nepochs": args.epochs,
        "mean_gain_pct": statistics.mean(x["mse_gain_pct"] for x in paired),
        "median_gain_pct": statistics.median(x["mse_gain_pct"] for x in paired),
        "median_time_ratio": statistics.median(x["time_ratio"] for x in paired),
        "local_wins": sum(x["mse_gain_pct"] > 0 for x in paired),
        "local_losses": sum(x["mse_gain_pct"] < 0 for x in paired),
        "paired": paired,
        "runs": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=[
        "all", "rotated", "isotropic", "zipf", "shifted",
        "digits_raw", "digits_std"], default="all")
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--observe-every", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.n, args.d, args.k, args.seeds, args.threads, args.observe_every) < 1:
        parser.error("n,d,k,seeds,threads must be positive")
    if args.n < args.k or args.epochs <= 0:
        parser.error("n >= k and epochs > 0 required")
    torch.set_num_threads(args.threads)
    kinds = ["rotated", "isotropic", "zipf", "shifted",
             "digits_raw", "digits_std"] if args.kind == "all" else [args.kind]
    datasets = [evaluate(k, args) for k in kinds]
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "num_threads": torch.get_num_threads(),
        },
        "settings": {**vars(args), "output": str(args.output) if args.output else None},
        "datasets": datasets,
    }
    payload = json.dumps(report, indent=2, default=str)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
