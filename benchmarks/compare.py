"""Reproducible held-out test MSE / wall-clock comparison against optional s-means.

Install upstream separately for --baseline: pip install git+https://github.com/suayptalha/s-means.git
No benchmark numbers are prefilled or claimed; this script produces empirical output.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from svenmeans import SMeansConfig, assign_clusters, smeans


def make_dataset(kind: str, n: int, d: int, k: int, seed: int, device: str):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    groups = max(8, min(k, 96))
    centers = torch.randn((groups, d), generator=rng) * 5.0
    weights = torch.ones(groups)
    if kind == "zipf":
        weights = (torch.arange(groups, dtype=torch.float32) + 1).pow(-1.2)
    weights /= weights.sum()

    def sample(count):
        ids = torch.multinomial(weights, count, replacement=True, generator=rng)
        points = centers[ids] + torch.randn((count, d), generator=rng) * 0.65
        if kind == "shifted":
            points += 100_000.0
        return points.contiguous().to(device)

    return sample(n), sample(max(1024, n // 3))


def eval_mse(test, centroids):
    with torch.no_grad():
        _, distances = assign_clusters(test, centroids)
        if test.is_cuda:
            torch.cuda.synchronize(test.device)
        return float(distances.double().mean().item())


def run_fit(name, fit, train, test, seed, device):
    if train.is_cuda:
        torch.cuda.synchronize(train.device)
        torch.cuda.reset_peak_memory_stats(train.device)
        before = torch.cuda.memory_allocated(train.device)
    else:
        before = None
    t0 = time.perf_counter()
    centers = fit(train, seed)
    if train.is_cuda:
        torch.cuda.synchronize(train.device)
    elapsed = time.perf_counter() - t0
    peak = (
        max(0, int(torch.cuda.max_memory_allocated(train.device) - before))
        if train.is_cuda else None
    )
    return {
        "method": name,
        "seed": seed,
        "time_ms": elapsed * 1000.0,
        "test_mse": eval_mse(test, centers),
        "extra_gpu_alloc_bytes": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["gaussian", "zipf", "shifted"], default="gaussian")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--center-data", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.n, args.d, args.k, args.seeds) < 1 or args.k > args.n:
        parser.error("n, d, k, seeds must be positive and k <= n")
    if args.epochs <= 0:
        parser.error("epochs must be positive")

    upstream = None
    if args.baseline:
        try:
            from smeans import smeans as upstream
        except ImportError as exc:
            parser.error("Install upstream s-means separately before --baseline: " + str(exc))

    train, test = make_dataset(args.kind, args.n, args.d, args.k, args.data_seed, args.device)
    config = SMeansConfig(center_data=args.center_data)
    methods = [
        ("sven-means", lambda x, seed: smeans(
            x, args.k, nepochs=args.epochs, seed=seed, device=args.device,
            compute_labels=False, config=config
        ))
    ]
    if upstream:
        methods.append(("upstream-s-means", lambda x, seed: upstream(
            x, args.k, nepochs=args.epochs, seed=seed, device=args.device,
            compute_labels=False
        )))
    rows = []
    for seed in range(args.seeds):
        for name, fit in methods:
            rows.append(run_fit(name, fit, train, test, seed, args.device))
    summary = {}
    for name, _ in methods:
        subset = [row for row in rows if row["method"] == name]
        summary[name] = {
            "mean_test_mse": statistics.mean(row["test_mse"] for row in subset),
            "median_fit_ms": statistics.median(row["time_ms"] for row in subset),
        }
    data = {
        "system": {"torch": torch.__version__, "device": args.device,
                   "cuda_device": torch.cuda.get_device_name() if train.is_cuda else None},
        "experiment": vars(args) | {"output": str(args.output) if args.output else None},
        "summary": summary,
        "runs": rows,
    }
    payload = json.dumps(data, indent=2, default=str)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
