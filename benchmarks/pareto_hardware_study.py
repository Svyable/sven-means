"""Exp 012: expensive-evaluation selection on real NATS-Bench-MCU v2 hardware.

Unlike Exstill's proprietary 5nm chip search, this is a measured Nordic
nRF5340 microcontroller / neural-architecture design table. Results DO
NOT establish performance on Exstill or physical IC floorplanning.

Only cell topology and pre-measurement INT8 model size are given to
policies. The 64 calibration evaluations are charged to EVERY budget.
All other accuracy, device energy and latency are hidden until the
policy has irrevocably selected architecture IDs. Offline oracle lookup
is for *evaluation* only: no new hardware is actually built or tested.

The v2 export has CC BY 4.0 provenance:
https://doi.org/10.5281/zenodo.21644470
https://github.com/ankilab/NATS-Bench-MCU
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from benchmarks.pareto_archive import (
    OPS, topology_features, fit_linear_proxy, choose_candidates,
)

METHODS = (
    "random", "stratified", "farthest", "kmeans++",
    "proxy_rank", "proxy_grid"
)


def load_nats(hardware_csv, accuracy_csv):
    """Read v2 successful deployments and align two CSVs on arch_idx."""
    hardware_csv, accuracy_csv = Path(hardware_csv), Path(accuracy_csv)
    actual = {}
    with accuracy_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = int(row["arch_idx"])
            if key in actual:
                raise ValueError("duplicate accuracy architecture index")
            actual[key] = float(row["cifar10"])
    ids, topologies, sizes, rows = [], [], [], []
    with hardware_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = int(row["arch_idx"])
            if key not in actual:
                raise ValueError(f"missing accuracy for arch {key}")
            ids.append(key)
            topologies.append(row["arch_str"])
            sizes.append(float(row["int8_kb"]))
            rows.append((
                float(row["latency_s"]),
                float(row["energy_mJ"]),
                float(row["rom_kb"]),
                actual[key],
            ))
    if len(ids) != len(set(ids)) or len(ids) != len(actual):
        raise ValueError("different numbers of measured architectures and accuracies")
    if len(ids) != 14281:
        raise ValueError(f"expected NATS-Bench-MCU v2's 14,281 deployed rows, got {len(ids)}")
    truth = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(truth).all() or np.any(truth[:, :3] <= 0):
        raise ValueError("hardware measurements must be positive finite")
    return {
        "ids": np.array(ids, dtype=np.int64),
        "topologies": topologies,
        "sizes": np.array(sizes, dtype=np.float64),
        # TRUE objectives may only be accessed in label-calibration and
        # post-selection evaluation; policy functions never receive them.
        "truth": truth,
        "dataset_provenance": {
            "zenodo_doi": "10.5281/zenodo.21644470",
            "version": "2.0",
            "license": "CC BY 4.0 (Zimmermann, Groh, Kist, 2026)",
            "n_deployable": len(ids),
            "hardware_sha256": hashlib.sha256(hardware_csv.read_bytes()).hexdigest(),
            "accuracies_sha256": hashlib.sha256(accuracy_csv.read_bytes()).hexdigest(),
        }
    }


def _pareto_mask3(points):
    """O(n log n) exact minimization Pareto front for three objectives.

    Sweep first objective, use Fenwick prefix minima for the third
    objective indexed by discretized second. Equal objective triples
    remain all nondominated. Distinct triples use strict Pareto rules.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("expected [n,3] costs")
    n = len(p)
    if not n:
        return np.zeros(0, dtype=bool)
    if not np.isfinite(p).all():
        raise ValueError("finite costs required")
    ranks = np.searchsorted(np.unique(p[:, 1]), p[:, 1]) + 1
    tree = np.full(int(ranks.max()) + 1, float("inf"))
    mask = np.zeros(n, dtype=bool)
    order = np.lexsort((p[:, 2], p[:, 1], p[:, 0]))
    i = 0
    while i < n:
        first = order[i]
        j = i+1
        while j < n and np.array_equal(p[order[j]], p[first]):
            j += 1
        r = int(ranks[first])
        prefix = float("inf")
        a = r
        while a > 0:
            prefix = min(prefix, tree[a])
            a -= a & -a
        on_front = not (prefix <= p[first, 2])
        if on_front:
            mask[order[i:j]] = True
        a = r
        while a < len(tree):
            tree[a] = min(tree[a], p[first, 2])
            a += a & -a
        i = j
    return mask


def _hv2d(points, reference):
    """Union of rectangles [y,ref_y] x [z,ref_z] for minimization."""
    a = np.asarray(points, dtype=np.float64)
    if len(a) == 0:
        return 0.
    a = a[(a[:, 0] < reference[0]) & (a[:, 1] < reference[1])]
    if not len(a):
        return 0.
    order = np.argsort(a[:, 0], kind="stable")
    a = a[order]
    best_z = float("inf")
    volume = 0.
    for i in range(len(a)):
        best_z = min(best_z, float(a[i, 1]))
        next_y = float(a[i+1, 0]) if i+1 < len(a) else reference[0]
        volume += max(0., next_y - float(a[i, 0])) * max(0., reference[1]-best_z)
    return volume


def hv3(points, reference=(1.05, 1.05, 1.05)):
    """Exact dominated 3D hypervolume for minimization; no libraries."""
    a = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(reference, dtype=np.float64)
    if not len(a):
        return 0.0
    a = a[np.all(a < ref, axis=1)]
    if not len(a):
        return 0.
    a = a[_pareto_mask3(a)]
    order = np.argsort(a[:, 0], kind="stable")
    a = a[order]
    volume = 0.
    previous = 0
    while previous < len(a):
        last = previous + 1
        while last < len(a) and a[last, 0] == a[previous, 0]:
            last += 1
        next_x = float(a[last, 0]) if last < len(a) else ref[0]
        volume += max(0., next_x-a[previous, 0]) * _hv2d(a[:last, 1:], ref[1:])
        previous = last
    return float(volume)


def truth_metrics(all_truth, target_pool, selected, *,
                  accuracy_floor, flash_cap, objective_ranges=None):
    """Post-selection evaluation, never invoked by candidate selectors."""
    truth = np.asarray(all_truth, dtype=np.float64)
    pool = np.asarray(target_pool, dtype=np.int64)
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) != len(set(selected.tolist())):
        raise ValueError("duplicate oracle evaluation indices")
    if not set(selected.tolist()).issubset(set(pool.tolist())):
        raise ValueError("selected indices outside measured design universe")
    objectives = truth[pool, :3]
    feasible = (
        (truth[pool, 3] >= accuracy_floor) &
        (truth[pool, 2] <= flash_cap)
    )
    fpool = pool[feasible]
    if not len(fpool):
        raise ValueError("no feasible designs under registered constraints")
    # Evaluator-only normalization using true full-pool bounds, to make
    # dominated hypervolumes unitless. Neither policy nor proxy sees them.
    lower = truth[fpool, :3].min(0)
    upper = truth[fpool, :3].max(0)
    span = np.maximum(upper - lower, 1e-9)
    true_cost = (truth[:, :3] - lower) / span
    fcost = true_cost[fpool]
    front_ids = fpool[_pareto_mask3(fcost)]
    full_hv = hv3(true_cost[fpool])
    if full_hv <= 0:
        raise AssertionError("empty feasible true hypervolume")
    selected_feasible = selected[
        (truth[selected, 3] >= accuracy_floor) &
        (truth[selected, 2] <= flash_cap)
    ]
    selected_hv = hv3(true_cost[selected_feasible])
    front_costs = true_cost[front_ids]
    if len(selected_feasible):
        chosen_costs = true_cost[selected_feasible]
        approx_coverage = float(np.mean([
            np.any(np.all(chosen_costs <= point + 0.05, axis=1))
            for point in front_costs
        ]))
        best_edp = float(np.min(
            truth[selected_feasible, 0] * truth[selected_feasible, 1]
        ))
    else:
        approx_coverage = 0.
        best_edp = None
    oracle_best_edp = float(np.min(
        truth[fpool, 0] * truth[fpool, 1]
    ))
    return {
        "feasible_pareto_points": len(front_ids),
        "true_feasible_designs": len(fpool),
        "discovered_feasible": len(selected_feasible),
        "front_exact_recall": float(len(set(front_ids.tolist()) &
                                           set(selected_feasible.tolist())) / len(front_ids)),
        "front_epsilon_coverage": approx_coverage,
        "hypervolume_ratio": float(min(1.0, selected_hv/full_hv)),
        "hypervolume_regret": float(max(0.0, 1.0-selected_hv/full_hv)),
        "best_feasible_edp_ratio": (float(best_edp/oracle_best_edp)
                                     if best_edp is not None else None),
    }


def run_replay(data, *, family_to_holdout, seeds, fractions,
               calibration_count, accuracy_floor, flash_cap,
               policies=METHODS):
    features, family = topology_features(data["topologies"], data["sizes"])
    n = len(features)
    true = data["truth"]
    labels = np.column_stack((np.log(true[:, 0]), np.log(true[:, 1]),
                              true[:, 3]))
    if family_to_holdout is None:
        target = np.arange(n, dtype=np.int64)
        calibrate_from = target
        split = "all-architectures (development)"
    else:
        if family_to_holdout not in OPS:
            raise ValueError("unknown holdout first-edge operator")
        family_id = OPS.index(family_to_holdout)
        target = np.flatnonzero(family == family_id)
        calibrate_from = np.flatnonzero(family != family_id)
        split = f"first-edge family {family_to_holdout} held out of calibration"
    if len(calibrate_from) < calibration_count or len(target) < 2*calibration_count:
        raise ValueError("not enough samples for requested calibration")
    budgets = [math.ceil(frac * len(target)) for frac in fractions]
    if min(budgets) <= calibration_count:
        raise ValueError("smallest budget must exceed number of paid calibration labels")

    report = {
        "source": data["dataset_provenance"], "split": split,
        "available_cheap_features": "6 categorical TSS edge operations plus log1p(INT8 converted model KB)",
        "strict_oracle_isolation": "only calibration indices passed to surrogate, all other hardware/accuracy values used after selection",
        "oracle_labels": "CIFAR-10 test accuracy, actual measured on-device latency and energy, ROM from measured deployment",
        "constraints": {"cifar10_min": accuracy_floor, "rom_kb_max": flash_cap},
        "n_target": len(target), "n_calibration_pool": len(calibrate_from),
        "total_evaluation_budgets": budgets,
        "fractions_of_target": list(fractions),
        "calibration_count_charged_to_every_budget": calibration_count,
        "results": [],
    }
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        calibration = rng.choice(calibrate_from, size=calibration_count,
                                 replace=False).astype(np.int64)
        candidates = np.setdiff1d(target, calibration, assume_unique=True)
        start = time.perf_counter()
        pred, proxy_meta = fit_linear_proxy(
            features, labels, calibration
        )
        proxy_ms = 1000*(time.perf_counter()-start)
        trial = {
            "seed": seed, "calibration_arch_ids": data["ids"][calibration].tolist(),
            "calibration_proxy": proxy_meta, "proxy_training_ms": proxy_ms,
            "budget_results": [],
        }
        for budget in budgets:
            evaluation = int(budget - calibration_count)
            if evaluation > len(candidates):
                raise AssertionError("cannot spend more than available candidates")
            experiment = {"oracle_budget": budget, "new_candidate_queries": evaluation,
                          "policies": {}}
            for method in policies:
                t0 = time.perf_counter()
                chosen = choose_candidates(
                    method, features, pred, data["sizes"], family,
                    calibration, candidates, evaluation,
                    # Identical stream/order for every budget makes
                    # random, stratified, D2/k-center and rank selectors
                    # nested. Grid capacity is budget-specific and may
                    # evict previous cells; explicitly NOT nested.
                    seed=int(seed)+37,
                )
                policy_ms = 1000*(time.perf_counter()-t0)
                # In all-architecture mode the calibration architectures
                # are target designs and count toward discovered Pareto. In
                # family-extrapolation mode calibration is outside target;
                # the paid queries are charged but do not count as wins.
                scored = chosen + (
                    calibration.tolist() if family_to_holdout is None else []
                )
                outcome = truth_metrics(
                    true, target, scored,
                    accuracy_floor=accuracy_floor, flash_cap=flash_cap,
                )
                experiment["policies"][method] = {
                    "policy_ms": policy_ms,
                    "shared_feature_and_proxy_fit_ms": proxy_ms,
                    "new_arch_ids": data["ids"][chosen].tolist(),
                    "metrics": outcome,
                }
            trial["budget_results"].append(experiment)
        report["results"].append(trial)
    return report


def summarize(report):
    summary = {}
    for bidx, budget in enumerate(report["total_evaluation_budgets"]):
        summary[str(budget)] = {}
        for method in METHODS:
            rows = [r["budget_results"][bidx]["policies"][method]["metrics"]
                    for r in report["results"]]
            summary[str(budget)][method] = {
                "mean_hypervolume_ratio": float(np.mean([
                    row["hypervolume_ratio"] for row in rows
                ])),
                "mean_front_epsilon_coverage": float(np.mean([
                    row["front_epsilon_coverage"] for row in rows
                ])),
                "mean_exact_front_recall": float(np.mean([
                    row["front_exact_recall"] for row in rows
                ])),
                "mean_feasible_found": float(np.mean([
                    row["discovered_feasible"] for row in rows
                ])),
                "num_seeds": len(rows),
            }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hardware", type=Path, required=True)
    p.add_argument("--accuracies", type=Path, required=True)
    p.add_argument("--holdout-first-op", choices=OPS, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[31,51,73])
    p.add_argument("--fractions", type=float, nargs="+", default=[.01,.05,.10])
    p.add_argument("--calibration", type=int, default=64)
    p.add_argument("--accuracy-floor", type=float, default=90.)
    p.add_argument("--flash-cap", type=float, default=900.)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not all(0 < v <= 0.25 for v in args.fractions):
        p.error("budgets must be in (0, .25]")
    if not 10 <= args.calibration <= 500 or len(args.seeds) < 2:
        p.error("10<=calibration<=500 and at least 2 independent seeds required")
    data = load_nats(args.hardware, args.accuracies)
    report = run_replay(
        data, family_to_holdout=args.holdout_first_op,
        seeds=args.seeds, fractions=args.fractions,
        calibration_count=args.calibration,
        accuracy_floor=args.accuracy_floor, flash_cap=args.flash_cap,
    )
    report["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    report["summary"] = summarize(report)
    report["settings"] = {
        "seeds": args.seeds, "fractions": args.fractions,
        "calibration": args.calibration,
        "holdout_first_op": args.holdout_first_op,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Physical hardware replay complete:", report["split"])
    for budget, methods in report["summary"].items():
        print("ORACLE BUDGET:", budget)
        for method, result in methods.items():
            print(f"  {method:12s} HV {result['mean_hypervolume_ratio']:.4f} "
                  f"epsilon-coverage {result['mean_front_epsilon_coverage']:.4f} "
                  f"front-recall {result['mean_exact_front_recall']:.4f} "
                  f"feasible {result['mean_feasible_found']:.1f}")


if __name__ == "__main__":
    main()
