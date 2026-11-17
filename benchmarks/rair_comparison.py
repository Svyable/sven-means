"""Experiment 012: SIGMOD'26 RAIRS AIR vs SOAR L2, naive, plain IVF.

Explicit prior art: Yang & Chen, SIGMOD 2026, arXiv:2601.07183.
This benchmark reproduces only the AIR *centroid assignment formula*
in ordinary FAISS IndexIVFFlat; it does NOT reproduce RAIRS's shared-cell
SEIL layout or PQ FastScan. All candidate IDs are evaluated by REAL
FAISS search against exhaustive ground truth, with exact per-entry costs.

Fixed experimental design:
  L2 center producers: Sven global, Sven local, Meta FAIR NumpyEmeans2,
                       FAISS Lloyd2
  Duplication quota: 25% or 50% of train vectors selected by smallest
                     first-to-second Voronoi margin, using train only
  Secondary choices: naive 2nd closest, SOAR L2 lambda=2 from earlier
                     pilot, SRAIR lambda=.5 shortlist=10, RAIR lambda=.5
                     shortlist=10 with abstention allowed (fewer entries)
  Two independent data seeds, 3 optimizer seeds, real LSA and Zipf
  Equal expected *scanned index entries* (not fixed nprobe), actual
  serialized bytes, query latency, index construction time, raw ID
  recall, tie-aware radius recall, and 100% full scan invariants.
No model tuning or evaluation-query leakage. No original novelty claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import faiss
import numpy as np
import torch

from benchmarks.orthogonal_spill_study import fit_centers, make_truth
from benchmarks.orthogonal_spill import (
    build_spill_ivf, evaluate_spill_index, secondary_candidates,
    select_spill_rows,
)
from benchmarks.rair_assignment import air_secondary
from benchmarks.release_candidate import dataset_arrays
from benchmarks.scan_budget import interpolated_recall_at_budget

MODELS = ("sven_global","sven_local","faiss_lloyd2","meta_emeans2")
FRACTIONS = (0.25,0.5)
BUDGETS = (0.05,0.10,0.20,0.30,0.50)
METHODS = ("nearest","soar","srair","rair")


def make_arm(
    train, centers, query, truth_id, truth_dist, *,
    chosen, secondary, name, measured_selection_ms, repeats,
):
    if secondary is None:
        index, build_ms = build_spill_ivf(train, centers)
    else:
        use = secondary != -1
        ids = chosen[use]
        to = secondary[use]
        index, build_ms = build_spill_ivf(
            train, centers,
            selected_rows=ids, secondary_ids=to
        )
    result = evaluate_spill_index(
        index,centers,query,truth_id,truth_dist,
        original_n=len(train),repeats=repeats
    )
    return {
        "arm":name,
        "index_build_ms":float(build_ms),
        "secondary_preparation_ms":float(measured_selection_ms),
        "actual_replica_count":int(index.ntotal-len(train)),
        **result,
        "recall_by_scanned_entries":{
            str(b):interpolated_recall_at_budget(result["curve"],b)
            for b in BUDGETS
        }
    }


def eval_model(train, query, exact_id, exact_dist, args, *, model, model_seed, data_seed):
    centers, fit_ms = fit_centers(train,args.k,args.epochs,model_seed,model)
    started = time.perf_counter()
    nearest = secondary_candidates(train,centers,weight=2.0,shortlist=10)
    inverse = air_secondary(train,centers,air_lambda=.5,shortlist=10)
    select_prep_ms = (time.perf_counter()-started)*1000
    baseline = make_arm(
        train,centers,query,exact_id,exact_dist,chosen=np.empty(0,dtype=np.int64),
        secondary=None,name="plain",measured_selection_ms=0.,repeats=args.repeats
    )
    records={"plain":baseline}
    for fraction in FRACTIONS:
        selected = select_spill_rows(
            nearest["relative_boundary_margin"], fraction, policy="boundary"
        )
        for method in METHODS:
            if method=="nearest":
                sec=nearest["nearest"][selected]
            elif method=="soar":
                sec=nearest["orthogonal"][selected]
            elif method=="srair":
                sec=inverse["strict"][selected]
            else:
                sec=inverse["optional"][selected]
                sec=np.where(
                    sec==inverse["primary"][selected],-1,sec
                )
            name=f"{method}_{int(fraction*100)}"
            row=make_arm(
                train,centers,query,exact_id,exact_dist,
                chosen=selected,secondary=np.asarray(sec,dtype=np.int64),
                name=name,measured_selection_ms=select_prep_ms,
                repeats=args.repeats,
            )
            records[name]=row

    paired={}
    for fraction in FRACTIONS:
        suffix=str(int(fraction*100))
        for method in METHODS:
            candidate=records[f"{method}_{suffix}"]
            for control_name in (f"nearest_{suffix}",f"soar_{suffix}",f"srair_{suffix}","plain"):
                if candidate["arm"]==control_name:
                    continue
                control=records[control_name]
                key=f"{candidate['arm']} vs {control_name}"
                item={
                    "candidate":candidate["arm"],"control":control_name,
                    "relative_serialized_bytes":candidate["serialized_bytes"]/
                                                control["serialized_bytes"],
                    "relative_total_entries":candidate["num_stored_entries"]/
                                             control["num_stored_entries"],
                    "search_ms_per_query_delta_at_nprobe1":
                        candidate["curve"][0]["median_search_ms_per_query"]
                        -control["curve"][0]["median_search_ms_per_query"],
                }
                for b in BUDGETS:
                    a=candidate["recall_by_scanned_entries"][str(b)]
                    z=control["recall_by_scanned_entries"][str(b)]
                    item[f"gain_points_scan_{b}"]=(100*(a-z)
                        if a is not None and z is not None else None)
                paired[key]=item
    return {
        "center_model":model,"fit_seed":model_seed,
        "data_seed":data_seed,
        "centroid_fit_ms":fit_ms,
        "air_abstention_on_all_train":inverse["abstention_fraction"],
        "raw_arms":records,"paired":paired,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=("news_lsa","zipf","rotated"),required=True)
    p.add_argument("--n",type=int,default=4000)
    p.add_argument("--test-n",type=int,default=1200)
    p.add_argument("--queries",type=int,default=96)
    p.add_argument("--d",type=int,default=96)
    p.add_argument("--k",type=int,default=64)
    p.add_argument("--epochs",type=float,default=2.)
    p.add_argument("--data-seeds",type=int,default=2)
    p.add_argument("--model-seeds",type=int,default=3)
    p.add_argument("--seed-offset",type=int,default=4111)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if min(a.n,a.test_n,a.queries,a.d,a.k,a.data_seeds,a.model_seeds,a.repeats,a.threads)<1:
        p.error("integer settings must all be positive")
    if a.k>=a.n or a.k<2 or a.epochs<=0:
        p.error("require 2<=k<n and positive epochs")
    torch.set_num_threads(a.threads)
    faiss.omp_set_num_threads(a.threads)
    records=[]
    for i in range(a.data_seeds):
        dataset_seed=a.seed_offset+i
        train_t,test_t,provenance=dataset_arrays(
            a.dataset,dataset_seed,a.n,a.test_n,a.d
        )
        train=np.ascontiguousarray(train_t.numpy(),dtype=np.float32)
        query=np.ascontiguousarray(test_t[:a.queries].numpy(),dtype=np.float32)
        exact_dist, exact_id=make_truth(train,query)
        for seed in range(a.model_seeds):
            for model in MODELS:
                records.append(eval_model(
                    train,query,exact_id,exact_dist,a,
                    model=model,model_seed=seed,data_seed=dataset_seed
                ))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps({
            "experiment":"RAIR/SRAIR vs SOAR vs naive; formula-only real FAISS",
            "prior_art":"Yang Chen RAIRS SIGMOD 2026 arXiv 2601.07183",
            "source":"https://github.com/schencoding/rairs",
            "not_implemented":"SEIL shared-cell layout and authors' PQ-FastScan C++",
            "settings":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            "environment":{
                "python":platform.python_version(),
                "numpy":np.__version__,
                "faiss":getattr(faiss,"__version__","unknown"),
                "torch":torch.__version__,
            },
            "current_corpus_provenance":{
                **provenance,
                "data_seed":dataset_seed,
                "train_sha256":hashlib.sha256(train.tobytes()).hexdigest(),
                "test_sha256":hashlib.sha256(query.tobytes()).hexdigest(),
            },
            "records":records,
        },indent=2)+"\n")
    for model in MODELS:
        item=[x for x in records if x["center_model"]==model]
        print("==",a.dataset,model,"==")
        for method in ("srair","rair"):
            for control in ("soar","nearest","plain"):
                key=(f"{method}_25 vs "
                     f"{control}_25" if control!="plain" else
                     f"{method}_25 vs plain")
                p=[row["paired"][key]["gain_points_scan_0.1"] for row in item]
                vals=[v for v in p if v is not None]
                bytes_ratio=statistics.mean(
                    row["paired"][key]["relative_serialized_bytes"]
                    for row in item
                )
                print(key,"recall pp at 10% scanned entries",
                      statistics.mean(vals) if vals else None,
                      "pairs",len(vals),
                      "relative index bytes",round(bytes_ratio,4))


if __name__=="__main__":
    main()
