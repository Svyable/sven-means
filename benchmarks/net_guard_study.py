"""Experiment 011: independent holdout net centroid-recycling gate.

Every candidate net gate uses a validation split *reserved before fit*.
For a FAIR four/four comparison, ordinary controls are trained on the
exact fit-only subset from reserve_validation(...); net arms receive
full data and reserve that SAME subset themselves. All methods are
evaluated on separate, untouched test data.

No formal confidence coverage is asserted. Repeated decisions adapt to
the same reserved split. Synthetic and LSA data here are development
datasets, not independent neural confirmation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import sklearn
import torch

from benchmarks.rare_guard_study import rare_islands
from benchmarks.release_candidate import dataset_arrays
from benchmarks.scan_budget import scan_curve, interpolated_recall_at_budget
from svenmeans import SMeansConfig, smeans, assign_clusters
from svenmeans._net_guard import reserve_validation

ARMS=(
    ("global_plain","global","none","none"),
    ("global_distance","global","distance","none"),
    ("global_net","global","none","net_holdout"),
    ("global_combined","global","distance","net_holdout"),
    ("local_plain","local_direction","none","none"),
    ("local_distance","local_direction","distance","none"),
    ("local_net","local_direction","none","net_holdout"),
    ("local_combined","local_direction","distance","net_holdout"),
)
BUDGETS=(0.05,0.10,0.25)


def benchmark_dataset(name,args):
    split_results=[]
    for dseed in range(args.data_seeds):
        seed=args.seed_offset+dseed
        if name=="rare_islands":
            full,test,source=rare_islands(args.n,args.test_n,args.d,seed)
        else:
            full,test,source=dataset_arrays(
                name,seed,args.n,args.test_n,args.d
            )
        provenance={
            "data_seed":seed,**source,"full_shape":list(full.shape),
            "test_shape":list(test.shape),
            "full_sha256":hashlib.sha256(full.numpy().tobytes()).hexdigest(),
            "test_sha256":hashlib.sha256(test.numpy().tobytes()).hexdigest(),
        }
        rows=[]
        for mseed in range(args.model_seeds):
            fit_only,validation=reserve_validation(
                full,mseed,args.net_fraction
            )
            for index in range(len(ARMS)):
                arm,geometry,protection,net=ARMS[(index+mseed)%len(ARMS)]
                x=full if net=="net_holdout" else fit_only
                cfg=SMeansConfig(
                    revive_geometry=geometry,
                    revive_protection=protection,
                    revive_acceptance=net,
                    net_validation_fraction=args.net_fraction,
                    net_lcb_z=args.lcb_z,
                    net_max_proposals=args.max_proposals,
                    local_observe_every=4,
                )
                t0=time.perf_counter()
                res=smeans(
                    x,args.k,nepochs=args.epochs,seed=mseed,
                    config=cfg,device="cpu",return_info=True,
                    compute_labels=False,
                )
                fit_ms=(time.perf_counter()-t0)*1000
                _,distance=assign_clusters(test,res.centroids)
                curve=scan_curve(fit_only,test,res.centroids,query_count=96)
                row={
                    "arm":arm,"model_seed":mseed,
                    "fit_only_train_rows":len(fit_only),
                    "validation_rows":len(validation),
                    "test_mse":float(distance.double().mean()),
                    "fit_ms":fit_ms,
                    "total_splits":res.total_splits,
                    "distance_guard_checks":res.guarded_candidate_checks,
                    "net_proposed":res.net_proposed,
                    "net_accepted":res.net_accepted,
                    "net_rejected":res.net_rejected,
                    "net_mean_accepted_gain":res.net_mean_accepted_gain,
                    "net_mean_accepted_lcb":res.net_mean_accepted_lcb,
                    "curve":curve,
                    "budgeted_recall":{
                        str(b):interpolated_recall_at_budget(curve,b) for b in BUDGETS
                    },
                }
                rows.append(row)
        pairs={}
        for geo in ("global","local"):
            base={x["model_seed"]:x for x in rows if x["arm"]==geo+"_plain"}
            for suffix in ("distance","net","combined"):
                arm=geo+"_"+suffix
                matched=[]
                for candidate in (r for r in rows if r["arm"]==arm):
                    control=base[candidate["model_seed"]]
                    pair={
                        "model_seed":candidate["model_seed"],
                        "mse_gain_pct":100*(1-candidate["test_mse"]/control["test_mse"]),
                        "fit_runtime_ratio":candidate["fit_ms"]/control["fit_ms"],
                        "accepted":candidate["net_accepted"],
                        "rejected":candidate["net_rejected"],
                        "total_splits":candidate["total_splits"],
                    }
                    for b in BUDGETS:
                        rc=candidate["budgeted_recall"][str(b)]
                        rb=control["budgeted_recall"][str(b)]
                        pair[f"recall_{b}_gain_pp"]=(
                            100*(rc-rb) if rc is not None and rb is not None else None
                        )
                    matched.append(pair)
                summary={
                    "mean_mse_gain_pct":statistics.mean(p["mse_gain_pct"] for p in matched),
                    "mse_wins":sum(p["mse_gain_pct"]>0 for p in matched),
                    "median_fit_ratio":statistics.median(
                        p["fit_runtime_ratio"] for p in matched
                    ),
                    "mean_net_accept":statistics.mean(p["accepted"] for p in matched),
                    "mean_net_reject":statistics.mean(p["rejected"] for p in matched),
                    "mean_splits":statistics.mean(p["total_splits"] for p in matched),
                }
                for b in BUDGETS:
                    values=[p[f"recall_{b}_gain_pp"] for p in matched
                            if p[f"recall_{b}_gain_pp"] is not None]
                    summary[f"recall_{b}_gain_pp"]=(
                        statistics.mean(values) if values else None
                    )
                    summary[f"recall_{b}_num_comparable"]=len(values)
                pairs[arm]={"summary":summary,"individual":matched}
        split_results.append({"provenance":provenance,"pairs":pairs,"raw":rows})
    aggregate={}
    for arm in ("global_distance","global_net","global_combined",
                "local_distance","local_net","local_combined"):
        matched=[p for result in split_results
                 for p in result["pairs"][arm]["individual"]]
        summary={
            "mean_mse_gain_pct":statistics.mean(p["mse_gain_pct"] for p in matched),
            "mse_wins":sum(p["mse_gain_pct"]>0 for p in matched),
            "pairs":len(matched),
            "median_fit_ratio":statistics.median(p["fit_runtime_ratio"] for p in matched),
            "mean_net_accept":statistics.mean(p["accepted"] for p in matched),
            "mean_net_reject":statistics.mean(p["rejected"] for p in matched),
        }
        for b in BUDGETS:
            values=[p[f"recall_{b}_gain_pp"] for p in matched
                    if p[f"recall_{b}_gain_pp"] is not None]
            summary[f"recall_{b}_gain_pp"]=(statistics.mean(values) if values else None)
            summary[f"recall_{b}_num_comparable"]=len(values)
        aggregate[arm]=summary
    return {"dataset":name,"aggregate":aggregate,"data_splits":split_results}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",default="all-offline",
                   choices=("all-offline","rare_islands","rotated",
                            "zipf","digits","cancer","news_lsa"))
    p.add_argument("--n",type=int,default=4000)
    p.add_argument("--test-n",type=int,default=1200)
    p.add_argument("--d",type=int,default=96)
    p.add_argument("--k",type=int,default=64)
    p.add_argument("--epochs",type=float,default=2.)
    p.add_argument("--data-seeds",type=int,default=2)
    p.add_argument("--model-seeds",type=int,default=3)
    p.add_argument("--seed-offset",type=int,default=4907)
    p.add_argument("--net-fraction",type=float,default=.10)
    p.add_argument("--lcb-z",type=float,default=1.645)
    p.add_argument("--max-proposals",type=int,default=4)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if min(a.n,a.test_n,a.d,a.k,a.data_seeds,a.model_seeds,a.threads)<1:
        p.error("all counts must be positive")
    if a.k>=int(.9*a.n) or a.epochs<=0:
        p.error("k must be <90% n and epochs positive")
    torch.set_num_threads(a.threads)
    names=["rare_islands","zipf","digits","cancer"] if a.dataset=="all-offline" else [a.dataset]
    results=[benchmark_dataset(n,a) for n in names]
    report={
        "environment":{
            "python":platform.python_version(),"torch":torch.__version__,
            "numpy":np.__version__,"sklearn":sklearn.__version__,
        },
        "settings":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        "results":results,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    for result in results:
        print("===",result["dataset"],"===")
        for arm,stat in result["aggregate"].items():
            print(f"{arm:17s} MSE {stat['mean_mse_gain_pct']:+7.2f}% "
                  f"wins {stat['mse_wins']}/{stat['pairs']} "
                  f"recall10 {stat['recall_0.1_gain_pp']} "
                  f"fit {stat['median_fit_ratio']:.2f}x "
                  f"accept {stat['mean_net_accept']:.1f} "
                  f"reject {stat['mean_net_reject']:.1f}")


if __name__=="__main__":
    main()
