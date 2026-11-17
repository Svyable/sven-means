"""Validate RAIRS published Euclidean AIR scoring and storage policy."""
import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from benchmarks.rair_assignment import air_secondary
from benchmarks.orthogonal_spill import (
    build_spill_ivf, secondary_candidates, select_spill_rows,
    evaluate_spill_index,
)


def test_strict_air_matches_direct_formula_and_is_not_orthogonal():
    train = np.array([[1., 0.], [3., 0.]], dtype="float32")
    centers = np.array([[0.,0.],[2.,0.],[1.,1.4],[5.,0.]], dtype="float32")
    got = air_secondary(train, centers, air_lambda=.5, shortlist=4)
    idx = got["shortlist_centers"]
    for i in range(len(train)):
        p = centers[got["primary"][i]] - train[i]
        loss = np.array([
            np.dot(centers[j]-train[i],centers[j]-train[i])
            + .5 * np.dot(p,centers[j]-train[i]) for j in idx[i]
        ])
        assert np.allclose(loss,got["air_scores"][i],atol=1e-5)
        assert got["strict"][i] == idx[i, 1+np.argmin(loss[1:])]
        assert got["optional"][i] == idx[i,np.argmin(loss)]
        assert got["strict"][i] != got["primary"][i]


def test_zero_lambda_is_naive_second_nearest():
    g = np.random.default_rng(10)
    train = g.normal(size=(80,11)).astype("float32")
    centers = g.normal(size=(15,11)).astype("float32")
    opts = secondary_candidates(train, centers, shortlist=10)
    air = air_secondary(train, centers, air_lambda=0.0, shortlist=10)
    assert np.array_equal(air["strict"],opts["nearest"])


def test_optional_can_abstain_for_points_at_primary_center():
    centers = np.array([[0.,0.],[50.,0.],[0.,50.]],dtype="float32")
    train = np.array([[0.,0.],[50.,0.]],dtype="float32")
    result = air_secondary(train,centers)
    assert result["abstention_fraction"] == pytest.approx(1.0)
    assert np.array_equal(result["optional"], result["primary"])
    assert np.all(result["strict"] != result["primary"])


def test_no_query_leakage_and_real_faiss_fullscan():
    rng = np.random.default_rng(13)
    train = rng.normal(size=(125,8)).astype("float32")
    test = rng.normal(size=(17,8)).astype("float32")
    centers = train[:10].copy()
    output = air_secondary(train,centers,shortlist=10)
    options = secondary_candidates(train,centers,shortlist=10)
    selected = select_spill_rows(options["relative_boundary_margin"],.25)
    secondary = output["optional"][selected]
    take = secondary != output["primary"][selected]
    index, _ = build_spill_ivf(
        train,centers,selected_rows=selected[take],
        secondary_ids=secondary[take],
    )
    assert index.ntotal == len(train) + int(take.sum())
    oracle = faiss.IndexFlatL2(train.shape[1])
    oracle.add(train)
    dist, ids = oracle.search(test,10)
    actual = evaluate_spill_index(
        index,centers,test,ids,dist,original_n=len(train),repeats=1
    )
    assert actual["curve"][-1]["tie_aware_recall_at_10"] == pytest.approx(1.)


def test_bad_lambda_fails():
    with pytest.raises(ValueError,match="lambda"):
        air_secondary(np.ones((10,2)),np.zeros((3,2)),air_lambda=-1)
    with pytest.raises(ValueError,match="shortlist"):
        air_secondary(np.ones((10,2)),np.zeros((3,2)),shortlist=1)
