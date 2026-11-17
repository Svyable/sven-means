"""SOAR-inspired L2 replication: explicit FAISS list entries and fair recall."""
import numpy as np
import pytest

faiss = pytest.importorskip("faiss")
from benchmarks.orthogonal_spill import (
    build_spill_ivf, deduplicate_ranked_ids, evaluate_spill_index,
    secondary_candidates, select_spill_rows,
)


def test_orthogonality_penalty_can_change_second_best_center():
    # At x=(1,0), c0=(0,0) and c1=(2,0) tie for first with c0 first.
    # The second-nearest c1 gives parallel residuals; c2 has a
    # perpendicular residual, selected under orthogonality penalty.
    train = np.array([[1., 0.]], dtype="float32")
    centers = np.array([[0., 0.], [2., 0.], [1., 1.3]], dtype="float32")
    got = secondary_candidates(train, centers, shortlist=3, weight=2.0)
    assert got["primary"].tolist() == [0]
    assert got["nearest"].tolist() == [1]
    assert got["orthogonal"].tolist() == [2]


def test_budget_selection_is_deterministic_and_fair():
    margins = np.array([.5, .02, .1, .4, .01, .8, .15, .9])
    boundary = select_spill_rows(margins, .25)
    assert boundary.tolist() == [4, 1]
    random1 = select_spill_rows(margins, .25, policy="random", seed=77)
    random2 = select_spill_rows(margins, .25, policy="random", seed=77)
    assert np.array_equal(random1, random2)
    assert len(random1) == len(boundary)
    assert len(select_spill_rows(margins, 0)) == 0


@pytest.mark.parametrize("fraction", [0., .25, .5, 1.])
@pytest.mark.parametrize("policy", ["nearest", "orthogonal"])
def test_real_ivf_index_duplicate_cost_and_exact_full_scan(fraction, policy):
    rng = np.random.default_rng(25)
    train = rng.normal(size=(160, 8)).astype("float32")
    test = rng.normal(size=(17, 8)).astype("float32")
    centers = train[:12].copy()
    options = secondary_candidates(train, centers)
    rows = select_spill_rows(
        options["relative_boundary_margin"], fraction, policy="boundary"
    )
    secondary = options[policy][rows] if len(rows) else np.empty(0, dtype=np.int64)
    index, cost = build_spill_ivf(
        train, centers, secondary_ids=secondary, selected_rows=rows
    )
    assert cost > 0
    assert index.ntotal == len(train) + len(rows)
    exact = faiss.IndexFlatL2(train.shape[1])
    exact.add(train)
    truth = exact.search(test, 10)[1]
    report = evaluate_spill_index(
        index, centers, test, truth, original_n=len(train), repeats=1
    )
    assert report["storage_amplification"] == pytest.approx(1+fraction)
    assert report["curve"][-1]["recall_at_10"] == pytest.approx(1.)
    scanned = [x["mean_scanned_entries"] for x in report["curve"]]
    assert scanned == sorted(scanned)


def test_deduplicate_top20_recovers_distinct_candidates():
    raw = np.array([3, 3, 1, 1, 7, 7, 2, 2, -1, -1])
    assert deduplicate_ranked_ids(raw, 3) == [3, 1, 7]
    assert deduplicate_ranked_ids(raw, 10) == [3, 1, 7, 2]


def test_invalid_double_assignment_rejected():
    x = np.arange(30, dtype=np.float32).reshape(10, 3)
    centers = x[:4].copy()
    opts = secondary_candidates(x, centers)
    with pytest.raises(ValueError, match="secondary"):
        build_spill_ivf(
            x, centers, selected_rows=np.array([0], dtype=np.int64),
            secondary_ids=opts["primary"][:1],
        )
    with pytest.raises(ValueError):
        select_spill_rows(np.array([1., 2., 3.]), 1.01)
    with pytest.raises(ValueError):
        secondary_candidates(x, centers, weight=-1.)
