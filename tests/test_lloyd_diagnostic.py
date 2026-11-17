"""Lloyd diagnostic: train-only deterministic exact monotone centroid repair."""
import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from benchmarks.lloyd_diagnostic import assign_faiss, lloyd_trace


def test_full_lloyd_updates_do_not_increase_train_sse():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(220, 9)).astype("float32")
    centers = x[:20].copy()
    trace = lloyd_trace(x, centers, steps=4)
    assert [r["steps"] for r in trace] == list(range(5))
    errors = [r["train_mse"] for r in trace]
    assert all(b <= a+1e-6 for a,b in zip(errors,errors[1:])), errors
    assert trace[0]["refinement_ms"] == 0.0
    assert np.array_equal(trace[0]["centers"], centers)
    second=lloyd_trace(x, centers, steps=4)
    assert all(np.array_equal(a["centers"],b["centers"]) for a,b in zip(trace,second))


def test_empty_clusters_preserve_old_centers():
    x = np.array([[0.],[0.],[3.],[3.]],dtype=np.float32)
    c = np.array([[0.],[3.],[100.]],dtype=np.float32)
    trace=lloyd_trace(x,c,steps=2)
    assert all(t["centers"][2,0] == 100 for t in trace)
    assert trace[0]["empty_centers"] >= 1


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        lloyd_trace(np.zeros((3,2)), np.zeros((2,2)), steps=-1)
    with pytest.raises(ValueError):
        lloyd_trace(np.zeros((3,2)), np.zeros((2,3)), steps=1)


def test_faiss_assignment_is_correct():
    x=np.array([[0.,0.],[3.,0.],[9.,0.]],dtype="float32")
    c=np.array([[0.,0.],[10.,0.]],dtype="float32")
    ids,errors=assign_faiss(x,c)
    assert ids.tolist() == [0,0,1]
    assert np.allclose(errors,[0.,9.,1.])
