"""Feistel/cycle-walking permutation invariants and integration tests."""
import pytest
import torch

from svenmeans import SMeansConfig, smeans
from svenmeans._recipe import feistel_epoch_indices, feistel_epoch_key


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 17, 63, 64, 65, 257, 1025, 4096])
@pytest.mark.parametrize("seed", [0, 42])
def test_bijection_on_non_power_of_two_domain(n, seed):
    device = torch.device("cpu")
    key = feistel_epoch_key(n, seed, epoch=1)
    chunks = []
    for begin in range(0, n, 73):
        chunks.append(feistel_epoch_indices(n, begin, min(n, begin + 73), key, device))
    values = torch.cat(chunks)
    assert values.shape == (n,)
    assert torch.equal(torch.sort(values).values, torch.arange(n))
    assert torch.equal(
        values[: min(n, 73)],
        feistel_epoch_indices(n, 0, min(n, 73), key, device),
    )


def test_odd_width_uses_minimum_domain():
    width, _ = feistel_epoch_key(65, 0, 0)
    assert width == 7  # M=128, not 256


def test_per_epoch_seed_changes_order():
    n = 1025
    one = feistel_epoch_indices(n, 0, n, feistel_epoch_key(n, 7, 1), torch.device("cpu"))
    two = feistel_epoch_indices(n, 0, n, feistel_epoch_key(n, 7, 2), torch.device("cpu"))
    assert not torch.equal(one, two)


def test_full_fit_does_not_allocate_full_shuffle(monkeypatch):
    original = torch.randperm
    n = 10_000

    def guarded(size, *args, **kwargs):
        if size >= n:
            raise AssertionError("full-size randperm should not be called")
        return original(size, *args, **kwargs)

    monkeypatch.setattr(torch, "randperm", guarded)
    x = torch.randn(n, 4, generator=torch.Generator().manual_seed(9))
    config = SMeansConfig(shuffle_mode="feistel", init_tile_cols=32)
    first = smeans(x, 24, nepochs=1.5, seed=3, config=config,
                   device="cpu", compute_labels=False, return_info=True)
    second = smeans(x, 24, nepochs=1.5, seed=3, config=config,
                    device="cpu", compute_labels=False, return_info=True)
    assert torch.allclose(first.centroids, second.centroids)
    assert first.inertia is None


def test_invalid_shuffle_mode_fails_fast():
    x = torch.randn(80, 3)
    with pytest.raises(ValueError, match="shuffle_mode"):
        smeans(x, 4, config=SMeansConfig(shuffle_mode="bogus"), device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not installed")
def test_cpu_gpu_index_permutation_parity():
    n = 1025
    key = feistel_epoch_key(n, 13, 0)
    cpu = feistel_epoch_indices(n, 0, n, key, torch.device("cpu"))
    gpu = feistel_epoch_indices(n, 0, n, key, torch.device("cuda")).cpu()
    assert torch.equal(cpu, gpu)
