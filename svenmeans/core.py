"""Core PyTorch and NumPy tensor interface for s-means clustering."""

from dataclasses import dataclass
import time
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ._cpu import fit_cpu, tiled_assign
from ._gpu import HAS_TRITON, assign_gpu, fit_gpu
from ._recipe import SMeansConfig

ArrayLike = Union[torch.Tensor, np.ndarray]


@dataclass
class SMeansResult:
    """Output container returned by `smeans(..., return_info=True)`."""

    centroids: torch.Tensor
    labels: Optional[torch.Tensor]
    inertia: Optional[float]
    mse: Optional[float]
    time_s: float
    n_iter: int
    dead_clusters: int
    total_splits: int
    backend: str
    dtype: str
    guarded_candidate_checks: int = 0
    net_proposed: int = 0
    net_accepted: int = 0
    net_rejected: int = 0
    net_validation_rows: int = 0
    net_mean_accepted_gain: float = 0.0
    net_mean_accepted_lcb: float = 0.0


def _resolve_device(x: torch.Tensor, device: str) -> torch.device:
    if device == "auto":
        if x.is_cuda:
            return x.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def assign_clusters(
    x: ArrayLike,
    centroids: torch.Tensor,
    backend: str = "auto",
    dtype: str = "auto",
    tile_rows: int = 512,
    tile_cols: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign each row of `x` to its nearest centroid and return `(labels, sq_dists)`."""
    xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x), dtype=torch.float32)
    if xt.device != centroids.device:
        xt = xt.to(centroids.device)
    # Translation-invariant assignment avoids catastrophic cancellation in
    # ||x||^2 + ||c||^2 - 2*x@c when coordinates have large common offsets.
    anchor = centroids[:1].detach().to(dtype=torch.float32)
    xt_f = (xt.detach().to(dtype=torch.float32) - anchor).contiguous()
    c_f = (centroids.detach().to(dtype=torch.float32) - anchor).contiguous()
    if xt_f.is_cuda:
        return assign_gpu(xt_f, c_f, backend=backend, dtype_mode=dtype)
    return tiled_assign(xt_f, c_f, tile_rows=tile_rows, tile_cols=tile_cols)


def smeans(
    x: ArrayLike,
    n_clusters: int,
    nepochs: float = 3.0,
    seed: int = 0,
    device: str = "auto",
    backend: str = "auto",
    dtype: str = "auto",
    config: Optional[SMeansConfig] = None,
    compute_labels: bool = True,
    return_info: bool = False,
) -> Union[torch.Tensor, SMeansResult]:
    """Cluster tensor or NumPy array `x` into `n_clusters` centroids using s-means.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray of shape (n_samples, n_features)
        Input dataset. CUDA tensors execute end-to-end on device without host copies.
    n_clusters : int
        Number of clusters `k`.
    nepochs : float, default=3.0
        Training budget in data epochs (supports fractional values such as `0.5` or `1.5`).
    seed : int, default=0
        Random seed controlling subsampled D^1.5 seeding, mini-batch shuffling, and splits.
    device : str, default="auto"
        Target execution device (`"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"`, ...).
    backend : str, default="auto"
        CUDA kernel backend (`"auto"`, `"triton"`, `"native"`). Ignored on CPU.
    dtype : str, default="auto"
        CUDA assignment precision (`"auto"`, `"float32"`, `"bfloat16"`, `"float16"`).
    config : SMeansConfig, optional
        Fine-grained algorithmic hyperparameters.
    compute_labels : bool, default=True
        Whether to compute full-dataset labels and inertia after training.
    return_info : bool, default=False
        If True, return an `SMeansResult` dataclass; otherwise return `centroids`.
    """
    cfg = config if config is not None else SMeansConfig()
    if not 0.0 <= float(cfg.donor_pass_retention) <= 1.0:
        raise ValueError("donor_pass_retention must be between 0 and 1")
    if isinstance(x, torch.Tensor):
        xt = x
    else:
        xt = torch.as_tensor(np.asarray(x), dtype=torch.float32)

    if xt.ndim != 2:
        raise ValueError(f"Expected 2D input tensor, got shape {tuple(xt.shape)}")
    n_samples = int(xt.shape[0])
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    if n_samples < n_clusters:
        raise ValueError(f"n_samples ({n_samples}) must be >= n_clusters ({n_clusters})")

    prep_start = time.perf_counter()
    target_dev = _resolve_device(xt, device)
    if xt.device != target_dev:
        xt = xt.to(target_dev)
    # Optional opt-in for shifted/large-magnitude data; extra O(n*d) storage.
    origin = None
    if cfg.center_data:
        origin = xt.to(dtype=torch.float32).mean(dim=0)
        xt = (xt.to(dtype=torch.float32) - origin).contiguous()

    if target_dev.type == "cuda":
        raw = fit_gpu(
            xt,
            int(n_clusters),
            nepochs=float(nepochs),
            seed=int(seed),
            backend=backend,
            dtype_mode=dtype,
            cfg=cfg,
            compute_labels=compute_labels,
        )
    else:
        raw = fit_cpu(
            xt,
            int(n_clusters),
            nepochs=float(nepochs),
            seed=int(seed),
            cfg=cfg,
            compute_labels=compute_labels,
        )

    if origin is not None:
        raw["centroids"] = raw["centroids"] + origin[None, :]
        if xt.is_cuda:
            torch.cuda.synchronize(xt.device)
        raw["time_s"] = time.perf_counter() - prep_start

    if not return_info:
        return raw["centroids"]

    mse = (raw["inertia"] / max(1, n_samples)) if raw["inertia"] is not None else None
    return SMeansResult(
        centroids=raw["centroids"],
        labels=raw["labels"],
        inertia=raw["inertia"],
        mse=mse,
        time_s=raw["time_s"],
        n_iter=raw["n_iter"],
        dead_clusters=raw["dead_clusters"],
        total_splits=raw["total_splits"],
        backend=raw["backend"],
        dtype=raw["dtype"],
        guarded_candidate_checks=raw.get("guarded_candidate_checks", 0),
        net_proposed=raw.get("net_proposed", 0),
        net_accepted=raw.get("net_accepted", 0),
        net_rejected=raw.get("net_rejected", 0),
        net_validation_rows=raw.get("net_validation_rows", 0),
        net_mean_accepted_gain=raw.get("net_mean_accepted_gain", 0.0),
        net_mean_accepted_lcb=raw.get("net_mean_accepted_lcb", 0.0),
    )


__all__ = ["HAS_TRITON", "SMeansConfig", "SMeansResult", "assign_clusters", "smeans"]
