"""Protect progressive final-pass statistics against stale pre-split assignments.

The same final-pass accumulator index can represent two physically different
clusters before and after a donor/target split. Using one average across those
states can pull the resulting centroid back toward its pre-split position.

This helper deliberately resets only affected indices; all unrelated clusters
retain their final-pass evidence. This changes output quality and therefore
remains an opt-in ablation until paired held-out benchmarks justify it.
"""

import torch


@torch.no_grad()
def invalidate_split_statistics(
    pass_sums: torch.Tensor,
    pass_counts: torch.Tensor,
    donors: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Zero both newly moved centers' pass evidence, preserving all others."""
    if donors.numel() == 0:
        return
    if donors.shape != targets.shape:
        raise ValueError("donors and targets must have the same shape")
    if pass_sums.ndim != 2 or pass_counts.ndim != 1:
        raise ValueError("expected [k,d] sums and [k] counts")
    if pass_sums.shape[0] != pass_counts.shape[0]:
        raise ValueError("pass-sum and pass-count sizes must match")
    if donors.device != pass_sums.device or targets.device != pass_sums.device:
        raise ValueError("split indices and pass statistics must share a device")
    changed = torch.cat((donors, targets)).to(dtype=torch.int64)
    pass_sums.index_fill_(0, changed, 0.0)
    pass_counts.index_fill_(0, changed, 0.0)
