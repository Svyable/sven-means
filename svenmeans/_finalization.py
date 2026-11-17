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
    donor_retention: float = 0.0,
) -> None:
    """Discard target evidence, retain a fraction of the donor's old history.

    Donor retention=0 reproduces reset-both; 1 resets *only* the revived
    target. Intermediate values discount stale parent statistics without
    discarding all observations. This is experimental and not a guarantee.
    """
    if not 0.0 <= float(donor_retention) <= 1.0:
        raise ValueError("donor_retention must lie in [0, 1]")
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
    sources = donors.to(dtype=torch.int64)
    changed = targets.to(dtype=torch.int64)
    if donor_retention == 0.0:
        pass_sums.index_fill_(0, sources, 0.0)
        pass_counts.index_fill_(0, sources, 0.0)
    elif donor_retention < 1.0:
        # Advanced indexing returns a copy; index_copy_ is required to persist.
        pass_sums.index_copy_(0, sources, pass_sums.index_select(0, sources) * donor_retention)
        pass_counts.index_copy_(0, sources, pass_counts.index_select(0, sources) * donor_retention)
    pass_sums.index_fill_(0, changed, 0.0)
    pass_counts.index_fill_(0, changed, 0.0)
