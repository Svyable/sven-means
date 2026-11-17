"""Experimental cluster-local, rank-one residual geometry for dead-centroid revival.

Maintains a streaming direction-gradient sketch and two scalar residual
moments per centroid, not a full d-by-d covariance matrix. Covariance
power iteration is the historical default; the experimental "split_gain"
direction follows the gradient of the *actual symmetric split objective*.
This module is CPU-only until CUDA and Triton are independently tested.
"""
from __future__ import annotations

import torch

from ._recipe import make_generator


class DirectionalRevival:
    """Online rank-one covariance sketch and distortion-gated split proposals.

    For a unit direction q, observed residual r=x-c has projected scalar
    z=<r,q>.  The rank-one power iteration uses E[r z] = E[rr^T]q,
    which learns a principal residual direction without O(k*d*d) memory.

    Splitting one donor into c+rho*q and c-rho*q changes its conditional SSE
    by (2*rho*|z| - rho**2) per assigned point. For optimal radius rho=E|z|,
    the gain is (E|z|)^2. Hence the objective-aligned unit-vector gradient
    is g=E[sign(r.q)*r], NOT the covariance power direction g=E[(r.q)*r].
    Both are classic PCA-like methods; the integrated split policy, not the
    underlying L1 principal-component idea, is being experimentally tested.

    EMA moments are an in-sample moving proxy; shifting assignments and
    centroids invalidate guarantees about unseen points.
    """

    def __init__(
        self,
        k: int,
        d: int,
        *,
        seed: int,
        device: torch.device,
        beta: float = 0.85,
        min_observations: int = 16,
        direction_objective: str = "covariance",
    ) -> None:
        if k < 1 or d < 1 or not 0.0 <= beta < 1.0 or min_observations < 1:
            raise ValueError("k,d,min_observations >= 1 and 0 <= beta < 1 required")
        if direction_objective not in ("covariance", "split_gain"):
            raise ValueError("direction_objective must be 'covariance' or 'split_gain'")
        self.direction_objective = direction_objective
        self.beta = float(beta)
        self.min_observations = int(min_observations)

        q = torch.randn((k, d), dtype=torch.float32, device=device,
                        generator=make_generator(device, seed ^ 0x5EA1C0DE))
        self.q = torch.nn.functional.normalize(q, dim=1, eps=1e-12)
        self.mean_abs = torch.zeros(k, dtype=torch.float32, device=device)
        self.mean_sq = torch.zeros(k, dtype=torch.float32, device=device)
        self.observed = torch.zeros(k, dtype=torch.float32, device=device)

        self._cov_q = torch.zeros((k, d), dtype=torch.float32, device=device)
        self._abs_sum = torch.zeros(k, dtype=torch.float32, device=device)
        self._sq_sum = torch.zeros(k, dtype=torch.float32, device=device)

    @torch.no_grad()
    def observe(
        self,
        xb: torch.Tensor,
        labels: torch.Tensor,
        centroids: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        """Update projected moments before centroids move, using existing labels.

        Additional time O(batch*d + k*d), memory O(batch*d + k*d).
        Only centroids touched in this batch update their estimates.
        """
        residual = xb.float() - centroids[labels].float()
        previous_projection = (residual * self.q[labels]).sum(dim=1)
        self._cov_q.zero_()
        touched = counts > 0
        if not bool(touched.any().item()):
            return
        denom = counts[touched].clamp_min(1)

        # The historical covariance power iteration is g=E[r*(r.q)].
        # The split-gain objective G(q)=(E|r.q|)^2 has tangent gradient
        # proportional to E[sign(r.q)*r] for nonzero projections.
        weight = (
            previous_projection.sign()
            if self.direction_objective == "split_gain"
            else previous_projection
        )
        self._cov_q.index_add_(0, labels, residual * weight[:, None])
        previous = self.q[touched]
        gradient = self._cov_q[touched] / denom[:, None]
        norm = gradient.norm(dim=1, keepdim=True)
        candidate = torch.where(
            norm > 1e-12, gradient / norm.clamp_min(1e-12), previous
        )
        mixed = self.beta * previous + (1.0 - self.beta) * candidate
        mixed_norm = mixed.norm(dim=1, keepdim=True)
        self.q[touched] = torch.where(
            mixed_norm > 1e-12, mixed / mixed_norm.clamp_min(1e-12), previous
        )

        # For the split-gain variant, measure projected moments ALONG THE
        # UPDATED DIRECTION, not the stale pre-update projection. This extra
        # O(batch*d) operation is charged by the benchmark. Preserve the
        # original historical covariance path using pre-update projections.
        if self.direction_objective == "split_gain":
            projection = (residual * self.q[labels]).sum(dim=1)
        else:
            projection = previous_projection
        self._abs_sum.zero_()
        self._sq_sum.zero_()
        self._abs_sum.scatter_add_(0, labels, projection.abs())
        self._sq_sum.scatter_add_(0, labels, projection.square())
        self.observed[touched] += counts[touched]
        self.mean_abs[touched] = (
            self.beta * self.mean_abs[touched]
            + (1.0 - self.beta) * self._abs_sum[touched] / denom
        )
        self.mean_sq[touched] = (
            self.beta * self.mean_sq[touched]
            + (1.0 - self.beta) * self._sq_sum[touched] / denom
        )

    @torch.no_grad()
    def propose(
        self,
        donors: torch.Tensor,
        n_requested: int,
        usage: torch.Tensor,
        split_mult: float,
        gain_floor: float = 0.0,
        radius_policy: str = "rms_scaled",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return eligible donor IDs and unit-direction-scaled perturbations.

        Score = usage * max(0, 2*rho*E|z| - rho**2), using either
        split_mult * sqrt(E[z**2]) or E|z| for the split radius. This
        proxies SSE reduction rather than using raw donor counts.
        """
        if n_requested < 0 or split_mult < 0 or gain_floor < 0:
            raise ValueError("n_requested, split_mult, gain_floor must be nonnegative")
        if radius_policy not in ("rms_scaled", "mean_abs"):
            raise ValueError("radius_policy must be 'rms_scaled' or 'mean_abs'")
        good = (donors & (self.observed >= self.min_observations)
                & (self.mean_sq > 1e-12))
        # For fixed assignments and q, E[2*rho*|z| - rho²] is maximized
        # at rho = E|z|. This is exact for the stored projected moments,
        # though not guaranteed optimal after reassignment or drift.
        radius = (self.mean_abs.clamp_min(0.0) if radius_policy == "mean_abs"
                  else split_mult * self.mean_sq.clamp_min(0.0).sqrt())
        gain_per_point = (2.0 * radius * self.mean_abs - radius.square()).clamp_min(0.0)
        scores = usage.clamp_min(0) * gain_per_point
        good &= torch.isfinite(scores) & (gain_per_point > gain_floor)
        n = min(n_requested, int(good.sum().item()))
        if not n:
            return (
                torch.empty(0, dtype=torch.int64, device=usage.device),
                torch.empty((0, self.q.shape[1]), dtype=torch.float32, device=usage.device),
            )
        scores = torch.where(good, scores, float("-inf"))
        src = torch.topk(scores, n, sorted=True).indices
        delta = radius[src, None] * self.q[src]
        return src, delta

    @torch.no_grad()
    def on_split(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """Preserve learned directions but discard stale pre-split moments."""
        self.q[target] = self.q[source].clone()
        for state in (self.observed, self.mean_abs, self.mean_sq):
            state[source] = 0.0
            state[target] = 0.0
