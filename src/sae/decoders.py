"""
Decoder dictionaries of :class:`sae.model.SAE`.

``SAEConfig.decoder`` selects one of:

* ``"linear"``: free atoms projected to unit norm after every step, with the
  gradient component parallel to each atom removed (Gao et al., 2024).
* ``"archetypal"``: atoms constrained to (a relaxation of) the convex hull of
  data candidates, ``D = (W C + Lambda) * exp(s)`` (RA-SAE; Fel et al., 2025),
  which improves dictionary stability across seeds.

New parametrizations subclass :class:`Decoder` and are added to
:data:`DECODERS`.

References:
    Gao, L. et al. (2024). Scaling and evaluating sparse autoencoders.
        arXiv:2406.04093.
    Fel, T. et al. (2025). Archetypal SAE: Adaptive and Stable Dictionary
        Learning for Concept Extraction in Large Vision Models. ICML 2025,
        arXiv:2502.12892.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    ClassVar
)

import torch
import torch.nn.functional as F
from einops import einsum
from torch import (
    Tensor,
    nn
)

if TYPE_CHECKING:
    from .model import SAEConfig


@torch.no_grad()
def kmeans(points: Tensor, n_clusters: int, n_iter: int = 25, seed: int = 0, chunk: int = 8192) -> Tensor:
    """
    Lloyd's k-means initialized from random data points.

    Produces the candidate points of the Archetypal decoder.

    Args:
        points: Data, shape ``(n, d)`` with ``n >= n_clusters``.
        n_clusters: Number of centroids.
        n_iter: Number of Lloyd iterations.
        seed: Seed for the initial centroid draw.
        chunk: Rows per chunk when computing distances, to bound memory.

    Returns:
        Centroids, shape ``(n_clusters, d)``.

    Raises:
        ValueError: If there are fewer points than clusters.
    """
    points = points.float()
    n, _ = points.shape
    if n < n_clusters:
        raise ValueError(f"need at least {n_clusters} points for k-means, got {n}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    centroids = points[torch.randperm(n, generator=g)[:n_clusters].to(points.device)].clone()
    for _ in range(n_iter):
        assign = torch.cat([torch.cdist(c, centroids).argmin(-1) for c in points.split(chunk)])
        sums = torch.zeros_like(centroids).index_add_(0, assign, points)
        counts = torch.bincount(assign, minlength=n_clusters).unsqueeze(-1)
        empty = counts.squeeze(-1) == 0
        centroids = torch.where(empty.unsqueeze(-1), centroids, sums / counts.clamp_min(1))
        if empty.any():  # re-seed empty clusters with random points
            idx = torch.randint(0, n, (int(empty.sum()),), generator=g).to(points.device)
            centroids[empty] = points[idx]
    return centroids


class Decoder(nn.Module):
    """
    Base class of decoder dictionaries.

    Args:
        cfg: The SAE configuration.

    Class attributes:
        min_init_tokens: Minimum sample size :meth:`initialize_from_data_` needs.
    """

    min_init_tokens: ClassVar[int] = 0

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg

    def weight(self) -> Tensor:
        """The dictionary, shape ``(d_sae, d_out)``."""
        raise NotImplementedError

    def reset_parameters(self) -> Tensor:
        """
        Data-free initialization.

        Returns:
            Unit-norm directions ``(d_sae, d_out)`` to tie an SAE's encoder to.
        """
        raise NotImplementedError

    def initialize_from_data_(self, centred: Tensor) -> Tensor | None:
        """
        Data-dependent initialization.

        Args:
            centred: Scaled reconstruction targets minus the decoder bias, ``(n, d_out)``.

        Returns:
            New directions to tie the encoder to, or ``None`` to keep it.
        """
        return None

    def refresh_(self) -> None:
        """Recomputes cached state derived from the parameters (no-op by default)."""

    def pre_step(self) -> None:
        """Gradient surgery after ``backward()``, before the optimizer step."""

    def post_step(self) -> None:
        """Constraint projections after the optimizer step."""

    def fold_(self, sd: dict[str, Tensor], prefix: str, scale: Tensor) -> None:
        """
        Re-expresses the state for inputs in raw units, keeping ``weight()`` fixed.

        Args:
            sd: State dict to modify in place.
            prefix: Key prefix of this module in ``sd``.
            scale: The SAE's input scale.
        """


class LinearDecoder(Decoder):
    """Unit-norm dictionary ``w_dec`` (Gao et al., 2024)."""

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        self.w_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_out))

    def weight(self) -> Tensor:
        return self.w_dec

    @torch.no_grad()
    def reset_parameters(self) -> Tensor:
        nn.init.normal_(self.w_dec)
        self.post_step()
        return self.w_dec

    @torch.no_grad()
    def pre_step(self) -> None:
        """Removes the gradient component parallel to each atom (Gao et al., 2024, App. A.3)."""
        if self.w_dec.grad is None:
            return
        proj = einsum(self.w_dec.grad, self.w_dec, "d_sae d_in, d_sae d_in -> d_sae")
        self.w_dec.grad.sub_(einsum(proj, self.w_dec, "d_sae, d_sae d_in -> d_sae d_in"))

    @torch.no_grad()
    def post_step(self) -> None:
        """Projects atoms to unit norm."""
        self.w_dec.div_(self.w_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))


class ArchetypalDecoder(Decoder):
    """
    Relaxed archetypal dictionary ``D = (W C + Lambda) * exp(s)`` (Fel et al., 2025).

    ``C`` holds ``archetypal_n_candidates`` k-means centroids of the centred
    data, ``W`` is row-stochastic and ``||Lambda_i|| <= archetypal_delta``. In
    eval mode the fused dictionary cached in ``w_dec`` is used.
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        d_sae, d_out, n_cand = cfg.d_sae, cfg.d_out, cfg.archetypal_n_candidates
        self.arch_weight = nn.Parameter(torch.empty(d_sae, n_cand))
        self.arch_relax = nn.Parameter(torch.zeros(d_sae, d_out))
        if cfg.archetypal_learn_scale:
            self.arch_log_scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("arch_log_scale", torch.zeros(()))
        self.register_buffer("arch_points", torch.zeros(n_cand, d_out))
        self.register_buffer("arch_initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("w_dec", torch.zeros(d_sae, d_out))

    @property
    def min_init_tokens(self) -> int:  # type: ignore[override]
        return 4 * self.cfg.archetypal_n_candidates

    def _fused(self) -> Tensor:
        d = einsum(self.arch_weight, self.arch_points, "d_sae n_cand, n_cand d_in -> d_sae d_in")
        return (d + self.arch_relax) * self.arch_log_scale.exp()

    def weight(self) -> Tensor:
        if not self.training:
            return self.w_dec
        if not bool(self.arch_initialized):
            raise RuntimeError("Archetypal SAE needs initialize_from_data() before training")
        return self._fused()

    @torch.no_grad()
    def refresh_(self) -> None:
        """Recomputes the cached dictionary ``w_dec``."""
        self.w_dec.copy_(self._fused())

    def train(self, mode: bool = True) -> "ArchetypalDecoder":
        super().train(mode)
        if not mode:
            self.refresh_()
        return self

    @torch.no_grad()
    def reset_parameters(self) -> Tensor:
        """Atom ``i`` starts on candidate ``i mod n_candidates`` plus small noise."""
        d_sae, n_cand = self.arch_weight.shape
        one_hot = F.one_hot(torch.arange(d_sae) % n_cand, n_cand).float()
        self.arch_weight.copy_(one_hot + 0.01 * torch.rand(d_sae, n_cand))
        self.arch_relax.zero_()
        self.arch_log_scale.zero_()
        self.post_step()
        return F.normalize(torch.randn(d_sae, self.cfg.d_out), dim=-1)

    @torch.no_grad()
    def initialize_from_data_(self, centred: Tensor) -> Tensor:
        self.arch_points.copy_(kmeans(centred, self.cfg.archetypal_n_candidates))
        self.arch_initialized.fill_(True)
        self.refresh_()
        return F.normalize(self.w_dec, dim=-1)

    @torch.no_grad()
    def post_step(self) -> None:
        """Makes ``W`` row-stochastic (ReLU + row normalization) and projects ``Lambda`` onto its ball."""
        w = F.relu(self.arch_weight)
        self.arch_weight.copy_(w / (w.sum(dim=-1, keepdim=True) + 1e-8))
        norm = self.arch_relax.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.arch_relax.mul_(torch.clamp(self.cfg.archetypal_delta / norm, max=1.0))

    def fold_(self, sd: dict[str, Tensor], prefix: str, scale: Tensor) -> None:
        # C lives in the scaled space; re-express it (and Lambda) in input
        # units and compensate with the learned scale, so D is unchanged.
        sd[prefix + "arch_points"] = sd[prefix + "arch_points"] / scale
        sd[prefix + "arch_relax"] = sd[prefix + "arch_relax"] / scale
        sd[prefix + "arch_log_scale"] = sd[prefix + "arch_log_scale"] + torch.log(scale)


DECODERS: dict[str, type[Decoder]] = {
    "linear": LinearDecoder,
    "archetypal": ArchetypalDecoder,
}
