"""
Activation functions of :class:`innards.sae.model.SAE`: how pre-activations become
sparse latents.

Most SAE architectures differ only here, in the rule that selects active
latents and in the sparsity penalty that comes with it. Each rule is a small
:class:`Activation` module registered under a name, which
``SAEConfig.activation`` selects:

* ``"topk"``: the ``k`` largest per token (Makhzani & Frey, 2013; Gao et al.,
  2024), optionally by absolute value (AbsTopK; Zhu et al., 2025).
* ``"batchtopk"``: the ``k * batch_size`` largest over the batch, with a
  learned global threshold at inference (Bussmann, Leask & Nanda, 2024).
* ``"jumprelu"``: per-latent learned thresholds trained with
  straight-through estimators and an L0 penalty (Rajamanoharan et al., 2024).
* ``"relu"``: ReLU with an L1 penalty weighted by decoder norms (Bricken et
  al., 2023; Conerly et al., 2024).

A new architecture is a subclass of :class:`Activation` decorated with
:func:`register_activation`; the model, trainer, saving and analysis code
need no changes.

References:
    Makhzani, A. & Frey, B. (2013). k-Sparse Autoencoders. arXiv:1312.5663.
    Bricken, T. et al. (2023). Towards Monosemanticity: Decomposing Language
        Models With Dictionary Learning. Transformer Circuits Thread.
    Conerly, T., Templeton, A., Bricken, T., Marcus, J. & Henighan, T. (2024).
        Update on how we train SAEs. Circuits Updates, April 2024.
    Gao, L. et al. (2024). Scaling and evaluating sparse autoencoders.
        arXiv:2406.04093.
    Rajamanoharan, S. et al. (2024). Jumping Ahead: Improving Reconstruction
        Fidelity with JumpReLU Sparse Autoencoders. arXiv:2407.14435.
    Bussmann, B., Leask, P. & Nanda, N. (2024). BatchTopK Sparse Autoencoders.
        arXiv:2412.06410.
    Zhu, X., Khalili, M. M. & Zhu, Z. (2025). AbsTopK: Rethinking Sparse
        Autoencoders For Bidirectional Features. ICLR 2026, arXiv:2510.00404.
"""

from __future__ import annotations

import math
from typing import (
    TYPE_CHECKING,
    ClassVar,
    NamedTuple
)

import torch
import torch.nn.functional as F
from einops import (
    einsum,
    rearrange,
    repeat
)
from torch import (
    Tensor,
    nn
)

if TYPE_CHECKING:
    from .model import SAEConfig


class SparseCode(NamedTuple):
    """
    Activations of ``n`` tokens, either sparse or dense.

    Sparse codes (TopK variants) are row-sorted COO matrices and decode by
    touching only active dictionary rows. Dense codes (JumpReLU, ReLU, built
    with :func:`dense_code`) set only ``dense`` and ``n``: their gradients need
    a dense decode anyway, and early in training they are far from sparse.

    Attributes:
        rows: Row (token) index of each non-zero, shape ``(nnz,)``.
        cols: Latent index of each non-zero, shape ``(nnz,)``.
        vals: Activation values, shape ``(nnz,)``.
        offsets: Start of each row's segment in ``cols``/``vals``, shape ``(n,)``.
        n: Number of rows.
        dense: The ``(n, d_sae)`` activations of a dense code, else ``None``.
    """

    rows: Tensor | None
    cols: Tensor | None
    vals: Tensor | None
    offsets: Tensor | None
    n: int
    dense: Tensor | None = None


def offsets_from_rows(rows: Tensor, n: int) -> Tensor:
    """Segment starts of a row-sorted COO matrix with ``n`` rows."""
    counts = torch.bincount(rows, minlength=n)
    return torch.cumsum(counts, 0) - counts


def dense_code(dense: Tensor) -> SparseCode:
    """
    Wraps dense activations ``(n, d_sae)`` as a :class:`SparseCode`.

    Args:
        dense: Activations, mostly zero.

    Returns:
        A code with only ``dense`` and ``n`` set.
    """
    return SparseCode(None, None, None, None, dense.shape[0], dense)


def rowwise_topk(pre: Tensor, score: Tensor, k: int, signed: bool, mask: Tensor | None = None) -> SparseCode:
    """
    Selects the ``k`` highest-scoring latents of each row.

    Args:
        pre: Pre-activations, shape ``(n, d_sae)``.
        score: Selection score, same shape (e.g. ``relu(pre)`` or ``|pre|``).
        k: Latents per row.
        signed: Keep the signed pre-activation as the value (AbsTopK)
            instead of the score.
        mask: Optional boolean ``(d_sae,)`` mask of selectable latents.

    Returns:
        A :class:`SparseCode` with exactly ``k`` entries per row; entries
        that are masked or have a zero score carry the value 0.
    """
    n = pre.shape[0]
    if mask is not None:
        score = score.masked_fill(~mask, float("-inf"))
    top_score, cols = score.topk(k, dim=-1, sorted=False)
    vals = pre.gather(-1, cols) if signed else top_score
    vals = torch.where(torch.isfinite(top_score), vals, torch.zeros_like(vals))
    return SparseCode(
        rows=repeat(torch.arange(n, device=pre.device), "n -> (n k)", k=k),
        cols=rearrange(cols, "n k -> (n k)"),
        vals=rearrange(vals, "n k -> (n k)"),
        offsets=torch.arange(0, n * k, k, device=pre.device),
        n=n,
    )


def decode_code(code: SparseCode, weight: Tensor, latent_mask: Tensor | None = None) -> Tensor:
    """
    Decodes a code with a dictionary, without bias.

    Sparse codes touch only active rows of the dictionary (Gao et al., 2024,
    App. D.2 motivate sparse kernels; ``embedding_bag`` is our portable
    substitute). Dense codes use a matmul so that latents just below a
    JumpReLU threshold still receive gradients.

    Args:
        code: The code of ``n`` tokens.
        weight: The dictionary, ``(d_sae, d_out)``.
        latent_mask: Optional ``(d_sae,)`` multiplier, e.g. a Matryoshka group.

    Returns:
        Shape ``(n, d_out)``.
    """
    if code.dense is not None:
        acts = code.dense if latent_mask is None else code.dense * latent_mask
        return einsum(acts.to(weight.dtype), weight, "n d_sae, d_sae d_out -> n d_out")
    vals = code.vals if latent_mask is None else code.vals * latent_mask[code.cols]
    return F.embedding_bag(code.cols, weight, code.offsets, mode="sum", per_sample_weights=vals.to(weight.dtype))


ACTIVATIONS: dict[str, type[Activation]] = {}


def register_activation(name: str):
    """
    Class decorator registering an :class:`Activation` under ``name``.

    Args:
        name: Value of ``SAEConfig.activation`` that selects the class.

    Returns:
        The decorator.
    """

    def wrap(cls: type[Activation]) -> type[Activation]:
        ACTIVATIONS[name] = cls
        return cls

    return wrap


class Activation(nn.Module):
    """
    Base class of activation functions.

    All inputs are in the SAE's scaled units (see :class:`innards.sae.model.SAE`).
    Subclasses implement :meth:`forward` and, where needed, :meth:`penalty`,
    :meth:`update_stats` and :meth:`fold_`. Parameters and buffers they
    register are trained, saved and loaded with the SAE.

    Args:
        cfg: The SAE configuration.

    Class attributes:
        uses_k: Whether the activation selects a fixed number ``k`` of
            latents (enables k annealing and Multi-TopK).
        default_aux_coeff: AuxK weight used when ``SAEConfig.aux_coeff`` is
            ``None``.
    """

    uses_k: ClassVar[bool] = False
    default_aux_coeff: ClassVar[float] = 0.0

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg

    @classmethod
    def validate(cls, cfg: SAEConfig) -> None:
        """
        Checks the configuration fields this activation relies on.

        Raises:
            ValueError: If the configuration is invalid.
        """

    def score(self, pre: Tensor) -> Tensor:
        """Ranking score of each latent, used by the AuxK loss to pick dead latents."""
        return F.relu(pre)

    def forward(self, pre: Tensor, k: int) -> SparseCode:
        """
        Selects active latents.

        Args:
            pre: Pre-activations, shape ``(n, d_sae)``.
            k: Current k (ignored unless ``uses_k``).

        Returns:
            The sparse code.
        """
        raise NotImplementedError

    def penalty(self, pre: Tensor, code: SparseCode, decoder_weight: Tensor) -> Tensor | None:
        """
        Sparsity penalty added to the training loss (before the warmup scale).

        Args:
            pre: Pre-activations, shape ``(n, d_sae)``.
            code: Output of :meth:`forward`.
            decoder_weight: Decoder dictionary, shape ``(d_sae, d_in)``.

        Returns:
            A scalar, or ``None`` for activations whose sparsity is structural.
        """
        return None

    @torch.no_grad()
    def update_stats(self, code: SparseCode) -> None:
        """Updates running statistics from a training batch."""

    def fold_(self, sd: dict[str, Tensor], prefix: str, scale: Tensor) -> None:
        """
        Re-expresses this module's state for inputs in raw (unscaled) units.

        Pre-activations become ``pre / scale``, so thresholds in scaled units
        are divided by ``scale``.

        Args:
            sd: State dict to modify in place.
            prefix: Key prefix of this module in ``sd``.
            scale: The SAE's input scale.
        """


@register_activation("topk")
class TopK(Activation):
    """The ``k`` latents with the largest (absolute, with ``abs_topk``) pre-activation per token."""

    uses_k = True
    default_aux_coeff = 1.0 / 32.0

    @classmethod
    def validate(cls, cfg: SAEConfig) -> None:
        if cfg.k is None or not (0 < cfg.k <= cfg.d_sae):
            raise ValueError(f"{cfg.activation} needs k in (0, d_sae], got k={cfg.k}")

    def score(self, pre: Tensor) -> Tensor:
        return pre.abs() if self.cfg.abs_topk else F.relu(pre)

    def forward(self, pre: Tensor, k: int) -> SparseCode:
        return rowwise_topk(pre, self.score(pre), k, signed=self.cfg.abs_topk)


@register_activation("batchtopk")
class BatchTopK(TopK):
    """
    The ``k * n`` largest activations of a batch of ``n`` tokens.

    At inference (eval mode, ``k == cfg.k``) latents above a global threshold
    fire instead, so a token's code does not depend on the rest of the batch.
    The threshold is an EMA of the smallest selected magnitude per training
    batch (``threshold_ema_decay``); Bussmann et al. (2024) use a plain
    average, the EMA follows activations that drift during training.
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        self.register_buffer("threshold", torch.full((), float("nan")))

    def forward(self, pre: Tensor, k: int) -> SparseCode:
        if not self.training and k == self.cfg.k and bool(torch.isfinite(self.threshold)):
            return self._threshold_select(pre)
        n, d_sae = pre.shape
        flat_pre = rearrange(pre, "n d_sae -> (n d_sae)")
        score = self.score(flat_pre)
        _, idx = score.topk(min(k * n, score.numel()), sorted=False)
        idx, _ = idx.sort()
        vals = flat_pre[idx] if self.cfg.abs_topk else score[idx]
        rows, cols = idx // d_sae, idx % d_sae
        return SparseCode(rows, cols, vals, offsets_from_rows(rows, n), n)

    def _threshold_select(self, pre: Tensor) -> SparseCode:
        thr = self.threshold.clamp_min(0.0).to(pre.dtype)
        score = pre.abs() if self.cfg.abs_topk else pre
        rows, cols = torch.nonzero(score > thr, as_tuple=True)  # row-major order
        return SparseCode(rows, cols, pre[rows, cols], offsets_from_rows(rows, pre.shape[0]), pre.shape[0])

    @torch.no_grad()
    def update_stats(self, code: SparseCode) -> None:
        mags = code.vals[code.vals != 0].abs()
        if mags.numel() == 0:
            return
        batch_min = mags.min().float()
        decay = self.cfg.threshold_ema_decay
        new = torch.where(torch.isfinite(self.threshold), decay * self.threshold + (1 - decay) * batch_min, batch_min)
        self.threshold.copy_(new)

    def fold_(self, sd: dict[str, Tensor], prefix: str, scale: Tensor) -> None:
        sd[prefix + "threshold"] = sd[prefix + "threshold"] / scale


def _rectangle(x: Tensor) -> Tensor:
    return (x.abs() < 0.5).to(x.dtype)


class _JumpReLUFn(torch.autograd.Function):
    """``pre * H(pre - theta)`` with the threshold pseudo-gradient of Rajamanoharan et al. (2024)."""

    @staticmethod
    def forward(ctx, pre: Tensor, theta: Tensor, bandwidth: float) -> Tensor:
        ctx.save_for_backward(pre, theta)
        ctx.bandwidth = bandwidth
        return pre * (pre > theta).to(pre.dtype)

    @staticmethod
    def backward(ctx, grad: Tensor):
        pre, theta = ctx.saved_tensors
        eps = ctx.bandwidth
        grad_pre = grad * (pre > theta).to(grad.dtype)
        kernel = _rectangle((pre.float() - theta) / eps)
        grad_theta = (-(theta / eps) * kernel * grad.float()).sum(0)
        return grad_pre, grad_theta.to(theta.dtype), None


class _StepFn(torch.autograd.Function):
    """Heaviside ``H(pre - theta)``; gradient only to the threshold."""

    @staticmethod
    def forward(ctx, pre: Tensor, theta: Tensor, bandwidth: float) -> Tensor:
        ctx.save_for_backward(pre, theta)
        ctx.bandwidth = bandwidth
        return (pre > theta).to(pre.dtype)

    @staticmethod
    def backward(ctx, grad: Tensor):
        pre, theta = ctx.saved_tensors
        eps = ctx.bandwidth
        kernel = _rectangle((pre.float() - theta) / eps)
        grad_theta = (-(1.0 / eps) * kernel * grad.float()).sum(0)
        return torch.zeros_like(pre), grad_theta.to(theta.dtype), None


@register_activation("jumprelu")
class JumpReLU(Activation):
    """
    JumpReLU with learned per-latent thresholds (Rajamanoharan et al., 2024).

    ``f(z) = z * H(z - theta)``, ``theta = exp(log_threshold)``. Thresholds
    get straight-through pseudo-gradients with a rectangle kernel of width
    ``jumprelu_bandwidth``, both through the reconstruction and through the
    L0 penalty ``sparsity_coeff * E[||f||_0]``. The penalty is relative to the
    normalised MSE, so tune ``sparsity_coeff`` for the L0 you want.
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        self.log_threshold = nn.Parameter(torch.full((cfg.d_sae,), math.log(cfg.jumprelu_init_threshold)))

    def forward(self, pre: Tensor, k: int) -> SparseCode:
        theta = self.log_threshold.exp()
        return dense_code(_JumpReLUFn.apply(pre, theta, self.cfg.jumprelu_bandwidth))

    def penalty(self, pre: Tensor, code: SparseCode, decoder_weight: Tensor) -> Tensor:
        l0 = _StepFn.apply(pre, self.log_threshold.exp(), self.cfg.jumprelu_bandwidth)
        return self.cfg.sparsity_coeff * l0.float().sum(-1).mean()

    def fold_(self, sd: dict[str, Tensor], prefix: str, scale: Tensor) -> None:
        sd[prefix + "log_threshold"] = sd[prefix + "log_threshold"] - torch.log(scale)


@register_activation("relu")
class ReLU(Activation):
    """
    ReLU with an L1 penalty ``sparsity_coeff * E[sum_i f_i ||d_i||]``.

    Weighting by decoder norms (Conerly et al., 2024) keeps the penalty
    meaningful for decoders whose atoms are not unit norm.
    """

    def forward(self, pre: Tensor, k: int) -> SparseCode:
        return dense_code(F.relu(pre))

    def penalty(self, pre: Tensor, code: SparseCode, decoder_weight: Tensor) -> Tensor:
        norms = decoder_weight.norm(dim=-1).to(code.dense.dtype)
        return self.cfg.sparsity_coeff * (code.dense * norms).float().sum(-1).mean()
