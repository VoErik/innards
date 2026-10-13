"""
Optimizers for SAE training: Adam, Muon and Signum.

The defaults and the parameter split follow EleutherAI's ``sparsify`` trainer:
Adam with ``lr = 2e-4 / sqrt(d_sae / 2**14)``; Muon (``lr = 2e-3``, independent
of ``d_sae``) on weight matrices with Adam on the rest; Signum with
``lr = 5e-3 / sqrt(d_sae / 2**14)``, capped at 5e-3 for ``d_sae < 2**14``
(our change). ``sparsify`` wraps Signum in a schedule-free wrapper; here every
optimizer uses the trainer's schedule.

# Note (Erik): I have only ever used Adam for my projects, cannot advise on the others.

References:
    Bernstein, J., Wang, Y.-X., Azizzadenesheli, K. & Anandkumar, A. (2018).
        signSGD: Compressed Optimisation for Non-Convex Problems. ICML 2018,
        arXiv:1802.04434.
    Jordan, K. et al. (2024). Muon: An optimizer for hidden layers in neural
        networks. Blog post, kellerjordan.github.io/posts/muon.
    Gao, L. et al. (2024). Scaling and evaluating sparse autoencoders.
        arXiv:2406.04093. (1/sqrt(d_sae) learning-rate scaling, Adam settings.)
    EleutherAI (2024-2025). sparsify. github.com/EleutherAI/sparsify.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import (
    Tensor,
    nn
)
from torch.optim import Optimizer


class Signum(Optimizer):
    """SignSGD with momentum (Bernstein et al., 2018).

    Update: ``m<-beta*m+(1-beta)*g``; ``p<-p-lr*sign(m)``.

    Args:
        params: Parameters or parameter groups.
        lr: Learning rate.
        momentum: Momentum coefficient ``beta``.
    """

    def __init__(self, params: Iterable, lr: float = 1e-3, momentum: float = 0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs one optimization step.

        Args:
            closure: Optional closure re-evaluating the loss.

        Returns:
            The loss returned by ``closure``, if given.
        """
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - group["momentum"])
                p.add_(buf.sign(), alpha=-group["lr"])
        return loss


def _newton_schulz(g: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate orthogonalization of ``g`` (Jordan et al., 2024)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.T if transposed else x).to(g.dtype)


class MuonFallback(Optimizer):
    """Minimal Muon for PyTorch versions without ``torch.optim.Muon``.

    Nesterov momentum, Newton-Schulz orthogonalization of the update, and the
    ``sqrt(max(1, rows / cols))`` scale of the original implementation
    (Jordan et al., 2024). Only for 2-D parameters.

    Args:
        params: 2-D parameters or parameter groups.
        lr: Learning rate.
        momentum: Momentum coefficient.
        ns_steps: Newton-Schulz iterations.
    """

    def __init__(self, params: Iterable, lr: float = 2e-3, momentum: float = 0.95, ns_steps: int = 5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs one optimization step.

        Args:
            closure: Optional closure re-evaluating the loss.

        Returns:
            The loss returned by ``closure``, if given.
        """
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - group["momentum"])
                update = p.grad.lerp(buf, group["momentum"])  # Nesterov
                update = _newton_schulz(update, group["ns_steps"])
                scale = math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
                p.add_(update, alpha=-group["lr"] * scale)
        return loss


def default_lr(optimizer: str, d_sae: int) -> float:
    """Default learning rate per optimiszr, following ``sparsify``.

    Args:
        optimizer: ``"adam"``, ``"muon"`` or ``"signum"``.
        d_sae: Number of latents.

    Returns:
        The learning rate.

    Raises:
        ValueError: If the optimizer name is unknown.
    """
    scale = math.sqrt(d_sae / 2**14)
    if optimizer == "adam":
        return 2e-4 / scale
    if optimizer == "muon":
        return 2e-3
    if optimizer == "signum":
        # Capped at the d_sae = 2**14 value: sign updates move every
        # coordinate by the full learning rate, and extrapolating the formula to
        # small dictionaries kills most latents.
        return 5e-3 / max(scale, 1.0)
    raise ValueError(f"unknown optimizer {optimizer!r}")


def build_optimizers(
    model: nn.Module,
    optimizer: str,
    lr: float,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 6.25e-10,
) -> list[tuple[Optimizer, bool]]:
    """
    Builds the optimizer(s) for an SAE.

    Adam uses betas ``(0.9,0.999)``, ``eps=6.25e-10`` and no weight decay
    (Gao et al., 2024, App. A.3).

    Args:
        model: The SAE.
        optimizer: ``"adam"``, ``"muon"`` or ``"signum"``.
        lr: Learning rate.
        adam_betas: Adam betas.
        adam_eps: Adam epsilon.

    Returns:
        A list of ``(optimizer, use_warmup)`` pairs. Muon gets no warmup, as in
        ``sparsify``; everything else follows the trainer's warmup.

    Raises:
        ValueError: If the optimizer name is unknown.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer == "adam":
        return [(torch.optim.Adam(params, lr=lr, betas=adam_betas, eps=adam_eps), True)]
    if optimizer == "signum":
        return [(Signum(params, lr=lr), True)]
    if optimizer == "muon":
        matrices = [p for p in params if p.ndim >= 2]
        rest = [p for p in params if p.ndim < 2]
        if hasattr(torch.optim, "Muon"):
            muon = torch.optim.Muon(matrices, lr=lr, weight_decay=0.0)
        else:
            muon = MuonFallback(matrices, lr=lr)
        out: list[tuple[Optimizer, bool]] = [(muon, False)]
        if rest:
            out.append((torch.optim.Adam(rest, lr=lr, betas=adam_betas, eps=adam_eps), True))
        return out
    raise ValueError(f"unknown optimizer {optimizer!r}")
