"""
Causal interventions through an SAE: ablating, clamping and steering latents.

:func:`splice_sae` replaces a module's output by the SAE's reconstruction
with the latents edited, optionally keeping the SAE error term so that only
the edit changes the model (Marks et al., 2024). Edits are plain functions
on the ``(n, d_sae)`` activations; the factories below cover the usual ones.
:func:`steer` adds a direction (e.g. a decoder atom) to the activations
without running the SAE, which also works during generation.

Typical usage::

    with splice_sae(sae, model.blocks[6], zero_ablate([123]), exclude_positions=(0,)):
        ablated = model(images)
    with steer(model.gpt_neox.layers[3], sae.decoder_weight()[42], alpha=8.0):
        model.generate(**inputs)

References:
    Marks, S. et al. (2024). Sparse Feature Circuits: Discovering and Editing
        Interpretable Causal Graphs in Language Models. arXiv:2403.19647.
    Templeton, A. et al. (2024). Scaling Monosemanticity: Extracting
        Interpretable Features from Claude 3 Sonnet. Transformer Circuits
        Thread. (Feature clamping / steering.)
    Turner, A. et al. (2023). Steering Language Models With Activation
        Engineering. arXiv:2308.10248.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterator,
    Sequence
)
from contextlib import contextmanager
from typing import Any

import torch
from torch import (
    Tensor,
    nn
)

from .data import splice
from .model import SAE

Edit = Callable[[Tensor], Tensor]


def _index(latents: int | Sequence[int] | Tensor, device: torch.device) -> Tensor:
    return torch.as_tensor([latents] if isinstance(latents, int) else latents, device=device, dtype=torch.long)


def zero_ablate(latents: int | Sequence[int] | Tensor) -> Edit:
    """
    Edit setting ``latents`` to 0.

    Args:
        latents: Latent index or indices.

    Returns:
        The edit.
    """

    def edit(acts: Tensor) -> Tensor:
        out = acts.clone()
        out[:, _index(latents, acts.device)] = 0
        return out

    return edit


def mean_ablate(latents: int | Sequence[int] | Tensor, means: Tensor) -> Edit:
    """
    Edit setting ``latents`` to their mean activation.

    Args:
        latents: Latent index or indices.
        means: Mean activation of every latent, shape ``(d_sae,)``, e.g.
            ``LatentStats.frequency * LatentStats.mean_activation`` (the mean
            over all tokens, including those where the latent is off).

    Returns:
        The edit.
    """

    def edit(acts: Tensor) -> Tensor:
        idx = _index(latents, acts.device)
        out = acts.clone()
        out[:, idx] = means.to(acts)[idx]
        return out

    return edit


def clamp(latents: int | Sequence[int] | Tensor, value: float, only_active: bool = False) -> Edit:
    """
    Edit clamping ``latents`` to ``value`` (Templeton et al., 2024).

    Args:
        latents: Latent index or indices.
        value: Activation value, in input units; e.g. a multiple of the
            latent's maximum activation to steer, or 0 to ablate.
        only_active: Clamp only where the latent already fires.

    Returns:
        The edit.
    """

    def edit(acts: Tensor) -> Tensor:
        idx = _index(latents, acts.device)
        out = acts.clone()
        target = torch.full_like(out[:, idx], value)
        out[:, idx] = torch.where(out[:, idx] != 0, target, out[:, idx]) if only_active else target
        return out

    return edit


def scale(latents: int | Sequence[int] | Tensor, factor: float) -> Edit:
    """
    Edit multiplying ``latents`` by ``factor``.

    Args:
        latents: Latent index or indices.
        factor: Multiplier; 0 ablates, values above 1 amplify.

    Returns:
        The edit.
    """

    def edit(acts: Tensor) -> Tensor:
        idx = _index(latents, acts.device)
        out = acts.clone()
        out[:, idx] = out[:, idx] * factor
        return out

    return edit


@contextmanager
def splice_sae(
    sae: SAE,
    hook_module: nn.Module,
    edit: Edit | None = None,
    keep_error: bool = True,
    layout: str = "btd",
    exclude_positions: Sequence[int] = (),
    token_mask_fn: Callable[[Any], Tensor] | None = None,
    batch: Any = None,
    target_module: nn.Module | None = None,
) -> Iterator[None]:
    """
    Splices the SAE into ``hook_module``, optionally editing its latents.

    At every kept token the output ``x`` becomes ``decode(edit(encode(x)))``
    plus, with ``keep_error``, the SAE error ``x - decode(encode(x))``. With
    the error kept and no edit the model is unchanged; with an edit, the
    output moves by exactly the edited latents' decoder contributions.
    Without the error term and without an edit, the model runs on the SAE
    reconstruction. For a transcoder (``target_module`` set) the latents are
    read from ``hook_module`` and ``target_module``'s output ``y`` is replaced
    by ``decode(edit(encode(x))) + skip(x)`` plus the error ``y - (decode(encode(x)) + skip(x))``.

    Args:
        sae: The SAE (should be in eval mode).
        hook_module: Module whose output the SAE reads.
        edit: Function on the ``(n, d_sae)`` activations (input units), e.g.
            from :func:`zero_ablate` or :func:`clamp`.
        keep_error: Add the SAE error term back.
        layout: Activation layout at the hook.
        exclude_positions: Positions left unchanged (e.g. CLS / BOS).
        token_mask_fn: Optional ``fn(batch) -> bool (batch, tokens)`` mask of
            tokens to change, e.g. non-padding tokens.
        batch: The model input batch passed to ``token_mask_fn``.
        target_module: For transcoders: the module whose output is replaced.

    Yields:
        None.
    """

    def fn(x: Tensor, y: Tensor) -> Tensor:
        x, y = x.to(sae.w_enc.dtype), y.to(sae.w_enc.dtype)
        acts = sae.encode(x)
        skip = sae.skip(x)
        recon = sae.decode(acts) + skip
        new = sae.decode(edit(acts)) + skip if edit is not None else recon
        return new + (y - recon) if keep_error else new

    with splice(hook_module, fn, target_module, layout, exclude_positions, token_mask_fn, batch):
        yield


@contextmanager
def steer(
    hook_module: nn.Module,
    direction: Tensor,
    alpha: float = 1.0,
    layout: str = "btd",
    exclude_positions: Sequence[int] = (),
    token_mask_fn: Callable[[Any], Tensor] | None = None,
    batch: Any = None,
) -> Iterator[None]:
    """
    Adds ``alpha * direction`` to the output of ``hook_module`` (Turner et al., 2023).

    With a unit-norm decoder atom as ``direction``, ``alpha`` is the latent
    activation being added. No SAE runs, so this also works token by token
    during generation.

    Args:
        hook_module: Module to steer.
        direction: Vector of shape ``(d,)``, e.g. ``sae.decoder_weight()[latent]``.
        alpha: Scale.
        layout: Activation layout at the hook.
        exclude_positions: Positions left unchanged. Positions count within
            each forward call: with KV-cached generation every new token is
            position 0 of its call, so leave this empty there.
        token_mask_fn: Optional ``fn(batch) -> bool (batch, tokens)`` mask of
            tokens to steer.
        batch: The model input batch passed to ``token_mask_fn``.

    Yields:
        None.
    """
    direction = direction.detach()

    with splice(hook_module, lambda x, y: y + alpha * direction.to(y), None, layout, exclude_positions,
                token_mask_fn, batch):
        yield
