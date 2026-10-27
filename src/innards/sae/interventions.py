"""
Causal interventions through an SAE: ablating, clamping and steering latents.

:func:`sae_edits` splices the SAE into a layer, as edits for
:func:`innards.edits.run`: the layer's output is replaced by the SAE's
reconstruction with the latents edited, optionally keeping the SAE error term
so that only the edit changes the model (Marks et al., 2024). Latent edits are
plain functions on the ``(n, d_sae)`` activations; the factories below cover
the usual ones. To add a direction without running the SAE (e.g. a decoder
atom), use :func:`innards.edits.steer`.

Typical usage::

    from innards.edits import run, rows, steer

    logits, _ = run(model, images, sae_edits(sae, "blocks.6", zero_ablate([123]), exclude_positions=(0,)))
    logits, _ = run(model, images, {"blocks.6": rows(steer(sae.decoder_weight()[42], 8.0), exclude=(0,))})

References:
    Marks, S. et al. (2024). Sparse Feature Circuits: Discovering and Editing
        Interpretable Causal Graphs in Language Models. arXiv:2403.19647.
    Templeton, A. et al. (2024). Scaling Monosemanticity: Extracting
        Interpretable Features from Claude 3 Sonnet. Transformer Circuits
        Thread. (Feature clamping / steering.)
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Sequence
)

import torch
from nnsight.intervention.envoy import Envoy
from torch import (
    Tensor,
    nn
)

from ..edits import (
    Edit,
    cross
)
from ..models import layer_name
from .model import SAE

LatentEdit = Callable[[Tensor], Tensor]


def _index(latents: int | Sequence[int] | Tensor, device: torch.device) -> Tensor:
    return torch.as_tensor([latents] if isinstance(latents, int) else latents, device=device, dtype=torch.long)


def zero_ablate(latents: int | Sequence[int] | Tensor) -> LatentEdit:
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


def mean_ablate(latents: int | Sequence[int] | Tensor, means: Tensor) -> LatentEdit:
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


def clamp(latents: int | Sequence[int] | Tensor, value: float, only_active: bool = False) -> LatentEdit:
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


def scale(latents: int | Sequence[int] | Tensor, factor: float) -> LatentEdit:
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


def splice_fn(sae: SAE, edit: LatentEdit | None = None, keep_error: bool = True) -> Callable[[Tensor, Tensor], Tensor]:
    """
    The SAE splice as a function of kept tokens ``x`` (read) and ``y`` (replaced).

    Returns ``decode(edit(encode(x))) + skip(x)``, plus with ``keep_error``
    the SAE error ``y - (decode(encode(x)) + skip(x))``. With the error kept
    and no edit, ``y`` is unchanged; with an edit, it moves by exactly the
    edited latents' decoder contributions. Without the error term and without
    an edit, it is the SAE reconstruction. For an SAE ``x is y``.

    Args:
        sae: The SAE or transcoder (should be in eval mode).
        edit: Function on the ``(n, d_sae)`` activations (input units), e.g.
            from :func:`zero_ablate` or :func:`clamp`.
        keep_error: Add the SAE error term back.

    Returns:
        ``fn(x, y) -> y'``, e.g. for :func:`innards.edits.cross`.
    """

    def fn(x: Tensor, y: Tensor) -> Tensor:
        x, y = x.to(sae.w_enc.dtype), y.to(sae.w_enc.dtype)
        acts = sae.encode(x)
        skip = sae.skip(x)
        recon = sae.decode(acts) + skip
        new = sae.decode(edit(acts)) + skip if edit is not None else recon
        return new + (y - recon) if keep_error else new

    return fn


def sae_edits(
    sae: SAE,
    layer: str | nn.Module | Envoy,
    edit: LatentEdit | None = None,
    keep_error: bool = True,
    exclude_positions: Sequence[int] = (),
    mask: Tensor | None = None,
    target: str | nn.Module | Envoy | None = None,
    model: nn.Module | Envoy | None = None,
) -> dict[str, Edit]:
    """
    Edits splicing the SAE into ``layer``, optionally editing its latents.

    See :func:`splice_fn` for what the kept tokens become. For a transcoder
    (``target`` set) the latents are read from ``layer`` and ``target``'s
    output is replaced.

    Args:
        sae: The SAE (should be in eval mode).
        layer: Layer whose output the SAE reads, as a name (or a module, with
            ``model`` given).
        edit: Function on the ``(n, d_sae)`` activations.
        keep_error: Add the SAE error term back.
        exclude_positions: Positions left unchanged (e.g. CLS / BOS).
        mask: Optional boolean ``(batch, tokens)`` mask of tokens to change,
            e.g. non-padding tokens.
        target: For transcoders: the layer whose output is replaced.
        model: Needed only to resolve modules given as ``layer`` / ``target``.

    Returns:
        Layer name -> edit, for :func:`innards.edits.run`.
    """
    name = layer if isinstance(layer, str) else layer_name(model, layer)
    target = name if target is None else target if isinstance(target, str) else layer_name(model, target)
    return cross(name, target, splice_fn(sae, edit, keep_error), exclude_positions, mask)
