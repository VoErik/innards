"""
Causal interventions through an SAE: splicing it into a layer, with its latents edited.

:func:`sae_edits` splices the SAE into a layer, as edits for
:func:`innards.edits.run`: the layer's output is replaced by the SAE's
reconstruction with the latents edited, optionally keeping the SAE error term
so that only the edit changes the model (Marks et al., 2024). Latent edits are
the unit edits of :mod:`innards.edits` (:func:`~innards.edits.zero`,
:func:`~innards.edits.scale`, :func:`~innards.edits.clamp`,
:func:`~innards.edits.mean_ablate`) applied to the ``(n, d_sae)`` latents. To
add a decoder direction without running the SAE, use :func:`innards.edits.steer`.

Typical usage::

    from innards.edits import clamp, rows, run, steer, zero

    logits, _ = run(model, images, sae_edits(sae, "blocks.6", zero([123]), exclude=(0,)))
    logits, _ = run(model, images, sae_edits(sae, "blocks.6", clamp(42, 5.0)))
    logits, _ = run(model, images, {"blocks.6": rows(steer(sae.decoder_weight()[42], 8.0), exclude=(0,))})

References:
    Marks, S. et al. (2024). Sparse Feature Circuits: Discovering and Editing
        Interpretable Causal Graphs in Language Models. arXiv:2403.19647.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Sequence
)

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


def splice_fn(sae: SAE, edit: Edit | None = None, keep_error: bool = True) -> Callable[[Tensor, Tensor], Tensor]:
    """
    The SAE splice as a function of kept tokens ``x`` (read) and ``y`` (replaced).

    Returns ``decode(edit(encode(x))) + skip(x)``, plus with ``keep_error``
    the SAE error ``y - (decode(encode(x)) + skip(x))``. With the error kept
    and no edit, ``y`` is unchanged; with an edit, it moves by exactly the
    edited latents' decoder contributions. Without the error term and without
    an edit, it is the SAE reconstruction. For an SAE ``x is y``.

    Args:
        sae: The SAE or transcoder (should be in eval mode).
        edit: Edit of the ``(n, d_sae)`` latents (input units), e.g.
            :func:`innards.edits.zero` or :func:`innards.edits.clamp`.
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
    edit: Edit | None = None,
    keep_error: bool = True,
    exclude: Sequence[int] = (),
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
        edit: Edit of the ``(n, d_sae)`` latents, e.g. :func:`innards.edits.zero`.
        keep_error: Add the SAE error term back.
        exclude: Positions left unchanged (e.g. CLS / BOS).
        mask: Optional boolean ``(batch, tokens)`` mask of tokens to change,
            e.g. non-padding tokens.
        target: For transcoders: the layer whose output is replaced.
        model: Needed only to resolve modules given as ``layer`` / ``target``.

    Returns:
        Layer name -> edit, for :func:`innards.edits.run`.
    """
    name = layer if isinstance(layer, str) else layer_name(model, layer)
    target = name if target is None else target if isinstance(target, str) else layer_name(model, target)
    return cross(name, target, splice_fn(sae, edit, keep_error), exclude, mask)
