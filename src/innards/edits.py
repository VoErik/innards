"""
Interventions on activations, applied with nnsight.

An edit is a plain function on a layer's tokens ``(batch, tokens, d)`` (see
:mod:`innards.layouts`) that returns tokens of the same shape. Edits compose
(:func:`compose`), can be restricted to some tokens (:func:`rows`), and work
the same for ViTs and CNNs; :func:`run` applies a dict of them in one nnsight
trace, :func:`edit_output` applies one inside your own trace.

Factories for the usual edits: :func:`steer`, :func:`ablate` (project out
directions), :func:`patch` (activation patching), and edits of single units
(neurons, or SAE latents): :func:`zero`, :func:`scale`, :func:`clamp`,
:func:`mean_ablate`. :func:`innards.sae.interventions.sae_edits` applies the
unit edits in an SAE's latent space.

Typical usage::

    logits, _ = run(model, images, {"blocks.6": ablate(v)})
    logits, reads = run(model, images, {"blocks.6": steer(v, 4.0)}, read=["blocks.9"])
    _, clean = run(model, images, read=["layer3"])
    patched, _ = run(model, corrupted, {"layer3": patch(clean["layer3"], exclude=(0,))})

    with model.trace(images):                       # inside your own trace
        edit_output(model.blocks[6], ablate(v))
        logits = model.output.save()

References:
    Templeton, A. et al. (2024). Scaling Monosemanticity: Extracting
        Interpretable Features from Claude 3 Sonnet. Transformer Circuits
        Thread. (Feature clamping.)
    Turner, A. et al. (2023). Steering Language Models With Activation
        Engineering. arXiv:2308.10248.
    Meng, K. et al. (2022). Locating and Editing Factual Associations in GPT.
        NeurIPS 2022. (Activation patching.)
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Sequence
)
from typing import Any

import torch
from nnsight.intervention.envoy import Envoy
from torch import (
    Tensor,
    nn
)

from .extract import (
    Layers,
    _as_list,
    _first,
    _reduce,
    default_input_fn,
    plan
)
from .layouts import (
    from_tokens,
    keep_mask,
    to_tokens
)
from .models import wrap

Edit = Callable[[Tensor], Tensor]


def rows(fn: Callable[[Tensor], Tensor], exclude: Sequence[int] = (), mask: Tensor | None = None) -> Edit:
    """
    Applies ``fn`` to the kept tokens only, as rows ``(n, d)``.

    Excluded positions and tokens outside ``mask`` pass through unchanged.

    Args:
        fn: Maps kept tokens ``(n, d)`` to replacements ``(n, d)``, e.g. an SAE
            reconstruction or any edit (edits on tokens also work on rows).
        exclude: Positions left unchanged, e.g. ``(0,)`` for CLS / BOS.
        mask: Optional boolean ``(batch, tokens)`` mask of tokens to change.

    Returns:
        The edit.
    """

    def edit(tokens: Tensor) -> Tensor:
        keep = keep_mask(tokens, exclude, mask)
        out = tokens.clone()
        out[keep] = fn(tokens[keep]).to(tokens.dtype)
        return out

    return edit


def cross(
    source: str,
    target: str,
    fn: Callable[[Tensor, Tensor], Tensor],
    exclude: Sequence[int] = (),
    mask: Tensor | None = None,
) -> dict[str, Edit]:
    """
    Edits replacing ``target``'s kept tokens ``y`` by ``fn(x, y)``, with ``x`` read at ``source``.

    For a transcoder ``source`` is its input (e.g. the norm before an MLP) and
    ``target`` the output it predicts (the MLP). ``source`` must run before
    ``target`` and have the same token grid. With ``source == target`` it is
    :func:`rows` with ``fn(y, y)``.

    Args:
        source: Layer whose tokens are read.
        target: Layer whose tokens are replaced.
        fn: Maps kept tokens ``x, y`` (both ``(n, d)``) to replacements for ``y``.
        exclude: Positions left unchanged.
        mask: Optional boolean ``(batch, tokens)`` mask of tokens to change.

    Returns:
        Layer name -> edit, for :func:`run`.
    """
    if source == target:
        return {target: rows(lambda y: fn(y, y), exclude, mask)}
    store: dict[str, Tensor] = {}

    def read(tokens: Tensor) -> Tensor:
        store["x"] = tokens
        return tokens

    def write(tokens: Tensor) -> Tensor:
        if "x" not in store:
            raise RuntimeError(f"{target!r} ran before {source!r}")
        keep = keep_mask(tokens, exclude, mask)
        out = tokens.clone()
        out[keep] = fn(store.pop("x")[keep], tokens[keep]).to(tokens.dtype)
        return out

    return {source: read, target: write}


def compose(*edits: Edit) -> Edit:
    """
    Applies edits one after the other.

    Args:
        *edits: Edits, applied left to right.

    Returns:
        The combined edit.
    """

    def edit(tokens: Tensor) -> Tensor:
        for e in edits:
            tokens = e(tokens)
        return tokens

    return edit


def steer(direction: Tensor, alpha: float = 1.0) -> Edit:
    """
    Adds ``alpha * direction`` to every token (Turner et al., 2023).

    Args:
        direction: Vector ``(d,)``, e.g. a probe direction or an SAE decoder atom.
        alpha: Scale.

    Returns:
        The edit.
    """
    direction = direction.detach()
    return lambda tokens: tokens + alpha * direction.to(tokens)


def ablate(directions: Tensor) -> Edit:
    """
    Projects out the span of ``directions``.

    Args:
        directions: One direction ``(d,)`` or several ``(k, d)``; they need
            not be orthonormal.

    Returns:
        The edit.
    """
    basis = torch.linalg.qr(directions.detach().reshape(-1, directions.shape[-1]).T.float()).Q  # (d, k)

    def edit(tokens: Tensor) -> Tensor:
        q = basis.to(tokens)
        return tokens - (tokens @ q) @ q.T

    return edit


Units = int | Sequence[int] | Tensor | None


def _units(units: Units, device: torch.device) -> Tensor | slice:
    if units is None:
        return slice(None)
    return torch.as_tensor([units] if isinstance(units, int) else units, device=device, dtype=torch.long)


def _set_units(fn: Callable[[Tensor, Tensor | slice], Tensor], units: Units) -> Edit:
    """An edit replacing the selected units (last dimension) by ``fn(values, index)``."""

    def edit(x: Tensor) -> Tensor:
        idx = _units(units, x.device)
        out = x.clone()
        out[..., idx] = fn(x[..., idx], idx).to(x.dtype)
        return out

    return edit


def zero(units: Units = None) -> Edit:
    """
    Sets units (or everything) to zero.

    Units index the last dimension: neurons of a layer's tokens, or the
    latents of an SAE (see :func:`innards.sae.interventions.sae_edits`).

    Args:
        units: Unit index or indices; ``None`` zeroes everything.

    Returns:
        The edit.
    """
    return _set_units(lambda v, idx: torch.zeros_like(v), units)


def scale(units: Units, factor: float) -> Edit:
    """
    Multiplies units by ``factor`` (0 ablates, above 1 amplifies).

    Args:
        units: Unit index or indices; ``None`` scales everything.
        factor: Multiplier.

    Returns:
        The edit.
    """
    return _set_units(lambda v, idx: v * factor, units)


def clamp(units: Units, value: float, only_active: bool = False) -> Edit:
    """
    Sets units to ``value``, e.g. to steer with an SAE latent (Templeton et al., 2024).

    Args:
        units: Unit index or indices.
        value: The value, e.g. a multiple of the unit's maximum activation.
        only_active: Only where the unit is non-zero (SAE latents that fire).

    Returns:
        The edit.
    """
    return _set_units(lambda v, idx: torch.where(v != 0, value, v) if only_active else torch.full_like(v, value), units)


def mean_ablate(mean: Tensor, units: Units = None) -> Edit:
    """
    Replaces units (or the whole vector) by their mean, e.g. the dataset mean of the layer.

    Args:
        mean: Mean of every unit ``(d,)``. For SAE latents, the mean over all
            tokens including those where a latent is off, e.g.
            ``frequency * mean_activation``.
        units: Unit index or indices; ``None`` replaces the whole vector.

    Returns:
        The edit.
    """
    mean = mean.detach()
    return _set_units(lambda v, idx: mean.to(v.device, v.dtype)[idx].expand_as(v), units)


def patch(source: Tensor, exclude: Sequence[int] = (), mask: Tensor | None = None) -> Edit:
    """
    Replaces the kept tokens by those of ``source`` (activation patching).

    Args:
        source: Tokens of the same shape as the edited ones, e.g. from
            ``run(model, other_inputs, read=[layer])``.
        exclude: Positions left unchanged.
        mask: Optional boolean ``(batch, tokens)`` mask of tokens to patch,
            e.g. a single position.

    Returns:
        The edit.
    """
    source = source.detach()

    def edit(tokens: Tensor) -> Tensor:
        src = source.to(tokens)
        src = src.unsqueeze(1) if src.ndim == tokens.ndim - 1 else src  # "bd" reads are (batch, d)
        keep = keep_mask(tokens, exclude, mask)
        out = tokens.clone()
        out[keep] = src[keep]
        return out

    return edit


def edit_output(envoy: Envoy, edit: Edit, layout: str = "btd") -> Tensor:
    """
    Replaces a layer's output by its edited version; call inside an nnsight trace.

    Tuple outputs have their first element edited.

    Args:
        envoy: The layer, e.g. ``model.blocks[6]``.
        edit: Edit on the layer's tokens.
        layout: The layer's activation layout.

    Returns:
        The new output (first element for tuples).
    """
    out = envoy.output
    act = _first(out)
    new = from_tokens(edit(to_tokens(act, layout)), layout, act)
    if isinstance(out, (tuple, list)):
        envoy.output = type(out)((new, *out[1:]))
    else:
        envoy.output = new
    return new


def run(
    model: nn.Module | Envoy,
    batch: Any,
    edits: dict[str, Edit] | None = None,
    read: Layers = (),
    layouts: str | dict[str, str] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
) -> tuple[Any, dict[str, Tensor]]:
    """
    Runs the model with edits applied, in one nnsight trace.

    Gradients flow through the edits unless the call is wrapped in
    ``torch.no_grad``, so the output can be backpropagated (e.g. to train
    the module inside an edit, or for attribution).

    Args:
        model: An nnsight model or a plain module.
        batch: An input batch (on the model's device).
        edits: Layer name -> edit; ``None`` runs the clean model.
        read: Layers whose (edited) activations to return.
        layouts: Layouts, see :func:`innards.extract.describe`.
        input_fn: Maps the batch to ``(args, kwargs)``.

    Returns:
        ``(output, reads)``: the model output and layer name -> tokens
        ``(batch, tokens, d)`` (``(batch, d)`` for ``"bd"`` layers).
    """
    model, edits, read = wrap(model), edits or {}, _as_list(read)
    sites = list(dict.fromkeys([*edits, *read]))
    order, site_layouts = plan(model, batch, sites, layouts, input_fn) if sites else ([], {})
    args, kwargs = input_fn(batch)
    reads: dict[str, Tensor] = {}
    with model.trace(*args, **kwargs):
        for name in order:
            envoy, layout = model.get(name), site_layouts[name]
            if name in edits:
                act = edit_output(envoy, edits[name], layout)
            else:
                act = _first(envoy.output)
            if name in read:
                reads[name] = _reduce(act, layout, None).save()
        output = model.output.save()
    return output, reads
