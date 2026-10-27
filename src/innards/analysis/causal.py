"""
Causal analysis: what the model uses, measured by intervening.

Built on :func:`innards.edits.run`. Interventions are dicts of edits (see
:mod:`innards.edits`; ``sae_edits`` for SAEs) and effects are measured with
the metrics of :mod:`innards.analysis.metrics`. Everything returns one value
per example (examples first), so effects can be averaged, bootstrapped
(``bootstrap_ci``) or related to per-example geometry.

* :func:`effect` / :func:`sweep`: the effect of an intervention, or of a
  family of them (steering strengths, layers, directions), over a dataset.
* :func:`patch_sweep`: activation patching from a clean into a corrupted
  input, per layer (and position).
* :func:`gradients`, :func:`attribution_patching`: the first-order versions.
* :func:`direction_attribution`: how much each direction of ``V`` contributes.
* :func:`relevant_subspace`: the directions the target is most sensitive to.

Typical usage::

    from innards.edits import ablate, steer

    values = effect(model, loader, {"blocks.6": ablate(V)}, change(correct()))
    values = sweep(model, loader, lambda a: {"blocks.6": steer(v, a)}, [0, 2, 4, 8], kl())
    patched = patch_sweep(model, clean, corrupt, ["blocks.2", "blocks.6"], change(logit("label")), positions=True)
    V_used, energy = relevant_subspace(model, loader, "blocks.6", logit("label"), k=10)

References:
    Meng, K. et al. (2022). Locating and Editing Factual Associations in GPT.
        NeurIPS 2022. (Activation patching.)
    Nanda, N. (2023). Attribution Patching: Activation Patching At Industrial
        Scale. Blog post, neelnanda.io.
    Constantine, P. (2015). Active Subspaces. SIAM. (Gradient-covariance
        subspaces.)
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterable,
    Sequence
)
from typing import Any

import torch
from nnsight.intervention.envoy import Envoy
from torch import (
    Tensor,
    nn
)

from ..edits import (
    Edit,
    patch,
    run
)
from ..extract import (
    Layers,
    _as_list,
    default_input_fn,
    to_device
)
from ..models import (
    unwrap,
    wrap
)
from .geometry import _2d
from .metrics import (
    Comparison,
    Target
)

Edits = dict[str, Edit] | Callable[[Any], dict[str, Edit]]
Model = nn.Module | Envoy


def _identity(out: Any) -> Any:
    return out


def _device(model: Envoy, device: torch.device | str | None) -> torch.device | str:
    return device if device is not None else next(unwrap(model).parameters()).device


def sweep(
    model: Model,
    batches: Iterable[Any],
    make_edits: Callable[[Any], Edits],
    values: Sequence[Any],
    metric: Comparison,
    output_fn: Callable[[Any], Tensor] = _identity,
    layouts: str | dict[str, str] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Effects of a family of interventions, e.g. steering strengths or layers.

    The clean output is computed once per batch; the batches are iterated
    once, so any iterable works.

    Args:
        model: An nnsight model or a plain module.
        batches: Iterable of batches, e.g. a ``DataLoader``.
        make_edits: Maps a value to edits: a dict layer -> edit, or a function
            of the batch returning one (for edits that depend on the batch).
        values: The values to sweep.
        metric: ``fn(clean, edited, batch) -> (batch,)``, see
            :mod:`innards.analysis.metrics`.
        output_fn: Maps the model output to what ``metric`` expects, e.g.
            ``lambda out: out.logits[:, -1]``.
        layouts: Layouts of the edited layers, see :func:`innards.extract.describe`.
        input_fn: Maps a batch to model ``(args, kwargs)``.
        device: Device the batches are moved to; defaults to the model's.

    Returns:
        ``(n, len(values))`` metric values per example and value.
    """
    model = wrap(model)
    device = _device(model, device)
    cols: list[list[Tensor]] = [[] for _ in values]
    for batch in batches:
        batch = to_device(batch, device)
        with torch.no_grad():
            clean = output_fn(run(model, batch, input_fn=input_fn)[0])
            for i, value in enumerate(values):
                edits = make_edits(value)
                edits = edits(batch) if callable(edits) else edits
                edited = output_fn(run(model, batch, edits, layouts=layouts, input_fn=input_fn)[0])
                cols[i].append(metric(clean, edited, batch).detach().cpu())
    return torch.stack([torch.cat(c) for c in cols], 1)


def effect(
    model: Model,
    batches: Iterable[Any],
    edits: Edits,
    metric: Comparison,
    output_fn: Callable[[Any], Tensor] = _identity,
    layouts: str | dict[str, str] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Effect of one intervention over a dataset.

    Args:
        model: An nnsight model or a plain module.
        batches: Iterable of batches.
        edits: Dict layer -> edit, or a function of the batch returning one.
        metric: ``fn(clean, edited, batch) -> (batch,)``.
        output_fn: Maps the model output to what ``metric`` expects.
        layouts: Layouts of the edited layers.
        input_fn: Maps a batch to model ``(args, kwargs)``.
        device: Device the batches are moved to; defaults to the model's.

    Returns:
        ``(n,)`` metric values; ``.mean()`` or ``bootstrap_ci`` to summarise.
    """
    return sweep(model, batches, lambda _: edits, [None], metric, output_fn, layouts, input_fn, device)[:, 0]


def patch_sweep(
    model: Model,
    clean_batch: Any,
    corrupt_batch: Any,
    layers: Layers,
    metric: Comparison,
    positions: bool = False,
    output_fn: Callable[[Any], Tensor] = _identity,
    layouts: str | dict[str, str] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    """
    Activation patching: run the corrupted input with clean activations patched in.

    Each layer (and with ``positions`` each token position separately) is
    patched in its own run. The metric compares the corrupted output with
    the patched one; e.g. ``change(logit("label"))`` measures how much of the
    clean label's logit is restored.

    Args:
        model: An nnsight model or a plain module.
        clean_batch: Clean inputs.
        corrupt_batch: Corrupted inputs with the same token grid (and batch size).
        layers: Layers to patch.
        metric: ``fn(corrupt_out, patched_out, batch) -> (batch,)``; it gets
            ``clean_batch``, so ``"label"`` refers to the clean labels.
        positions: Patch every token position separately.
        output_fn: Maps the model output to what ``metric`` expects.
        layouts: Layouts of the layers.
        input_fn: Maps a batch to model ``(args, kwargs)``.
        device: Device the batches are moved to; defaults to the model's.

    Returns:
        Layer name -> ``(batch,)`` values, or ``(batch, tokens)`` with ``positions``.
    """
    model, layers = wrap(model), _as_list(layers)
    device = _device(model, device)
    clean_batch, corrupt_batch = to_device(clean_batch, device), to_device(corrupt_batch, device)
    out: dict[str, Tensor] = {}
    with torch.no_grad():
        _, clean = run(model, clean_batch, read=layers, layouts=layouts, input_fn=input_fn)
        corrupt = output_fn(run(model, corrupt_batch, input_fn=input_fn)[0])

        def patched(name: str, mask: Tensor | None = None) -> Tensor:
            res, _ = run(model, corrupt_batch, {name: patch(clean[name], mask=mask)}, layouts=layouts,
                         input_fn=input_fn)
            return metric(corrupt, output_fn(res), clean_batch).detach().cpu()

        for name in layers:
            tokens = clean[name]
            if not positions or tokens.ndim == 2:
                out[name] = patched(name)
                continue
            cols = []
            for p in range(tokens.shape[1]):
                mask = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
                mask[:, p] = True
                cols.append(patched(name, mask))
            out[name] = torch.stack(cols, 1)
    return out


def gradients(
    model: Model,
    batch: Any,
    layers: Layers,
    target: Target,
    output_fn: Callable[[Any], Tensor] = _identity,
    layouts: str | dict[str, str] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    target_batch: Any = None,
) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
    """
    Gradients of a target with respect to layer activations.

    Uses ``torch.autograd.grad``, so parameter gradients are left untouched.

    Args:
        model: An nnsight model or a plain module.
        batch: An input batch (on the model's device).
        layers: Layers to differentiate.
        target: ``fn(out, batch) -> (batch,)``, e.g. ``logit("label")``; the
            gradient is that of its sum (examples are independent).
        output_fn: Maps the model output to what ``target`` expects.
        layouts: Layouts of the layers.
        input_fn: Maps the batch to model ``(args, kwargs)``.
        target_batch: Batch passed to ``target`` (for its labels); defaults
            to ``batch``.

    Returns:
        ``(grads, acts, values)``: layer name -> gradient and activation
        tokens ``(batch, tokens, d)`` (one token for ``"bd"`` layers), and
        the target values ``(batch,)``.
    """
    model, layers = wrap(model), _as_list(layers)
    store: dict[str, Tensor] = {}

    def tap(name: str) -> Edit:
        def edit(tokens: Tensor) -> Tensor:
            if not tokens.requires_grad:  # frozen model: the first tapped layer becomes a leaf
                tokens = tokens.detach().requires_grad_(True)
            store[name] = tokens
            return tokens
        return edit

    with torch.enable_grad():
        out, _ = run(model, batch, {n: tap(n) for n in layers}, layouts=layouts, input_fn=input_fn)
        values = target(output_fn(out), batch if target_batch is None else target_batch)
        grads = torch.autograd.grad(values.sum(), [store[n] for n in layers], allow_unused=True)
    result_g, result_a = {}, {}
    for n, g in zip(layers, grads):
        a = store[n].detach()
        g = torch.zeros_like(a) if g is None else g
        result_g[n], result_a[n] = g, a
    return result_g, result_a, values.detach()


def attribution_patching(
    model: Model,
    clean_batch: Any,
    corrupt_batch: Any,
    layers: Layers,
    target: Target,
    output_fn: Callable[[Any], Tensor] = _identity,
    layouts: str | dict[str, str] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    """
    First-order estimate of :func:`patch_sweep` with ``positions=True`` (Nanda, 2023).

    ``(h_clean - h_corrupt) . dT/dh`` at the corrupted run: the change in the
    target when a position's clean activation is patched in, for every layer
    and position from one forward and one backward pass. Exact where the
    target is linear in the patched activation.

    Args:
        model: An nnsight model or a plain module.
        clean_batch: Clean inputs.
        corrupt_batch: Corrupted inputs with the same token grid.
        layers: Layers to attribute.
        target: ``fn(out, batch) -> (batch,)``; it gets ``clean_batch``.
        output_fn: Maps the model output to what ``target`` expects.
        layouts: Layouts of the layers.
        input_fn: Maps a batch to model ``(args, kwargs)``.
        device: Device the batches are moved to; defaults to the model's.

    Returns:
        Layer name -> ``(batch, tokens)`` estimated target changes.
    """
    model, layers = wrap(model), _as_list(layers)
    device = _device(model, device)
    clean_batch, corrupt_batch = to_device(clean_batch, device), to_device(corrupt_batch, device)
    with torch.no_grad():
        _, clean = run(model, clean_batch, read=layers, layouts=layouts, input_fn=input_fn)
    grads, acts, _ = gradients(model, corrupt_batch, layers, target, output_fn, layouts, input_fn,
                               target_batch=clean_batch)
    out = {}
    for n in layers:
        h_clean = clean[n].reshape(acts[n].shape)
        out[n] = ((h_clean - acts[n]) * grads[n]).sum(-1).detach().cpu()
    return out


def direction_attribution(
    model: Model,
    batches: Iterable[Any],
    layer: str,
    V: Tensor,
    target: Target,
    output_fn: Callable[[Any], Tensor] = _identity,
    layouts: str | dict[str, str] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    First-order contribution of each direction to a target.

    For a unit direction ``v``, ``sum_tokens (h . v)(g . v)`` with ``g`` the
    target's gradient: the linear estimate of ``T(x) - T(x with v ablated)``
    (``ablate(v)``). Positive directions support the target. Each direction
    is scored on its own, so for overlapping directions the scores overlap.

    Args:
        model: An nnsight model or a plain module.
        batches: Iterable of batches.
        layer: The layer.
        V: Directions ``(k, d)`` or ``(d,)``, e.g. from a probe, PCA or an SAE.
        target: ``fn(out, batch) -> (batch,)``.
        output_fn: Maps the model output to what ``target`` expects.
        layouts: Layout of the layer.
        input_fn: Maps a batch to model ``(args, kwargs)``.
        device: Device the batches are moved to; defaults to the model's.

    Returns:
        ``(n, k)`` contributions.
    """
    model = wrap(model)
    device = _device(model, device)
    U = _2d(V)
    U = U / U.norm(dim=1, keepdim=True)
    rows = []
    for batch in batches:
        batch = to_device(batch, device)
        grads, acts, _ = gradients(model, batch, layer, target, output_fn, layouts, input_fn)
        h, g = acts[layer].double(), grads[layer].double()
        u = U.to(h.device)
        hv, gv = h @ u.T, g @ u.T
        prod = hv * gv
        rows.append(prod.sum(1).cpu())
    return torch.cat(rows)


def relevant_subspace(
    model: Model,
    batches: Iterable[Any],
    layer: str,
    target: Target,
    k: int = 10,
    output_fn: Callable[[Any], Tensor] = _identity,
    layouts: str | dict[str, str] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """
    The directions of a layer the target is most sensitive to (an active subspace).

    Top eigenvectors of the uncentered gradient covariance ``E[g g^T]`` over
    all examples and tokens (Constantine, 2015): moving the activation along
    them changes the target most, to first order. Compare it with what the
    representation spends its variance on (``pca``) via ``subspace_angles``.

    Args:
        model: An nnsight model or a plain module.
        batches: Iterable of batches.
        layer: The layer.
        target: ``fn(out, batch) -> (batch,)``.
        k: Number of directions.
        output_fn: Maps the model output to what ``target`` expects.
        layouts: Layout of the layer.
        input_fn: Maps a batch to model ``(args, kwargs)``.
        device: Device the batches are moved to; defaults to the model's.

    Returns:
        ``(V, energy)``: unit directions ``(k, d)``, most sensitive first, and
        the fraction of the gradient energy each captures ``(k,)``.
    """
    model = wrap(model)
    device = _device(model, device)
    cov, n = None, 0
    for batch in batches:
        batch = to_device(batch, device)
        grads, _, _ = gradients(model, batch, layer, target, output_fn, layouts, input_fn)
        g = grads[layer].double().reshape(-1, grads[layer].shape[-1])
        cov = g.T @ g if cov is None else cov + g.T @ g
        n += g.shape[0]
    evals, evecs = torch.linalg.eigh(cov / n)
    order = evals.flip(0)
    return evecs.flip(1)[:, :k].T.cpu(), (order[:k] / order.sum()).cpu()
