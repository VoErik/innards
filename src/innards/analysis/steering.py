"""
Steering geometry: the spline partition along steering paths in activation space.

Steering adds ``alpha * v`` to a layer's activations ``h``. For a network of
affine layers and piecewise-linear activations, everything after the layer
is a continuous piecewise-affine function, so the output along the path
``y(alpha) = f(h + alpha v)`` is exactly piecewise linear in ``alpha``: it
bends only where the path crosses a unit's boundary (the unit switches on or
off), and within a piece its slope ``dy/dalpha = J v`` is the region's
Jacobian applied to ``v``. For smooth activations (GELU, SiLU) the crossings
are soft: units move probability between their pieces
(:func:`innards.analysis.spline.gate`).

* :func:`steer_path`: outputs, exact slopes, unit flips (or soft transition
  mass) and the crossings, with their positions, along ``h + alpha v``.
* :func:`slope_jumps`, :func:`decisive`: how much the slope changes between
  path points, along a readout, and the crossings that change it most.
* :func:`activation_slice`: the partition on a plane through ``h`` in
  activation space (the steering counterpart of
  :func:`innards.analysis.spline.slice_partition`).
* :func:`spline_steer`: steering that re-solves its direction from the local
  Jacobian at every step, so it turns whenever the path enters a new region.
* :func:`shift`: the edit adding a per-example vector, as used here.

Typical usage::

    layers = activation_layers(model, r"layer4")
    path = steer_path(model, images, "layer3", v, torch.linspace(0, 20, 201), layers)
    jumps = slope_jumps(path, target=cls)                  # (batch, steps - 1)
    where, size, seg = decisive(path, target=cls, k=3)     # the three largest slope changes per example
    ids, logits, (u, w) = activation_slice(model, images[0], "layer3", v, v2, layers)
    deltas, values = spline_steer(model, images, "layer3", logit(cls), steps=20, step_size=0.5)

References:
    Balestriero, R. & Baraniuk, R. (2018). A Spline Theory of Deep Learning.
        ICML 2018.
    Balestriero, R. & Baraniuk, R. (2019). From Hard to Soft: Understanding
        Deep Network Nonlinearities via Vector Quantization and Statistical
        Inference. ICLR 2019.
    Turner, A. et al. (2023). Steering Language Models With Activation
        Engineering. arXiv:2308.10248.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Sequence
)
from typing import (
    Any,
    NamedTuple
)

import torch
from torch import Tensor

from ..edits import (
    Edit,
    edit_output,
    run
)
from ..extract import (
    Layers,
    _as_list,
    plan
)
from ..layouts import (
    keep_mask,
    to_tokens
)
from ..models import wrap
from .spline import (
    Model,
    _gates,
    _math_attention,
    region_ids
)


def _identity(out: Any) -> Any:
    return out


def shift(delta: Tensor, exclude: Sequence[int] = (), mask: Tensor | None = None) -> Edit:
    """
    Adds a vector per example to the kept tokens.

    Args:
        delta: ``(batch, d)`` (the same vector at every kept token) or
            ``(batch, tokens, d)``.
        exclude: Positions left unchanged, e.g. ``(0,)`` for CLS.
        mask: Optional boolean ``(batch, tokens)`` mask of tokens to change.

    Returns:
        The edit.
    """

    def edit(tokens: Tensor) -> Tensor:
        keep = keep_mask(tokens, exclude, mask).unsqueeze(-1).to(tokens.dtype)
        d = delta.to(tokens.dtype)
        return tokens + keep * (d.unsqueeze(1) if d.ndim == 2 else d)

    return edit


def _steered(model, x: Tensor, layer: str, delta: Tensor, layers: list[str], exclude: Sequence[int],
             mask: Tensor | None, output_fn: Callable) -> tuple[Any, dict[str, Tensor]]:
    """One pass with ``delta`` added at ``layer``: output and the pre-activations of ``layers`` as tokens."""
    order, layouts = plan(model, x[:1], list(dict.fromkeys([layer, *layers])))
    pre: dict[str, Tensor] = {}
    with model.trace(x):
        for name in order:
            if name in layers:  # read before in-place activations overwrite it
                pre[name] = model.get(name).input.clone().save()
            if name == layer:
                edit_output(model.get(name), shift(delta, exclude, mask), layouts[name])
        out = model.output.save()
    return output_fn(out), {n: to_tokens(pre[n], layouts[n]).detach().float() for n in layers}


def _rate(y: Tensor, a: Tensor) -> Tensor:
    """``dy/da`` per row (rows independent) with one double backward."""
    dummy = torch.zeros_like(y, requires_grad=True)
    g = torch.autograd.grad(y, a, dummy, create_graph=True)[0]  # sum_c dummy_c dy_c / da, linear in dummy
    return torch.autograd.grad(g, dummy, torch.ones_like(g))[0].detach()


class Crossings(NamedTuple):
    """
    Unit boundaries crossed along steering paths, one entry per crossing.

    Attributes:
        example: Example index ``(n,)``.
        segment: Path segment ``j`` (between ``alphas[j]`` and ``alphas[j + 1]``) ``(n,)``.
        layer: Index into ``SteerPath.layers`` ``(n,)``.
        unit: Flat unit index ``token * units + unit`` within the layer ``(n,)``;
            ``SteerPath.shapes`` has ``(tokens, units)``.
        alpha: Where the pre-activation crosses 0, interpolated linearly
            between the segment's ends (exact if no other unit upstream
            switches within the segment) ``(n,)``.
    """

    example: Tensor
    segment: Tensor
    layer: Tensor
    unit: Tensor
    alpha: Tensor


class SteerPath(NamedTuple):
    """
    The partition along steering paths ``h + alpha v`` (see :func:`steer_path`).

    Attributes:
        alphas: Steering strengths ``(steps,)``.
        outputs: Outputs ``(batch, steps, ...)`` (after ``output_fn``).
        slopes: Exact ``dy/dalpha`` ``(batch, steps, ...)``, or ``None``.
        flips: Layer -> ``(batch, steps - 1)`` units switching per segment, or
            with ``soft`` the probability mass moved between pieces.
        crossings: The individual crossings (by hard code, i.e. pre-activation sign).
        layers: Activation layers, in the order of ``Crossings.layer``.
        shapes: Layer -> ``(tokens, units)``.
    """

    alphas: Tensor
    outputs: Tensor
    slopes: Tensor | None
    flips: dict[str, Tensor]
    crossings: Crossings
    layers: list[str]
    shapes: dict[str, tuple[int, int]]


def steer_path(
    model: Model,
    x: Tensor,
    layer: str,
    v: Tensor,
    alphas: Tensor | Sequence[float],
    layers: Layers,
    exclude: Sequence[int] = (),
    mask: Tensor | None = None,
    output_fn: Callable[[Any], Tensor] = _identity,
    soft: bool = False,
    derivative: bool = True,
    chunk: int = 64,
) -> SteerPath:
    """
    Follows steering paths ``h + alpha v`` through the partition of the downstream units.

    For piecewise-linear networks the output is exactly piecewise linear in
    ``alpha``, with kinks at the crossings and slope ``J_region v`` between
    them; ``slopes`` are exact derivatives at the path points (one double
    backward per chunk), so ``slope_jumps`` measure what each segment's
    crossings change. A dense ``alphas`` grid resolves the crossings; units
    switching twice within one segment are missed (flip counts are lower bounds).

    Args:
        model: An nnsight model or a plain module (eval mode).
        x: Inputs ``(batch, ...)`` on the model's device.
        layer: Steered layer.
        v: Direction ``(d,)`` added at every kept token, or ``(tokens, d)``.
        alphas: Steering strengths, increasing, e.g. ``torch.linspace(0, 10, 101)``.
        layers: Activation layers to track, after ``layer`` (e.g. from
            :func:`~innards.analysis.spline.activation_layers`).
        exclude: Positions not steered, e.g. ``(0,)`` for CLS.
        mask: Optional boolean ``(batch, tokens)`` mask of tokens to steer.
        output_fn: Maps the model output to a tensor ``(n, ...)``, e.g. logits.
        soft: Count flips as soft probability mass (for GELU / SiLU networks).
        derivative: Compute ``slopes`` (needs double backward).
        chunk: Path points per forward pass.

    Returns:
        A :class:`SteerPath`.
    """
    model, layers = wrap(model), _as_list(layers)
    alphas = torch.as_tensor(alphas, dtype=torch.float32).cpu()
    b, n_steps = x.shape[0], len(alphas)
    per = max(1, chunk // b)
    v = v.detach().to(x.device, torch.float32)
    gates = _gates(model, layers) if soft else {}
    outputs, slopes = [], []
    flips: dict[str, list[Tensor]] = {n: [] for n in layers}
    found: list[tuple[Tensor, ...]] = []
    prev: dict[str, Tensor] = {}
    shapes: dict[str, tuple[int, int]] = {}
    for j0 in range(0, n_steps, per):
        k = min(per, n_steps - j0)
        xs = x.unsqueeze(1).expand(b, k, *x.shape[1:]).reshape(b * k, *x.shape[1:])  # example-major
        a = alphas[j0 : j0 + k].repeat(b).to(x.device)
        m = mask.repeat_interleave(k, 0) if mask is not None else None
        with torch.enable_grad() if derivative else torch.no_grad(), _math_attention():
            a = a.requires_grad_(derivative)
            delta = a.reshape(-1, *[1] * v.ndim) * v
            y, pre = _steered(model, xs, layer, delta, layers, exclude, m, output_fn)
            y = y.float()
            if derivative:
                slopes.append(_rate(y, a).reshape(b, k, *y.shape[1:]).cpu())
        outputs.append(y.detach().reshape(b, k, *y.shape[1:]).cpu())
        for li, name in enumerate(layers):
            z = pre[name].reshape(b, k, -1).cpu()
            shapes[name] = tuple(pre[name].shape[1:])
            full = torch.cat([prev[name].unsqueeze(1), z], 1) if name in prev else z
            first = j0 - 1 if name in prev else j0  # global index of full[:, 0]
            on = full > 0
            changed = on[:, 1:] != on[:, :-1]
            if soft:
                g = gates[name](full)
                flips[name].append((g[:, 1:] - g[:, :-1]).abs().sum(-1))
            else:
                flips[name].append(changed.sum(-1))
            ex, seg, unit = changed.nonzero(as_tuple=True)
            z0, z1 = full[ex, seg, unit], full[ex, seg + 1, unit]
            a0, a1 = alphas[first + seg], alphas[first + seg + 1]
            where = a0 + z0 / (z0 - z1) * (a1 - a0)
            found.append((ex, first + seg, torch.full_like(ex, li), unit, where))
            prev[name] = z[:, -1]
    cols = list(zip(*found)) if found else [[torch.zeros(0, dtype=torch.long)]] * 4 + [[torch.zeros(0)]]
    crossings = Crossings(*(torch.cat(c) for c in cols))
    order = torch.argsort(crossings.example * (n_steps + 1) + crossings.segment, stable=True)
    return SteerPath(
        alphas=alphas,
        outputs=torch.cat(outputs, 1),
        slopes=torch.cat(slopes, 1) if derivative else None,
        flips={n: torch.cat(f, 1) for n, f in flips.items()},
        crossings=Crossings(*(c[order] for c in crossings)),
        layers=layers,
        shapes=shapes,
    )


def slope_jumps(path: SteerPath, target: int | Tensor | None = None) -> Tensor:
    """
    How much the slope of the output changes over each path segment.

    For piecewise-linear networks this is the summed slope change of the
    segment's crossings (zero where nothing crosses): ``target`` picks what
    the change is measured on.

    Args:
        path: From :func:`steer_path` with ``derivative=True``.
        target: ``None`` for the norm of the change over all outputs, an
            output index (e.g. a class), or a readout vector ``(...)`` (or
            one per example, ``(batch, ...)``) to project on.

    Returns:
        ``(batch, steps - 1)``; signed unless ``target`` is ``None``.
    """
    if path.slopes is None:
        raise ValueError("the path has no slopes; use steer_path(..., derivative=True)")
    d = path.slopes[:, 1:] - path.slopes[:, :-1]
    if target is None:
        return d.flatten(2).norm(dim=-1)
    if isinstance(target, int):
        return d.flatten(2)[..., target]
    w = torch.as_tensor(target, dtype=d.dtype)
    w = w.reshape(1, 1, -1) if w.ndim == d.ndim - 2 else w.reshape(w.shape[0], 1, -1)
    return (d.flatten(2) * w).sum(-1)


def decisive(path: SteerPath, target: int | Tensor | None = None, k: int = 5) -> tuple[Tensor, Tensor, Tensor]:
    """
    The path segments whose crossings change the slope most, per example.

    Args:
        path: From :func:`steer_path` with ``derivative=True``.
        target: As in :func:`slope_jumps`.
        k: Segments per example.

    Returns:
        ``(alpha, jump, segment)``, each ``(batch, k)``, largest ``|jump|``
        first: where the segment's first crossing lies (the segment's
        midpoint if no tracked unit crosses there), the slope change, and the
        segment index (select its crossings with ``path.crossings.segment``).
    """
    jumps = slope_jumps(path, target)
    k = min(k, jumps.shape[1])
    seg = jumps.abs().topk(k, dim=1).indices
    jump = jumps.gather(1, seg)
    alphas = path.alphas
    where = (alphas[seg] + alphas[seg + 1]) / 2
    c = path.crossings
    for i in range(jumps.shape[0]):
        mine = c.example == i
        for r, s in enumerate(seg[i].tolist()):
            hit = mine & (c.segment == s)
            if hit.any():
                where[i, r] = c.alpha[hit].min()
    return where, jump, seg


@torch.no_grad()
def activation_slice(
    model: Model,
    x: Tensor,
    layer: str,
    v1: Tensor,
    v2: Tensor,
    layers: Layers,
    res: int = 64,
    extent: tuple[float, float] = (-1.0, 1.0),
    exclude: Sequence[int] = (),
    mask: Tensor | None = None,
    output_fn: Callable[[Any], Tensor] = _identity,
    chunk: int = 256,
) -> tuple[dict[str, Tensor], Tensor, tuple[Tensor, Tensor]]:
    """
    Regions on the plane ``h + u v1 + w v2`` through one example's activations.

    The steering counterpart of
    :func:`~innards.analysis.spline.slice_partition`: the plane lies in the
    activation space of ``layer`` (the same shift at every kept token), so a
    steering path along ``v1`` is the line ``w = 0``. Regions are sampled on
    a grid.

    Args:
        model: An nnsight model or a plain module (eval mode).
        x: One input (no batch dimension), e.g. an image ``(3, H, W)``.
        layer: The layer whose activation space is sliced.
        v1: First direction ``(d,)`` or ``(tokens, d)``, e.g. the steering direction.
        v2: Second direction, e.g. another concept or a random direction.
        layers: Activation layers after ``layer``.
        res: Grid points per axis.
        extent: Range of ``u`` and ``w``.
        exclude: Positions not shifted.
        mask: Optional boolean ``(1, tokens)`` mask of tokens to shift.
        output_fn: Maps the model output to a tensor, e.g. logits.
        chunk: Grid points per forward pass.

    Returns:
        ``(ids, outputs, (u, w))``: layer -> ``(res, res)`` region ids of the
        network from ``layer`` up to and including that layer (in execution
        order), the outputs ``(res, res, ...)`` (e.g. ``argmax(-1)`` for the
        decision regions) and the grid coordinates. Draw with
        :func:`~innards.analysis.plot.partition` or
        :func:`~innards.analysis.spline.boundaries`.
    """
    model, layers = wrap(model), _as_list(layers)
    t = torch.linspace(*extent, res)
    u, w = torch.meshgrid(t, t, indexing="xy")
    coef = torch.stack([u.reshape(-1), w.reshape(-1)], 1).to(x.device)
    V = torch.stack([v1.detach(), v2.detach()]).to(x.device, torch.float32)  # (2, [tokens,] d)
    outs, codes = [], {n: [] for n in layers}
    for i in range(0, len(coef), chunk):
        c = coef[i : i + chunk]
        delta = torch.einsum("nk,k...->n...", c, V)
        m = mask.expand(len(c), -1) if mask is not None else None
        y, pre = _steered(model, x.unsqueeze(0).expand(len(c), *x.shape), layer, delta, layers, exclude, m,
                          output_fn)
        outs.append(y.float().cpu())
        for n in layers:
            codes[n].append((pre[n] > 0).cpu())
    order = plan(model, x.unsqueeze(0), layers)[0]
    ids, joined = {}, []
    for name in order:
        joined.append(torch.cat(codes[name]))
        ids[name] = region_ids(joined).reshape(res, res)
    out = torch.cat(outs)
    return ids, out.reshape(res, res, *out.shape[1:]), (u, w)


def spline_steer(
    model: Model,
    x: Tensor,
    layer: str,
    target: Callable[..., Tensor],
    steps: int = 20,
    step_size: float = 1.0,
    basis: Tensor | None = None,
    y: Tensor | None = None,
    exclude: Sequence[int] = (),
    mask: Tensor | None = None,
    output_fn: Callable[[Any], Tensor] = _identity,
) -> tuple[Tensor, Tensor]:
    """
    Steering that follows the local Jacobian: one vector per example, re-solved at every step.

    Each step adds ``step_size`` times the unit-norm gradient of the target
    with respect to the shift (shared by the kept tokens). In a
    piecewise-linear network that gradient is ``J_region^T w``, the exact
    best direction within the current region, so the path turns whenever it
    enters a new region, unlike a fixed steering vector. ``basis`` keeps the
    shift in a subspace, e.g. a few SAE decoder atoms or the data's principal
    directions. Apply the result with ``run(model, x, {layer: shift(deltas[:, -1])})``.

    Args:
        model: An nnsight model or a plain module (eval mode).
        x: Inputs ``(batch, ...)`` on the model's device.
        layer: Steered layer.
        target: ``fn(out, batch) -> (batch,)`` to increase, e.g.
            ``metrics.logit(3)`` or ``metrics.log_prob("label")``.
        steps: Number of steps.
        step_size: Norm of each step, in activation units.
        basis: Optional ``(k, d)`` directions the shift is confined to.
        y: Optional labels, passed to ``target`` as ``(x, y)``.
        exclude: Positions not steered, e.g. ``(0,)`` for CLS.
        mask: Optional boolean ``(batch, tokens)`` mask of tokens to steer.
        output_fn: Maps the model output to what ``target`` expects.

    Returns:
        ``(deltas, values)``: the shifts ``(batch, steps + 1, d)`` (starting at
        zero) and the target along the path ``(batch, steps + 1)``.
    """
    model = wrap(model)
    batch = (x, y) if y is not None else x
    with torch.no_grad():
        d = run(model, x[:1], read=layer)[1][layer].shape[-1]
    Q = None
    if basis is not None:
        Q = torch.linalg.qr(basis.detach().reshape(-1, d).T.float().to(x.device)).Q  # (d, k)
    delta = torch.zeros(x.shape[0], d, device=x.device)
    deltas, values = [delta.cpu()], []
    for i in range(steps + 1):
        with torch.enable_grad(), _math_attention():
            leaf = delta.clone().requires_grad_(True)
            out, _ = _steered(model, x, layer, leaf, [], exclude, mask, output_fn)
            value = target(out, batch)
            values.append(value.detach().float().cpu())
            if i == steps:
                break
            g = torch.autograd.grad(value.sum(), leaf)[0]
        if Q is not None:
            g = (g @ Q) @ Q.T
        delta = delta + step_size * g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        deltas.append(delta.cpu())
    return torch.stack(deltas, 1), torch.stack(values, 1)

