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

* :func:`steer_path`: outputs, exact slopes, and every crossing along
  ``h + alpha v`` with its position and the slope change it causes
  (segments with several kinks are bisected until each kink is alone), or
  the soft transition mass for smooth networks.
* :func:`slope_jumps`, :func:`decisive`: how much the slope changes per grid
  segment, along a readout, and the crossings that change it most.
* :func:`activation_slice`: the partition on a plane through ``h`` in
  activation space (the steering counterpart of
  :func:`innards.analysis.spline.slice_partition`).
* :func:`spline_steer`: steering that re-solves its direction from the local
  Jacobian at every step, so it turns whenever the path enters a new region.
* :func:`shift`: the edit adding a per-example vector, as used here.

Typical usage::

    path = steer_path(model, images, "layer3", v, torch.linspace(0, 20, 201), target=cls)  # all units after layer3
    path.crossings.alpha, path.crossings.jump              # every crossing, and how much it bends the class logit
    where, size, idx = decisive(path, k=3)                 # the three crossings that bend it most, per example
    ids, logits, (u, w) = activation_slice(model, images[0], "layer3", v, v2, activation_layers(model, r"layer4"))
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
             mask: Tensor | None, output_fn: Callable, keep_graph: bool = False) -> tuple[Any, dict[str, Tensor]]:
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
    pre = {n: to_tokens(pre[n], layouts[n]).float() for n in layers}
    return output_fn(out), (pre if keep_graph else {n: p.detach() for n, p in pre.items()})


def _rates(outs: list[Tensor], a: Tensor) -> list[Tensor]:
    """``d out / da`` per row for several outputs (rows independent), with one double backward."""
    dummies = [torch.zeros_like(o, requires_grad=True) for o in outs]
    g = torch.autograd.grad(outs, a, dummies, create_graph=True)[0]  # linear in the dummies
    ds = torch.autograd.grad(g, dummies, torch.ones_like(g), allow_unused=True)
    return [torch.zeros_like(o) if d is None else d.detach() for o, d in zip(outs, ds)]


class Crossings(NamedTuple):
    """
    Unit boundaries crossed along steering paths, one entry per crossing.

    Attributes:
        example: Example index ``(n,)``.
        segment: Grid segment ``j`` (between ``alphas[j]`` and ``alphas[j + 1]``) ``(n,)``.
        layer: Index into ``SteerPath.layers``; ``-1`` for a kink of the
            output whose unit is not tracked ``(n,)``.
        unit: Flat unit index ``token * units + unit`` within the layer (``-1``
            for untracked kinks); ``SteerPath.shapes`` has ``(tokens, units)``.
        alpha: Where the unit's pre-activation crosses 0 ``(n,)``.
        jump: The change of the output's slope at this crossing, as a norm or
            along ``target`` (see :func:`steer_path`); 0 for crossings that do
            not change it, ``nan`` where the segment was not resolved ``(n,)``.
        exact: Whether the crossing comes from a resolved segment (exact up to
            floating point) or from a sign change between two points of an
            unresolved one (position interpolated, jump unknown) ``(n,)``.
    """

    example: Tensor
    segment: Tensor
    layer: Tensor
    unit: Tensor
    alpha: Tensor
    jump: Tensor
    exact: Tensor


class SteerPath(NamedTuple):
    """
    The partition along steering paths ``h + alpha v`` (see :func:`steer_path`).

    Attributes:
        alphas: Steering strengths ``(steps,)``.
        outputs: Outputs ``(batch, steps, ...)`` (after ``output_fn``).
        slopes: Exact ``dy/dalpha`` ``(batch, steps, ...)``, or ``None``.
        flips: Layer -> ``(batch, steps - 1)`` crossings per segment (each
            crossing of a unit counts, so a unit switching on and off again
            counts twice), or with ``soft`` the probability mass moved between pieces.
        crossings: The individual crossings, sorted by example and ``alpha``.
        resolved: ``(batch, steps - 1)`` whether every crossing of the segment
            is exact (always ``False`` without slopes or with ``soft``).
        layers: Activation layers, in the order of ``Crossings.layer``.
        shapes: Layer -> ``(tokens, units)``.
    """

    alphas: Tensor
    outputs: Tensor
    slopes: Tensor | None
    flips: dict[str, Tensor]
    crossings: Crossings
    resolved: Tensor
    layers: list[str]
    shapes: dict[str, tuple[int, int]]


class _State(NamedTuple):
    y: Tensor  # (n, outputs)
    s: Tensor | None  # dy/dalpha
    z: Tensor  # (n, units) tracked pre-activations, layers concatenated
    zd: Tensor | None  # dz/dalpha


def _cat_states(states: Sequence[_State]) -> _State:
    return _State(*(None if parts[0] is None else torch.cat(parts) for parts in zip(*states)))


def _take(state: _State, idx: Tensor) -> _State:
    return _State(*(None if t is None else t[idx] for t in state))


def _project(d: Tensor, target: int | Tensor | None) -> Tensor:
    """Slope changes ``(n, outputs)`` -> ``(n,)``: norm, one output, or along a readout."""
    if target is None:
        return d.norm(dim=-1)
    if isinstance(target, int):
        return d[:, target]
    return d @ torch.as_tensor(target, dtype=d.dtype).reshape(-1)


def _events(ex, seg, unit, alpha, jump, exact) -> tuple[Tensor, ...]:
    n = len(unit)
    return (ex.expand(n) if ex.ndim == 0 else ex, seg.expand(n) if seg.ndim == 0 else seg, unit, alpha,
            jump if jump.ndim else jump.expand(n), torch.full((n,), exact))


def _resolve(ex: Tensor, seg: Tensor, a0: Tensor, a1: Tensor, S0: _State, S1: _State, tol: float,
             target) -> tuple[Tensor, list[tuple[Tensor, ...]]]:
    """
    Tests segments for at most one kink and returns their crossings.

    Between two points of a piecewise-linear network every tracked
    pre-activation and the output are affine until something crosses. A
    segment is resolved if all of them fit one line, or two lines meeting at
    one common kink ``alpha*``; units then cross at most once per side, at
    exactly known places, and the slope change belongs to the unit that is 0
    at ``alpha*`` (``layer -1`` if no tracked unit is).

    Returns:
        ``(ok, events)``: which segments are resolved, and their crossings as
        tuples ``(example, segment, flat unit, alpha, jump, exact)``.
    """
    D = (a1 - a0).double()[:, None]
    lo, hi = a0.double()[:, None], a1.double()[:, None]
    vals = [(S0.z.double(), S1.z.double(), S0.zd.double(), S1.zd.double()),
            (S0.y.double(), S1.y.double(), S0.s.double(), S1.s.double())]
    kinks, scales = [], []
    eps = 64 * torch.finfo(torch.float32).eps  # values come from float32 passes: rounding of the layer's magnitude
    for v0, v1, d0, d1 in vals:
        scale = tol * (v0.abs() + v1.abs() + (d0.abs() + d1.abs()) * D) + eps * v0.abs().amax(1, keepdim=True) + 1e-12
        # any slope change is a kink (relative to the slopes, so short segments and kinks near an end count)
        strong = (d1 - d0).abs() > tol * (d0.abs() + d1.abs()) + eps * d0.abs().amax(1, keepdim=True) + 1e-12
        star = lo + (v1 - v0 - d1 * D) / torch.where(strong, d0 - d1, torch.ones_like(d0))
        sharp = (d1 - d0).abs() / (d0.abs() + d1.abs() + 1e-30)  # relative slope change, up to 2
        kinks.append((torch.where(strong, star, torch.full_like(star, float("nan"))), torch.where(strong, sharp, 0)))
        scales.append(scale)
    stars, sharps = (torch.cat(t, 1) for t in zip(*kinks))
    clear = sharps >= 0.1 * sharps.amax(1, keepdim=True)  # locate the kink from the clearly kinked values
    star = torch.where(clear, stars, torch.full_like(stars, float("nan"))).nanmedian(1).values  # (n,) nan: no kink
    kinked = ~star.isnan()
    t = torch.where(kinked, star, a1.double())[:, None]
    ok = kinked.logical_not() | ((star >= a0.double()) & (star <= a1.double()))
    for (v0, v1, d0, d1), scale in zip(vals, scales):  # one line, or two lines meeting at alpha*
        pred = v0 + d0 * (t - lo) + d1 * (hi - t)
        ok &= ((pred - v1).abs() <= scale).all(1)
    events = []
    z0, z1, d0, d1 = vals[0]
    jumps = _project((S1.s - S0.s).double(), target) if S0.s is not None else None
    for i in ok.nonzero().squeeze(-1).tolist():
        e, sg, a_lo, a_hi = ex[i], seg[i], float(a0[i]), float(a1[i])
        a_star = float(star[i]) if kinked[i] else a_hi
        zm = z0[i] + d0[i] * (a_star - a_lo)  # at the kink (no kink: the end)
        left = ((z0[i] > 0) != (zm > 0)).nonzero().squeeze(-1)
        right = ((zm > 0) != (z1[i] > 0)).nonzero().squeeze(-1)
        if len(left) + len(right) > 1:  # a tracked crossing bends what follows: one crossing per segment
            ok[i] = False
            continue
        crossing = None
        if len(left):
            u = int(left[0])
            crossing = (u, a_lo - float(z0[i, u] / d0[i, u]))
        elif len(right):
            u = int(right[0])
            crossing = (u, a_hi - float(z1[i, u] / d1[i, u]))
        if not kinked[i]:
            if crossing is not None:  # a unit without visible downstream effect
                events.append(_events(e, sg, torch.tensor([crossing[0]]), torch.tensor([crossing[1]]),
                                      torch.zeros(()), True))
            continue
        jump = jumps[i].float()
        if crossing is not None:
            u, a_u = crossing
            own_kink = (d1[i, u] - d0[i, u]).abs() > tol * (d0[i, u].abs() + d1[i, u].abs()) + 1e-12
            if not own_kink and abs(a_u - a_star) <= 0.05 * (a_hi - a_lo):  # the unit causing the kink
                events.append(_events(e, sg, torch.tensor([u]), torch.tensor([a_u]), jump, True))
                continue
            events.append(_events(e, sg, torch.tensor([u]), torch.tensor([a_u]), torch.zeros(()), True))
        # the kink comes from a unit that is not tracked
        events.append(_events(e, sg, torch.tensor([-1]), torch.tensor([a_star]), jump, True))
    return ok, events


def _fallback(ex, seg, a0, a1, S0: _State, S1: _State) -> list[tuple[Tensor, ...]]:
    """Crossings from sign changes between the ends, positions interpolated, jumps unknown."""
    changed = (S0.z > 0) != (S1.z > 0)
    i, unit = changed.nonzero(as_tuple=True)
    z0, z1 = S0.z[i, unit], S1.z[i, unit]
    alpha = a0[i] + z0 / (z0 - z1) * (a1[i] - a0[i])
    return [_events(ex[i], seg[i], unit, alpha.float(), torch.full((len(i),), float("nan")), False)]


def _after(model, x: Tensor, layer: str) -> list[str]:
    from .spline import activation_layers

    acts = activation_layers(model)
    order = plan(model, x[:1], [layer, *acts])[0]
    return [n for n in order[order.index(layer) + 1 :] if n in acts]


def steer_path(
    model: Model,
    x: Tensor,
    layer: str,
    v: Tensor,
    alphas: Tensor | Sequence[float],
    layers: Layers | None = None,
    exclude: Sequence[int] = (),
    mask: Tensor | None = None,
    output_fn: Callable[[Any], Tensor] = _identity,
    target: int | Tensor | None = None,
    refine: int = 8,
    tol: float = 1e-4,
    soft: bool = False,
    derivative: bool = True,
    chunk: int = 64,
) -> SteerPath:
    """
    Follows steering paths ``h + alpha v`` through the partition of the downstream units.

    For piecewise-linear networks the output is exactly piecewise linear in
    ``alpha``, with kinks at the crossings and slope ``J_region v`` between
    them. With exact derivatives at the path points, each segment between
    two points is checked for at most one kink (see ``Crossings``); segments
    with more are bisected, up to ``refine`` times. In resolved segments
    every crossing is found exactly, including units that switch on and off
    again, and the slope change of the output is assigned to the crossing
    that causes it. Segments still unresolved (more kinks than ``refine``
    bisections separate, or crossings closer than float32 can tell apart)
    fall back to sign changes between their ends (``SteerPath.resolved``
    says where). Track every
    activation after ``layer`` (the default) so kinks can be attributed to
    units; kinks of untracked units appear with ``layer -1``.

    Args:
        model: An nnsight model or a plain module (eval mode).
        x: Inputs ``(batch, ...)`` on the model's device.
        layer: Steered layer.
        v: Direction ``(d,)`` added at every kept token, or ``(tokens, d)``.
        alphas: Steering strengths, increasing, e.g. ``torch.linspace(0, 10, 101)``.
        layers: Activation layers to track; ``None`` takes every activation
            layer after ``layer`` (see :func:`~innards.analysis.spline.activation_layers`).
        exclude: Positions not steered, e.g. ``(0,)`` for CLS.
        mask: Optional boolean ``(batch, tokens)`` mask of tokens to steer.
        output_fn: Maps the model output to a tensor ``(n, ...)``, e.g. logits.
        target: What ``Crossings.jump`` measures: ``None`` for the norm of the
            slope change, an output index (e.g. a class) or a readout vector.
        refine: Maximum bisections per grid segment (each level evaluates
            the midpoints of all unresolved segments of a grid step at once).
            Resolving ``m`` crossings exactly costs about ``2 m`` path points;
            segments with more sign changes between their ends than
            ``2^refine`` can separate fall back at once. Dense partitions
            (ResNet stages: ~10^5 crossings per unit ``alpha`` per image)
            need short paths, or a fine ``alphas`` grid and ``slope_jumps``.
        tol: Relative tolerance of the one-kink test.
        soft: Count flips as soft probability mass (for GELU / SiLU networks;
            no refinement, as there are no affine pieces).
        derivative: Compute ``slopes`` (needs double backward; without them
            nothing is refined).
        chunk: Path points per forward pass.

    Returns:
        A :class:`SteerPath`.
    """
    model = wrap(model)
    layers = _after(model, x, layer) if layers is None else _as_list(layers)
    alphas = torch.as_tensor(alphas, dtype=torch.float32).cpu()
    b, n_steps = x.shape[0], len(alphas)
    v = v.detach().to(x.device, torch.float32)
    exact_mode = derivative and not soft
    shapes: dict[str, tuple[int, int]] = {}

    def evaluate(ex: Tensor, a: Tensor) -> _State:
        states = []
        for i in range(0, len(ex), chunk):
            e, al = ex[i : i + chunk], a[i : i + chunk].to(x.device)
            m = mask[e.to(mask.device)] if mask is not None else None
            with torch.enable_grad() if derivative else torch.no_grad(), _math_attention():
                al = al.clone().requires_grad_(derivative)
                delta = al.reshape(-1, *[1] * v.ndim) * v
                y, pre = _steered(model, x[e.to(x.device)], layer, delta, layers, exclude, m, output_fn,
                                  keep_graph=derivative)
                y = y.float().reshape(len(e), -1)
                z = torch.cat([pre[n].reshape(len(e), -1) for n in layers], 1)
                for n in layers:
                    shapes[n] = tuple(pre[n].shape[1:])
                s, zd = _rates([y, z], al) if derivative else (None, None)
            states.append(_State(y.detach().cpu(), None if s is None else s.cpu(), z.detach().cpu(),
                                 None if zd is None else zd.cpu()))
        return _cat_states(states)

    per = max(2, chunk // b)
    outputs, slopes, events = [], [], []
    resolved = torch.zeros(b, max(n_steps - 1, 0), dtype=torch.bool)
    soft_mass: list[Tensor] = []
    prev: _State | None = None
    gates = _gates(model, layers) if soft else {}
    examples = torch.arange(b)
    for j0 in range(0, n_steps, per):
        k = min(per, n_steps - j0)
        ex = examples.repeat_interleave(k)
        cur = evaluate(ex, alphas[j0 : j0 + k].repeat(b))
        grid = _State(*(None if t is None else t.reshape(b, k, *t.shape[1:]) for t in cur))
        outputs.append(grid.y)
        if derivative:
            slopes.append(grid.s)
        if prev is not None:  # carry the last point of the previous block
            grid = _State(*(None if t is None else torch.cat([p.unsqueeze(1), t], 1) for t, p in zip(grid, prev)))
        first = j0 - 1 if prev is not None else j0
        if soft:
            parts = []
            for n, (lo, hi) in zip(layers, _bounds(layers, shapes)):
                g = gates[n](grid.z[..., lo:hi])
                parts.append((g[:, 1:] - g[:, :-1]).abs().sum(-1))
            soft_mass.append(torch.stack(parts, -1))
        for j in range(grid.y.shape[1] - 1):
            seg = torch.full((b,), first + j)
            S0, S1 = _take(grid, (examples, j)), _take(grid, (examples, j + 1))
            a0, a1 = alphas[seg], alphas[seg + 1]
            todo = (examples, seg, a0, a1, S0, S1)
            resolved[:, first + j] = exact_mode  # until a sub-segment falls back
            depth = 0
            while todo is not None and len(todo[0]):
                ex_t, seg_t, a0_t, a1_t, S0_t, S1_t = todo
                if not exact_mode or depth >= refine:
                    ok = torch.zeros(len(ex_t), dtype=torch.bool)
                    if exact_mode:
                        ok, found = _resolve(ex_t, seg_t, a0_t, a1_t, S0_t, S1_t, tol, target)
                        events += found
                    bad = (~ok).nonzero().squeeze(-1)
                    events += _fallback(ex_t[bad], seg_t[bad], a0_t[bad], a1_t[bad], _take(S0_t, bad), _take(S1_t, bad))
                    resolved[ex_t[bad], seg_t[bad]] = False
                    break
                ok, found = _resolve(ex_t, seg_t, a0_t, a1_t, S0_t, S1_t, tol, target)
                events += found
                bad = (~ok).nonzero().squeeze(-1)
                # sign changes bound the crossings from below; refine - depth bisections separate at most 2^(...)
                lower = ((S0_t.z[bad] > 0) != (S1_t.z[bad] > 0)).sum(1)
                hopeless = bad[lower > 2 ** (refine - depth)]
                if len(hopeless):
                    events += _fallback(ex_t[hopeless], seg_t[hopeless], a0_t[hopeless], a1_t[hopeless],
                                        _take(S0_t, hopeless), _take(S1_t, hopeless))
                    resolved[ex_t[hopeless], seg_t[hopeless]] = False
                bad = bad[lower <= 2 ** (refine - depth)]
                if not len(bad):
                    break
                ex_b, seg_b, a0_b, a1_b = ex_t[bad], seg_t[bad], a0_t[bad], a1_t[bad]
                mid = (a0_b + a1_b) / 2
                Sm = evaluate(ex_b, mid)
                S0_b, S1_b = _take(S0_t, bad), _take(S1_t, bad)
                todo = (ex_b.repeat(2), seg_b.repeat(2), torch.cat([a0_b, mid]), torch.cat([mid, a1_b]),
                        _cat_states([S0_b, Sm]), _cat_states([Sm, S1_b]))
                depth += 1
        prev = _take(grid, (examples, -1))

    cols = list(zip(*events)) if events else [[torch.zeros(0, dtype=torch.long)]] * 3 + [[torch.zeros(0)]] * 2 \
        + [[torch.zeros(0, dtype=torch.bool)]]
    ex_c, seg_c, flat, alpha, jump, exact = (torch.cat([t.reshape(-1) for t in c]) for c in cols)
    ex_c, seg_c, flat = ex_c.long(), seg_c.long(), flat.long()
    offsets = torch.tensor([0, *torch.tensor([hi for _, hi in _bounds(layers, shapes)]).tolist()])
    layer_c = torch.where(flat >= 0, torch.bucketize(flat, offsets[1:], right=True), torch.full_like(flat, -1))
    unit_c = torch.where(flat >= 0, flat - offsets[layer_c.clamp_min(0)], flat)
    order = torch.argsort(ex_c.double() * (n_steps + 1) + alpha.double().nan_to_num(), stable=True)
    crossings = Crossings(ex_c, seg_c, layer_c, unit_c, alpha.float(), jump.float(), exact.bool())
    crossings = Crossings(*(c[order] for c in crossings))
    if soft:
        mass = torch.cat(soft_mass, 1)
        flips = {n: mass[..., i] for i, n in enumerate(layers)}
    else:
        flips = {}
        for i, n in enumerate(layers):
            hit = crossings.layer == i
            counts = torch.zeros(b * max(n_steps - 1, 0))
            counts.index_add_(0, crossings.example[hit] * (n_steps - 1) + crossings.segment[hit],
                              torch.ones(int(hit.sum())))
            flips[n] = counts.reshape(b, -1).long()
    out = torch.cat(outputs, 1)
    sl = torch.cat(slopes, 1) if derivative else None
    return SteerPath(
        alphas=alphas,
        outputs=out,
        slopes=sl,
        flips=flips,
        crossings=crossings,
        resolved=resolved,
        layers=layers,
        shapes=shapes,
    )


def _bounds(layers: list[str], shapes: dict[str, tuple[int, int]]) -> list[tuple[int, int]]:
    out, lo = [], 0
    for n in layers:
        size = shapes[n][0] * shapes[n][1]
        out.append((lo, lo + size))
        lo += size
    return out


def slope_jumps(path: SteerPath, target: int | Tensor | None = None) -> Tensor:
    """
    How much the slope of the output changes over each grid segment.

    For piecewise-linear networks this is the summed slope change of the
    segment's crossings (zero where nothing crosses); per crossing, see
    ``SteerPath.crossings.jump``.

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


def decisive(path: SteerPath, k: int = 5) -> tuple[Tensor, Tensor, Tensor]:
    """
    The crossings that change the output's slope most, per example.

    Uses the exact per-crossing jumps (along the ``target`` given to
    :func:`steer_path`); crossings of unresolved segments are skipped.

    Args:
        path: From :func:`steer_path`.
        k: Crossings per example.

    Returns:
        ``(alpha, jump, index)``, each ``(batch, k)``, largest ``|jump|``
        first: where the crossing lies, its slope change, and its index into
        ``path.crossings`` (``-1``, with ``nan`` values, if an example has
        fewer than ``k``).
    """
    c = path.crossings
    b = path.outputs.shape[0]
    alpha = torch.full((b, k), float("nan"))
    jump = torch.full((b, k), float("nan"))
    index = torch.full((b, k), -1, dtype=torch.long)
    for i in range(b):
        mine = ((c.example == i) & c.exact & ~c.jump.isnan()).nonzero().squeeze(-1)
        top = mine[c.jump[mine].abs().argsort(descending=True)[:k]]
        alpha[i, : len(top)], jump[i, : len(top)], index[i, : len(top)] = c.alpha[top], c.jump[top], top
    return alpha, jump, index


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

