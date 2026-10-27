"""
Spline geometry: the partition of input space and the local affine maps.

A network built from affine layers and piecewise-linear activations (ReLU,
leaky ReLU, max-pooling) is a continuous piecewise-affine spline: every
activation unit cuts its input space with a hyperplane, together they
partition the input space into regions, and inside each region the network
is one affine map (Balestriero & Baraniuk, 2018). A region is identified by
its activation code, the on/off pattern of all units.

* :func:`codes`, :func:`region_ids`, :func:`hamming`, :func:`code_agreement`:
  which region each input lies in, and how regions relate to labels.
* :func:`local_complexity`: how many unit boundaries pass near an input
  (Humayun, Balestriero & Baraniuk, 2024).
* :func:`boundary_distance`: distances to the hyperplanes of a linear layer.
* :func:`path_crossings`, :func:`slice_partition`: the partition along a line
  and on a 2-d slice through three inputs (a grid-based SplineCam, Humayun et
  al., 2023).
* :func:`jacobian_spectrum`: singular values of the local (affine) map
  between two layers.
* :func:`in_hull`: interpolation vs. extrapolation (Balestriero, Pesenti &
  LeCun, 2021).

Smooth activations (GELU, SiLU, as in ViTs and ConvNeXts) are not piecewise
linear and softmax attention and normalisation layers are not affine, so for
those models codes describe where units are mostly on or off and Jacobians
are local linearisations, not exact affine pieces (see
:func:`is_piecewise_linear`).

Typical usage::

    layers = activation_layers(model)
    c = codes(model, loader, layers)
    within, between = code_agreement(c["layer4.1.act2"], labels)
    lc = local_complexity(model, images, layers, radius=0.05)
    ids, (u, v) = slice_partition(model, img_a, img_b, img_c, layers, res=128)

References:
    Balestriero, R. & Baraniuk, R. (2018). A Spline Theory of Deep Learning.
        ICML 2018.
    Humayun, A. I., Balestriero, R., Balakrishnan, G. & Baraniuk, R. (2023).
        SplineCam: Exact Visualization and Characterization of Deep Network
        Geometry and Decision Boundaries. CVPR 2023.
    Humayun, A. I., Balestriero, R. & Baraniuk, R. (2024). Deep Networks
        Always Grok and Here is Why. ICML 2024. (Local complexity.)
    Balestriero, R., Pesenti, J. & LeCun, Y. (2021). Learning in High
        Dimension Always Amounts to Extrapolation. arXiv:2110.09485.
"""

from __future__ import annotations

import re
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    Sequence
)
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from nnsight.intervention.envoy import Envoy
from scipy.optimize import linprog
from torch import (
    Tensor,
    nn
)

from ..edits import (
    Edit,
    edit_output,
    run
)
from ..extract import (
    Layers,
    _as_list,
    default_input_fn,
    plan,
    to_device
)
from ..layouts import to_tokens
from ..models import (
    unwrap,
    wrap
)

try:  # double backward needs attention built from differentiable ops, not fused kernels
    from torch.nn.attention import (
        SDPBackend,
        sdpa_kernel
    )
    _math_attention = lambda: sdpa_kernel(SDPBackend.MATH)
except ImportError:  # torch < 2.3
    _math_attention = nullcontext

ACTIVATIONS = {"ReLU", "ReLU6", "LeakyReLU", "PReLU", "RReLU", "ELU", "CELU", "SELU", "GELU", "GELUTanh",
               "QuickGELU", "SiLU", "Swish", "Mish", "Hardswish", "HardSwish", "Softplus"}
PIECEWISE_LINEAR = {"ReLU", "ReLU6", "LeakyReLU", "PReLU", "RReLU"}
Model = nn.Module | Envoy


def activation_layers(model: Model, pattern: str | None = None) -> list[str]:
    """
    Activation modules of a model, whose inputs define the partition.

    Recognises the usual activation classes by name (PyTorch's and timm's, see
    :data:`ACTIVATIONS`); activations applied as functions inside a module's
    ``forward`` are invisible. A module called several times per forward pass
    is read at its first call only.

    Args:
        model: An nnsight model or a plain module.
        pattern: Keep only paths matching this regular expression.

    Returns:
        Module paths, in definition order.
    """
    names = [n for n, m in unwrap(model).named_modules() if n and type(m).__name__ in ACTIVATIONS]
    return [n for n in names if pattern is None or re.search(pattern, n)]


def is_piecewise_linear(model: Model) -> bool:
    """
    Whether all recognised activations are piecewise linear (ReLU family).

    Args:
        model: An nnsight model or a plain module.

    Returns:
        ``True`` if every activation layer is piecewise linear. Attention or
        normalisation layers can still make the network non-affine per region.
    """
    module = unwrap(model)
    return all(type(module.get_submodule(n)).__name__ in PIECEWISE_LINEAR for n in activation_layers(model))


def _signs(model: Envoy, batch: Any, layers: list[str], input_fn: Callable,
           edits: dict[str, Edit] | None = None) -> dict[str, Tensor]:
    edits = edits or {}
    order, layouts = plan(model, batch, list(dict.fromkeys([*layers, *edits])), input_fn=input_fn)
    last = max(order.index(n) for n in layers)
    args, kwargs = input_fn(batch)
    out: dict[str, Tensor] = {}
    with torch.no_grad(), model.trace(*args, **kwargs) as tracer:
        for name in order[: last + 1]:
            if name in layers:
                out[name] = (model.get(name).input > 0).save()  # read before in-place activations overwrite it
            if name in edits:
                edit_output(model.get(name), edits[name], layouts[name])
        tracer.stop()
    return {n: to_tokens(out[n], layouts[n]) for n in layers}


def codes(
    model: Model,
    batches: Iterable[Any],
    layers: Layers,
    edits: dict[str, Edit] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    """
    Activation codes: which units are on (pre-activation > 0), per input.

    With ``edits`` the codes are those of the edited model, e.g. under
    steering: ``hamming(codes(m, x, L), codes(m, x, L, edits={...}))`` is the
    fraction of downstream units an intervention switches.

    Args:
        model: An nnsight model or a plain module.
        batches: Any iterable of batches, e.g. a ``DataLoader``.
        layers: Activation layers, e.g. from :func:`activation_layers`.
        edits: Optional edits (see :mod:`innards.edits`) applied while reading.
        input_fn: Maps a batch to model ``(args, kwargs)``.
        device: Device the batches are moved to; defaults to the model's.

    Returns:
        Layer name -> boolean codes ``(n, tokens, units)`` (one token per
        spatial position for feature maps).
    """
    model, layers = wrap(model), _as_list(layers)
    device = device if device is not None else next(unwrap(model).parameters()).device
    parts: dict[str, list[Tensor]] = {n: [] for n in layers}
    for batch in batches:
        for n, c in _signs(model, to_device(batch, device), layers, input_fn, edits).items():
            parts[n].append(c.cpu())
    return {n: torch.cat(p) for n, p in parts.items()}


def _flat_codes(codes: Tensor | Mapping[str, Tensor] | Sequence[Tensor]) -> Tensor:
    if isinstance(codes, Tensor):
        return codes.reshape(codes.shape[0], -1)
    parts = list(codes.values()) if isinstance(codes, Mapping) else list(codes)
    return torch.cat([c.reshape(c.shape[0], -1) for c in parts], 1)


def region_ids(codes: Tensor | Mapping[str, Tensor] | Sequence[Tensor]) -> Tensor:
    """
    Region of every input: inputs with identical codes share an id.

    Args:
        codes: Codes ``(n, ...)`` of one layer, or several layers (dict or
            list), which are joined (the region of the network up to them).
            For one region per token, pass ``c.reshape(-1, c.shape[-1])``.

    Returns:
        ``(n,)`` region ids ``0 .. n_regions - 1``; ``ids.max() + 1`` regions,
        ``torch.bincount(ids)`` inputs per region.
    """
    return torch.unique(_flat_codes(codes), dim=0, return_inverse=True)[1]


def hamming(A: Tensor, B: Tensor | None = None) -> Tensor:
    """
    Fraction of units whose state differs between codes.

    Args:
        A: Codes ``(n, ...)``.
        B: Codes of the same shape; ``None`` compares all pairs within ``A``.

    Returns:
        ``(n,)`` distances between ``A[i]`` and ``B[i]``, or ``(n, n)``
        pairwise distances within ``A``.
    """
    a = _flat_codes(A).float()
    if B is not None:
        return (a != _flat_codes(B).float()).float().mean(1)
    d = a.shape[1]
    same = a @ a.T + (1 - a) @ (1 - a).T
    return 1 - same / d


def code_agreement(codes: Tensor | Mapping[str, Tensor], y: Tensor) -> tuple[float, float]:
    """
    Mean code distance within and between classes.

    A partition aligned with the labels has ``within`` well below ``between``.

    Args:
        codes: Codes ``(n, ...)`` (or several layers, joined).
        y: Labels ``(n,)``.

    Returns:
        ``(within, between)`` mean :func:`hamming` distances.
    """
    D = hamming(_flat_codes(codes))
    same = y.reshape(-1, 1).cpu() == y.reshape(1, -1).cpu()
    off = ~torch.eye(len(y), dtype=torch.bool)
    return float(D[same & off].mean()), float(D[~same].mean())


def _codes_of(model: Envoy, x: Tensor, layers: list[str], chunk: int) -> dict[str, Tensor]:
    return codes(model, [x[i : i + chunk] for i in range(0, len(x), chunk)], layers, device=x.device)


def local_complexity(
    model: Model,
    x: Tensor,
    layers: Layers,
    radius: float = 0.05,
    n_dirs: int = 8,
    seed: int = 0,
    chunk: int = 256,
) -> dict[str, Tensor]:
    """
    Local complexity: units whose boundary passes through a neighbourhood of each input.

    The neighbourhood is a cross-polytope of ``radius`` spanned by ``n_dirs``
    random orthonormal input directions (Humayun, Balestriero & Baraniuk,
    2024). A unit counts if its state is not the same on all vertices; for
    piecewise-linear networks this counts the hyperplanes cutting the
    neighbourhood up to that layer.

    Args:
        model: An nnsight model or a plain module.
        x: Inputs ``(batch, ...)``, e.g. images, on the model's device.
        layers: Activation layers.
        radius: Neighbourhood radius in input units.
        n_dirs: Dimension of the random neighbourhood.
        seed: Seed of the directions.
        chunk: Inputs per forward pass (``batch * (2 * n_dirs + 1)`` are run).

    Returns:
        Layer name -> ``(batch,)`` number of units cut.
    """
    model, layers = wrap(model), _as_list(layers)
    b, shape = x.shape[0], x.shape[1:]
    g = torch.Generator().manual_seed(seed)
    dirs = torch.linalg.qr(torch.randn(shape.numel(), n_dirs, generator=g, dtype=torch.float64)).Q.T
    dirs = torch.cat([torch.zeros(1, shape.numel(), dtype=torch.float64), dirs, -dirs]).to(x)
    points = (x.reshape(b, 1, -1) + radius * dirs).reshape(-1, *shape)
    out = {}
    for name, c in _codes_of(model, points, layers, chunk).items():
        c = c.reshape(b, len(dirs), -1)
        out[name] = (c != c[:, :1]).any(1).sum(1)
    return out


def boundary_distance(h: Tensor, W: Tensor, b: Tensor | None = None) -> Tensor:
    """
    Distances to the hyperplanes ``{z : W_i z + b_i = 0}`` of a linear layer.

    Exact distances to the unit boundaries of the layer following ``h`` (in
    the space of ``h``), e.g. ``h`` read at ``blocks.3.norm2`` with ``W, b``
    of ``blocks.3.mlp.fc1``.

    Args:
        h: Inputs of the linear layer ``(..., d)``.
        W: Weight ``(m, d)``.
        b: Bias ``(m,)``.

    Returns:
        ``(..., m)`` distances.
    """
    W = W.detach().to(h)
    pre = h @ W.T + (b.detach().to(h) if b is not None else 0)
    return pre.abs() / W.norm(dim=1)


def path_crossings(
    model: Model,
    x0: Tensor,
    x1: Tensor,
    layers: Layers,
    steps: int = 64,
    chunk: int = 256,
) -> dict[str, Tensor]:
    """
    Unit flips along straight lines between inputs.

    The number of units changing state between consecutive points of
    ``(1 - s) x0 + s x1``: a lower bound on the region boundaries crossed (two
    flips between the same pair of points count once per unit).

    Args:
        model: An nnsight model or a plain module.
        x0: Start inputs ``(batch, ...)``.
        x1: End inputs ``(batch, ...)``.
        layers: Activation layers.
        steps: Points per line, including both ends.
        chunk: Inputs per forward pass.

    Returns:
        Layer name -> ``(batch, steps - 1)`` flips per segment; ``.sum(1)``
        is the total per line.
    """
    model, layers = wrap(model), _as_list(layers)
    s = torch.linspace(0, 1, steps, dtype=x0.dtype, device=x0.device).reshape(1, steps, *[1] * (x0.ndim - 1))
    points = ((1 - s) * x0.unsqueeze(1) + s * x1.unsqueeze(1)).reshape(-1, *x0.shape[1:])
    out = {}
    for name, c in _codes_of(model, points, layers, chunk).items():
        c = c.reshape(x0.shape[0], steps, -1)
        out[name] = (c[:, 1:] != c[:, :-1]).sum(-1)
    return out


def slice_partition(
    model: Model,
    x0: Tensor,
    x1: Tensor,
    x2: Tensor,
    layers: Layers,
    res: int = 64,
    extent: tuple[float, float] = (-0.25, 1.25),
    chunk: int = 256,
) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor]]:
    """
    Regions on the 2-d plane through three inputs (a grid-based SplineCam).

    The plane is ``x0 + u (x1 - x0) + v (x2 - x0)``; the inputs sit at
    ``(u, v) = (0, 0), (1, 0), (0, 1)``. Regions are sampled on a grid, so
    regions smaller than a grid cell can be missed (SplineCam computes them
    exactly for ReLU networks).

    Args:
        model: An nnsight model or a plain module.
        x0: Input (one example, no batch dimension), e.g. an image ``(3, H, W)``.
        x1: Input of the same shape.
        x2: Input of the same shape.
        layers: Activation layers.
        res: Grid points per axis.
        extent: Range of ``u`` and ``v``.
        chunk: Inputs per forward pass.

    Returns:
        ``(ids, (u, v))``: layer name -> ``(res, res)`` region ids of the
        network up to and including that layer (in execution order), and the
        ``(res, res)`` grid coordinates. Plot with ``plt.pcolormesh(u, v,
        ids[name] % 20, cmap="tab20")`` or draw :func:`boundaries`.
    """
    model, layers = wrap(model), _as_list(layers)
    t = torch.linspace(*extent, res, dtype=x0.dtype, device=x0.device)
    u, v = torch.meshgrid(t, t, indexing="xy")
    shape = (-1, *[1] * x0.ndim)
    points = x0 + u.reshape(shape) * (x1 - x0) + v.reshape(shape) * (x2 - x0)
    c = _codes_of(model, points, layers, chunk)
    order = plan(model, points[:1], layers)[0]
    ids, joined = {}, []
    for name in order:
        joined.append(c[name])
        ids[name] = region_ids(joined).reshape(res, res)
    return ids, (u.cpu(), v.cpu())


def boundaries(ids: Tensor) -> Tensor:
    """
    Grid cells next to a region boundary.

    Args:
        ids: Region ids ``(h, w)``, e.g. from :func:`slice_partition`.

    Returns:
        Boolean ``(h, w)`` mask of cells whose right or lower neighbour lies
        in another region.
    """
    edge = torch.zeros_like(ids, dtype=torch.bool)
    edge[:, :-1] |= ids[:, 1:] != ids[:, :-1]
    edge[:-1] |= ids[1:] != ids[:-1]
    return edge


def jacobian_spectrum(
    model: Model,
    x: Tensor,
    dst: str,
    src: str | None = None,
    k: int = 10,
    oversample: int = 5,
    n_iter: int = 2,
    layouts: str | dict[str, str] | None = None,
    seed: int = 0,
) -> Tensor:
    """
    Top singular values of the Jacobian from ``src`` to ``dst``, per input.

    The local linear (for piecewise-linear networks: the region's affine) map
    between two layers; its singular values measure how much the network
    stretches each direction there (the largest is the local Lipschitz
    constant). Computed with randomized SVD (Halko et al., 2011) from
    Jacobian-vector products, so the Jacobian is never formed; these need
    double backward, so ``scaled_dot_product_attention`` runs on its math
    backend here. Assumes the examples of a batch do not interact (eval mode).

    Args:
        model: An nnsight model or a plain module.
        x: Inputs ``(batch, ...)`` on the model's device.
        dst: Output layer.
        src: Input layer (must run before ``dst``); ``None`` for the model input.
        k: Number of singular values.
        oversample: Extra random directions for accuracy.
        n_iter: Power iterations for accuracy.
        layouts: Layouts of the layers.
        seed: Seed of the random directions.

    Returns:
        ``(batch, k)`` singular values, descending.
    """
    model = wrap(model)
    store: dict[str, Tensor] = {}

    def make_leaf(tokens: Tensor) -> Tensor:
        store["leaf"] = tokens.detach().requires_grad_(True)
        return store["leaf"]

    with torch.enable_grad(), _math_attention():
        if src is None:
            store["leaf"] = x.detach().requires_grad_(True)
            _, reads = run(model, store["leaf"], read=dst, layouts=layouts)
        else:
            _, reads = run(model, x, {src: make_leaf}, read=dst, layouts=layouts)
        leaf, y = store["leaf"], reads[dst]
        b = leaf.shape[0]
        m, p = leaf[0].numel(), y[0].numel()
        s = min(k + oversample, m, p)
        dummy = torch.zeros_like(y, requires_grad=True)
        g = torch.autograd.grad(y, leaf, dummy, create_graph=True)[0]  # J^T dummy, linear in dummy

        def jvp(vec: Tensor) -> Tensor:  # (b, m) -> (b, p)
            return torch.autograd.grad(g, dummy, vec.reshape(leaf.shape), retain_graph=True)[0].reshape(b, p)

        def vjp(vec: Tensor) -> Tensor:  # (b, p) -> (b, m)
            return torch.autograd.grad(y, leaf, vec.reshape(y.shape), retain_graph=True)[0].reshape(b, m)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        omega = torch.randn(b, m, s, generator=gen).to(leaf)
        Y = torch.stack([jvp(omega[..., i]) for i in range(s)], -1)
        for _ in range(n_iter):
            Q = torch.linalg.qr(Y).Q
            Z = torch.linalg.qr(torch.stack([vjp(Q[..., i]) for i in range(s)], -1)).Q
            Y = torch.stack([jvp(Z[..., i]) for i in range(s)], -1)
        Q = torch.linalg.qr(Y).Q
        Bt = torch.stack([vjp(Q[..., i]) for i in range(s)], -1)  # J^T Q, (b, m, s)
    return torch.linalg.svdvals(Bt.detach())[:, :k]


def in_hull(X: Tensor, points: Tensor) -> Tensor:
    """
    Whether points lie in the convex hull of ``X`` (one linear program each, scipy).

    In high dimension new samples almost never fall inside the training
    data's hull, so generalisation is extrapolation (Balestriero, Pesenti &
    LeCun, 2021). Cost grows with ``n``; subsample ``X`` for large sets.

    Args:
        X: Reference points ``(n, d)``, e.g. training representations.
        points: Query points ``(m, d)``.

    Returns:
        Boolean ``(m,)``.
    """
    A = X.reshape(X.shape[0], -1).double().cpu().numpy()
    A_eq = np.vstack([A.T, np.ones((1, A.shape[0]))])
    out = []
    for p in points.reshape(points.shape[0], -1).double().cpu().numpy():
        res = linprog(np.zeros(A.shape[0]), A_eq=A_eq, b_eq=np.append(p, 1.0), bounds=(0, None), method="highs")
        out.append(res.status == 0)
    return torch.tensor(out)
