"""
Interpretation: what a direction responds to, where, and what it does.

* :func:`top_k`: the most activating examples (and positions) of a direction.
* :func:`spatial_map`: where in each image directions fire.
* :func:`input_attribution`: pixel attribution of a direction (gradient x
  input, integrated gradients).
* :func:`readout_effect`, :func:`logit_lens`: what directions and
  intermediate tokens do to the logits.
* :func:`attention_maps`, :func:`attention_rollout`: ViT attention.

Directions are ``v (d,)`` or ``V (k, d)``, e.g. from a probe, PCA,
``relevant_subspace`` or an SAE decoder (``sae.decoder_weight()``).

Typical usage::

    acts, labels = load_acts("acts/run")
    values, examples, positions = top_k(acts["blocks.6"], v, k=16)
    grid = load_info("acts/run")["layers"]["blocks.6"]["grid"] or (14, 14)
    maps = spatial_map(acts["blocks.6"], V, grid, exclude=(0,), size=(224, 224))
    maps = attention_maps(model, images)            # {"blocks.0.attn": (b, heads, t, t), ...}
    cls_map = attention_rollout(maps)[:, 0, 1:]

References:
    Simonyan, K., Vedaldi, A. & Zisserman, A. (2013). Deep Inside
        Convolutional Networks. arXiv:1312.6034. (Gradient saliency.)
    Sundararajan, M., Taly, A. & Yan, Q. (2017). Axiomatic Attribution for
        Deep Networks. ICML 2017. (Integrated gradients.)
    nostalgebraist (2020). interpreting GPT: the logit lens. LessWrong.
    Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow in
        Transformers. ACL 2020. (Attention rollout.)
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence
)

import torch
import torch.nn.functional as F
from einops import rearrange
from nnsight.intervention.envoy import Envoy
from torch import (
    Tensor,
    nn
)

from ..extract import (
    _as_list,
    capture,
    plan
)
from ..layouts import (
    keep_positions,
    to_tokens
)
from ..models import (
    unwrap,
    wrap
)
from .geometry import _2d

Model = nn.Module | Envoy


def top_k(X: Tensor, v: Tensor, k: int = 10, largest: bool = True) -> tuple[Tensor, ...]:
    """
    Examples (and positions) where a direction is most active.

    Args:
        X: Representation ``(n, d)`` or tokens ``(n, t, d)``.
        v: Direction ``(d,)``; scores are ``X @ v``.
        k: Number of results.
        largest: Most positive (``True``) or most negative scores.

    Returns:
        ``(values, examples)`` for ``(n, d)`` inputs, ``(values, examples,
        positions)`` for tokens; each ``(k,)``, best first.
    """
    scores = X.double() @ v.double().to(X.device)
    values, idx = scores.reshape(-1).topk(min(k, scores.numel()), largest=largest)
    if scores.ndim == 1:
        return values.cpu(), idx.cpu()
    return values.cpu(), (idx // scores.shape[1]).cpu(), (idx % scores.shape[1]).cpu()


def spatial_map(
    tokens: Tensor,
    V: Tensor,
    grid: tuple[int, int],
    exclude: Sequence[int] = (),
    size: tuple[int, int] | None = None,
) -> Tensor:
    """
    Activation of directions over the spatial grid of each image.

    Works for ViT patch tokens and CNN feature maps alike (both are tokens,
    row-major over the grid).

    Args:
        tokens: Tokens ``(n, t, d)``, e.g. from ``extract`` without reduction.
        V: Direction ``(d,)`` or directions ``(k, d)``.
        grid: Spatial ``(h, w)`` with ``h * w`` the number of kept tokens,
            e.g. ``load_info(run)["layers"][layer]["grid"]``.
        exclude: Non-spatial tokens to drop, e.g. ``range(model.num_prefix_tokens)``.
        size: Upsample the maps to ``(H, W)`` (bilinear), e.g. the image size.

    Returns:
        ``(n, h, w)`` for one direction, ``(n, k, h, w)`` for several.
    """
    tokens = tokens[:, keep_positions(tokens.shape[1], list(exclude), tokens.device)]
    maps = rearrange(tokens.double() @ _2d(V).to(tokens.device).T, "n (h w) k -> n k h w", h=grid[0], w=grid[1])
    if size is not None:
        maps = F.interpolate(maps, size=size, mode="bilinear", align_corners=False)
    return maps[:, 0] if V.ndim == 1 else maps


def input_attribution(
    model: Model,
    x: Tensor,
    layer: str,
    v: Tensor,
    method: str = "grad_x_input",
    exclude: Sequence[int] = (),
    steps: int = 32,
    baseline: Tensor | None = None,
    layouts: str | dict[str, str] | None = None,
) -> Tensor:
    """
    Input attribution of a direction's activation at a layer.

    The attributed score is ``sum over kept tokens of (h . v)``.
    ``"grad_x_input"`` is gradient times input (Simonyan et al., 2013);
    ``"integrated_gradients"`` averages the gradient along the straight path
    from ``baseline`` (Sundararajan et al., 2017), so the attributions sum to
    ``score(x) - score(baseline)``.

    Args:
        model: An nnsight model or a plain module.
        x: Inputs ``(batch, ...)``, e.g. images, on the model's device.
        layer: The layer.
        v: Direction ``(d,)``.
        method: ``"grad_x_input"`` or ``"integrated_gradients"``.
        exclude: Token positions left out of the score, e.g. ``(0,)``.
        steps: Path points for integrated gradients.
        baseline: Integrated-gradients baseline, shaped like ``x``; zeros by default.
        layouts: Layout of the layer.

    Returns:
        Attributions shaped like ``x``.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method not in ("grad_x_input", "integrated_gradients"):
        raise ValueError(f"unknown method {method!r}")
    model = wrap(model)
    layout = plan(model, x[:1], [layer], layouts)[1][layer]
    v = v.detach().to(x)

    def grad(inputs: Tensor) -> Tensor:
        inputs = inputs.detach().requires_grad_(True)
        with torch.enable_grad():
            tokens = to_tokens(capture(model, inputs, layer)[layer], layout)
            tokens = tokens[:, keep_positions(tokens.shape[1], list(exclude), tokens.device)]
            return torch.autograd.grad((tokens @ v).sum(), inputs)[0]

    if method == "grad_x_input":
        return (grad(x) * x).detach()
    baseline = torch.zeros_like(x) if baseline is None else baseline.to(x)
    total = torch.zeros_like(x)
    for alpha in (torch.arange(steps, dtype=x.dtype) + 0.5) / steps:  # midpoint rule
        total += grad(baseline + alpha * (x - baseline))
    return ((x - baseline) * total / steps).detach()


def readout_effect(V: Tensor, W: Tensor) -> Tensor:
    """
    Direct effect of directions on the logits of a linear readout.

    Exact for the representation the readout reads (e.g. the pooled features
    before the head); ignores normalisation in between.

    Args:
        V: Direction ``(d,)`` or directions ``(k, d)``.
        W: Readout weight ``(classes, d)``, e.g. ``model.head.weight``.

    Returns:
        ``(k, classes)`` logit change per unit along each direction;
        ``.topk(5)`` gives the classes a direction promotes.
    """
    return _2d(V) @ W.detach().double().T.to(V.device)


def logit_lens(model: Model, tokens: Tensor) -> Tensor:
    """
    Logits of intermediate tokens, read out by the final norm and head.

    For timm models with a ``forward_head`` whose input is the token sequence
    (ViT family): ``forward_head(norm(tokens))``. At the last block this is
    the model output; earlier it shows what the residual stream already
    encodes about the classes (nostalgebraist, 2020).

    Args:
        model: A timm ViT-style model (plain or nnsight).
        tokens: Tokens ``(n, t, d)`` of any block.

    Returns:
        Logits ``(n, classes)``.

    Raises:
        ValueError: If the model has no ``forward_head``.
    """
    module = unwrap(model)
    if not hasattr(module, "forward_head"):
        raise ValueError("logit_lens needs a timm model with forward_head")
    norm = getattr(module, "norm", nn.Identity())
    with torch.no_grad():
        return module.forward_head(norm(tokens.to(next(module.parameters()))))


def attention_modules(model: Model) -> list[str]:
    """
    timm attention modules (those with a ``fused_attn`` switch and ``attn_drop``).

    Args:
        model: An nnsight model or a plain module.

    Returns:
        Module paths, e.g. ``["blocks.0.attn", ...]``.
    """
    return [n for n, m in unwrap(model).named_modules()
            if n and hasattr(m, "fused_attn") and isinstance(getattr(m, "attn_drop", None), nn.Module)]


def attention_maps(model: Model, x: Tensor, layers: str | Sequence[str] | None = None) -> dict[str, Tensor]:
    """
    Attention probabilities of timm attention modules.

    Fused attention (``scaled_dot_product_attention``) never forms the
    attention matrix, so it is switched off for this call (and restored
    after); the probabilities are read where they enter ``attn_drop``.

    Args:
        model: A timm model (plain or nnsight).
        x: Inputs ``(batch, ...)`` on the model's device.
        layers: Attention modules (see :func:`attention_modules`); ``None``
            reads all.

    Returns:
        Module path -> ``(batch, heads, tokens, tokens)`` attention, rows
        summing to 1.
    """
    model = wrap(model)
    names = attention_modules(model) if layers is None else _as_list(layers)
    modules = [unwrap(model).get_submodule(n) for n in names]
    flags = [m.fused_attn for m in modules]
    try:
        for m in modules:
            m.fused_attn = False
        drops = [f"{n}.attn_drop" for n in names]
        order = plan(model, x, drops, layouts="bchw")[0]  # layouts are unused here; skips inference
        saved: dict[str, Tensor] = {}
        with torch.no_grad(), model.trace(x) as tracer:
            for name in order:
                saved[name] = model.get(name).input.save()
            tracer.stop()
    finally:
        for m, flag in zip(modules, flags):
            m.fused_attn = flag
    return {n: saved[d] for n, d in zip(names, drops)}


def attention_rollout(maps: Mapping[str, Tensor] | Sequence[Tensor], residual: bool = True,
                      head_fusion: str = "mean") -> Tensor:
    """
    Attention rollout (Abnar & Zuidema, 2020): attention flow through all layers.

    Heads are fused per layer, the residual connection is modelled as
    ``0.5 A + 0.5 I`` (renormalised), and the layers are multiplied in order.

    Args:
        maps: Attention per layer in execution order, e.g. from
            :func:`attention_maps`.
        residual: Model the residual connection.
        head_fusion: ``"mean"``, ``"max"`` or ``"min"`` over heads.

    Returns:
        ``(batch, tokens, tokens)``; row 0 is the CLS token's flow to every
        token, e.g. ``rollout[:, 0, 1:]`` as a spatial map.

    Raises:
        ValueError: If ``head_fusion`` is unknown.
    """
    fuse = {"mean": lambda a: a.mean(1), "max": lambda a: a.amax(1), "min": lambda a: a.amin(1)}
    if head_fusion not in fuse:
        raise ValueError(f"head_fusion must be one of {list(fuse)}, got {head_fusion!r}")
    rollout = None
    for a in (maps.values() if isinstance(maps, Mapping) else maps):
        a = fuse[head_fusion](a.double())
        if residual:
            a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
            a = a / a.sum(-1, keepdim=True)
        rollout = a if rollout is None else a @ rollout
    return rollout
