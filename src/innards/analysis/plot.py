"""
Plots for the analysis tools (needs the ``viz`` extra).

Every function draws in the innards style (:mod:`innards.style`): print
sizes, hairline axes, thin marks, the validated palette, text in ink colors.
Functions take an optional ``ax`` (a new figure is made when it is ``None``)
and return it, so they compose into multi-panel figures; image grids return
the figure. Values given per example (``(n,)`` or ``(n, k)`` tensors, as the
causal tools return) are drawn as the mean with a 95% bootstrap interval.
:func:`save` writes a figure with the style's output settings.

Typical usage::

    from innards.analysis import plot

    ax = plot.layer_curve({"probe": probe_layers(acts, y), "chance": {l: 0.1 for l in acts}}, ylabel="accuracy")
    plot.similarity_matrix(cka_matrix(acts), labels=list(acts))
    plot.sweep(sweep(model, loader, lambda a: {"blocks.6": steer(v, a)}, alphas, kl()), alphas, xlabel="alpha")
    plot.patching(patch_sweep(model, clean, corrupt, layers, metric, positions=True), grid=(14, 14), exclude=(0,))
    fig = plot.images(images, maps=spatial_map(tokens, v, (14, 14), exclude=(0,)))
    plot.save(fig, "figures/maps.pdf")
"""

from __future__ import annotations

import functools
from collections.abc import (
    Callable,
    Mapping,
    Sequence
)
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.axes import Axes
from matplotlib import patheffects
from matplotlib.colors import (
    ListedColormap,
    to_rgba
)
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor

from .. import style
from ..style import (
    INK,
    color
)
from .metrics import bootstrap_ci

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def styled(fn: Callable) -> Callable:
    """Decorator running a plotting function inside the innards style."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with style.context():
            return fn(*args, **kwargs)

    return wrapper


def _np(x) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().double().cpu().numpy()
    return np.asarray(x, dtype=float)


def _ax(ax: Axes | None, figsize: tuple[float, float] | None = None) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _summary(values) -> tuple[float, float, float]:
    """Mean and 95% bootstrap interval of per-example values; scalars have no interval."""
    v = _np(values).reshape(-1)
    if v.size == 1:
        return float(v[0]), float("nan"), float("nan")
    return bootstrap_ci(torch.from_numpy(v), n_resamples=1000)


def _line(ax: Axes, x: np.ndarray, values: Sequence, label: str | None, c: str) -> None:
    stats = np.array([_summary(v) for v in values])
    mean, lo, hi = stats.T
    if np.isfinite(lo).any():
        ax.fill_between(x, lo, hi, color=c, alpha=0.12, linewidth=0)
    ax.plot(x, mean, color=c, label=label, marker="o", markeredgecolor=INK["surface"], markeredgewidth=0.8)


def _is_series_dict(values: Mapping) -> bool:
    return all(isinstance(v, Mapping) for v in values.values())


def _tick_labels(ax: Axes, names: Sequence[str], axis: str = "x") -> None:
    rotate = sum(len(n) for n in names) > 40 or max(len(n) for n in names) > 10  # only when labels would collide
    pos = np.arange(len(names))
    if axis == "x":
        ax.set_xticks(pos, names, rotation=45 if rotate else 0, ha="right" if rotate else "center",
                      rotation_mode="anchor")
    else:
        ax.set_yticks(pos, names)


def _legend(ax: Axes, n_series: int) -> None:
    if n_series >= 2:
        ax.legend(loc="best")


@styled
def save(fig: Figure, path: str | Path, **kwargs) -> Path:
    """
    Saves a figure with the style's output settings (tight bounding box, 300 dpi, embedded fonts).

    Args:
        fig: The figure, e.g. ``ax.figure``.
        path: Output path; the suffix picks the format (``.pdf``, ``.svg``, ``.png``).
        **kwargs: Passed to ``Figure.savefig``.

    Returns:
        The path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **kwargs)
    return path


@styled
def layer_curve(
    values: Mapping[str, float | Tensor] | Mapping[str, Mapping[str, float | Tensor]],
    ax: Axes | None = None,
    ylabel: str = "",
    xlabel: str = "layer",
    reference: float | None = None,
    reference_label: str = "",
    title: str = "",
) -> Axes:
    """
    A quantity across layers, e.g. probe accuracy, intrinsic dimension, effect size.

    Args:
        values: Layer -> value for one series, or series name -> (layer ->
            value) for several (one color each, in palette order, with a
            legend). A value is a number or per-example values (drawn as the
            mean with a 95% bootstrap band).
        ax: Axes to draw on.
        ylabel: y-axis label.
        xlabel: x-axis label.
        reference: Optional horizontal reference, e.g. chance level.
        reference_label: Label written at the reference line.
        title: Axes title.

    Returns:
        The axes.
    """
    ax = _ax(ax)
    series = values if _is_series_dict(values) else {"": values}
    layers = list(dict.fromkeys(layer for s in series.values() for layer in s))
    x = np.arange(len(layers))
    for i, (name, s) in enumerate(series.items()):
        idx = np.array([layers.index(layer) for layer in s])
        _line(ax, x[idx], list(s.values()), name or None, color(i))
    if reference is not None:
        ax.axhline(reference, color=INK["axis"], linewidth=0.8, zorder=0)
        if reference_label:
            ax.annotate(reference_label, (x[-1], reference), xytext=(0, 2), textcoords="offset points",
                        ha="right", va="bottom", color=INK["muted"], fontsize=6.5)
    _tick_labels(ax, layers)
    ax.grid(axis="y")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _legend(ax, len(series))
    return ax


@styled
def similarity_matrix(
    matrix: Tensor | np.ndarray,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    signed: bool = False,
    cbar_label: str = "CKA",
    title: str = "",
) -> Axes:
    """
    Heatmap of a square similarity matrix, e.g. CKA between layers.

    Args:
        matrix: ``(L, L)`` values, e.g. from ``cka_matrix``.
        labels: Row / column names.
        ax: Axes to draw on.
        vmin: Lower end of the color scale (``None``: data minimum).
        vmax: Upper end of the color scale (``None``: data maximum).
        signed: Diverging scale centred on 0 (e.g. for cosine similarities).
        cbar_label: Color-bar label.
        title: Axes title.

    Returns:
        The axes.
    """
    ax = _ax(ax, figsize=(3.5, 3.0))
    m = _np(matrix)
    if signed:
        lim = float(np.abs(m).max()) if vmax is None else vmax
        vmin, vmax, cmap = -lim, lim, "innards_div"
    else:
        cmap = "innards_seq"
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.8, aspect=25, label=cbar_label)
    cb.outline.set_visible(False)
    if labels is not None:
        _tick_labels(ax, list(labels), "x")
        _tick_labels(ax, list(labels), "y")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title)
    return ax


@styled
def spectrum(
    X: Tensor | Mapping[str, Tensor],
    ax: Axes | None = None,
    fit: tuple[int, int] | None = None,
    eigenvalues: bool = False,
    title: str = "",
) -> Axes:
    """
    Covariance eigenspectrum on log-log axes, with its power-law exponent.

    Args:
        X: Representation ``(n, d)``, or name -> representation for several
            (e.g. layers); with ``eigenvalues`` the values themselves.
        ax: Axes to draw on.
        fit: Index range ``(first, last)`` (1-based) of the power-law fit, as
            in ``spectrum_decay``; ``None`` fits the first half.
        eigenvalues: ``X`` already holds eigenvalues.
        title: Axes title.

    Returns:
        The axes.
    """
    from .geometry import eigenspectrum

    ax = _ax(ax)
    series = X if isinstance(X, Mapping) else {"": X}
    for i, (name, rep) in enumerate(series.items()):
        lam = _np(rep if eigenvalues else eigenspectrum(rep))
        lam = lam / lam.sum()
        rank = np.arange(1, len(lam) + 1)
        lo, hi = fit or (1, max(len(lam) // 2, 2))
        keep = slice(lo - 1, hi)
        slope, intercept = np.polyfit(np.log(rank[keep]), np.log(np.clip(lam[keep], 1e-300, None)), 1)
        label = f"{name}  α = {-slope:.2f}".strip()
        ax.loglog(rank[keep], np.exp(intercept) * rank[keep] ** slope, color=INK["grid"], linewidth=5,
                  solid_capstyle="butt", zorder=1)  # the fit, as a halo behind the series
        ax.loglog(rank, lam, color=color(i), label=label, zorder=2)
    ax.grid(axis="both", which="major")
    ax.set_xlabel("rank")
    ax.set_ylabel("variance fraction")
    ax.set_title(title)
    ax.legend(loc="lower left")
    return ax


@styled
def sweep(
    values: Tensor | Mapping[str, Tensor],
    xs: Sequence[float],
    ax: Axes | None = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
) -> Axes:
    """
    Effects across a family of interventions, e.g. steering strengths.

    Args:
        values: ``(n, len(xs))`` per-example values from ``causal.sweep``, or
            name -> such values for several sweeps.
        xs: The swept values.
        ax: Axes to draw on.
        xlabel: x-axis label.
        ylabel: y-axis label (the metric).
        title: Axes title.

    Returns:
        The axes.
    """
    ax = _ax(ax)
    series = values if isinstance(values, Mapping) else {"": values}
    x = _np(xs)
    lows = []
    for i, (name, v) in enumerate(series.items()):
        v = _np(v)
        _line(ax, x, [v[:, j] for j in range(v.shape[1])], name or None, color(i))
        lows.append(v.mean(0).min())
    if min(lows) < 0:
        ax.axhline(0, color=INK["axis"], linewidth=0.8, zorder=0)
    ax.grid(axis="y")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _legend(ax, len(series))
    return ax


@styled
def patching(
    results: Mapping[str, Tensor],
    ax: Axes | None = None,
    grid: tuple[int, int] | None = None,
    exclude: Sequence[int] = (),
    label: str = "effect",
    title: str = "",
) -> Axes | Figure:
    """
    Activation-patching (or attribution-patching) results.

    * ``(batch,)`` per layer: mean effect per layer with 95% intervals.
    * ``(batch, tokens)`` per layer: heatmap of layer x position (mean over
      the batch), or with ``grid`` one spatial map per layer (small multiples).

    Args:
        results: Layer -> values, from ``patch_sweep`` or ``attribution_patching``.
        ax: Axes to draw on (not used for small multiples).
        grid: Spatial ``(h, w)`` of the patch tokens; draws maps.
        exclude: Non-spatial positions to drop for maps, e.g. ``(0,)``.
        label: Color-bar / axis label.
        title: Title.

    Returns:
        The axes, or the figure for small multiples.
    """
    layers = list(results)
    per_position = _np(next(iter(results.values()))).ndim == 2
    if not per_position:
        return layer_curve.__wrapped__({n: results[n] for n in layers}, ax=ax, ylabel=label, title=title)
    means = [_np(results[n]).mean(0) for n in layers]
    lim = max(float(np.abs(m).max()) for m in means) or 1.0
    if grid is not None:
        maps = []
        for m in means:
            keep = np.setdiff1d(np.arange(len(m)), np.asarray(exclude, dtype=int) % len(m))
            maps.append(m[keep].reshape(grid))
        return _map_grid(dict(zip(layers, maps)), -lim, lim, "innards_div", label, title)
    ax = _ax(ax, figsize=(3.5, 0.35 * len(layers) + 0.9))
    width = max(len(m) for m in means)
    heat = np.full((len(layers), width), np.nan)
    for i, m in enumerate(means):
        heat[i, : len(m)] = m
    im = ax.imshow(heat, aspect="auto", cmap="innards_div", vmin=-lim, vmax=lim, interpolation="nearest")
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9, aspect=20, label=label)
    cb.outline.set_visible(False)
    _tick_labels(ax, layers, "y")
    ax.set_xlabel("position")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    return ax


def _map_grid(maps: Mapping[str, np.ndarray], vmin: float, vmax: float, cmap: str, label: str,
              title: str, ncols: int = 4) -> Figure:
    n = len(maps)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(style.width(2) if ncols > 2 else style.width(1),
                                                    1.6 * nrows + 0.3), squeeze=False)
    im = None
    for ax, (name, m) in zip(axes.flat, maps.items()):
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name, fontweight="normal", color=INK["secondary"])
        ax.axis("off")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, aspect=30, label=label)
    cb.outline.set_visible(False)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=8, fontweight="semibold")
    return fig


@styled
def embedding(
    X: Tensor,
    y: Tensor | None = None,
    ax: Axes | None = None,
    method: str = "pca",
    highlight: Sequence | None = None,
    class_names: Mapping | Sequence[str] | None = None,
    title: str = "",
    seed: int = 0,
) -> Axes:
    """
    2-D embedding of a representation, colored by class.

    At most three classes get a color (more cannot be told apart reliably in
    a scatter); the others are drawn in gray as context.

    Args:
        X: Representation ``(n, d)``.
        y: Labels ``(n,)``.
        ax: Axes to draw on.
        method: ``"pca"`` or ``"tsne"`` (scikit-learn).
        highlight: Up to three classes to color; ``None`` takes the first three.
        class_names: Names of the classes (by label value or index).
        title: Axes title.
        seed: Seed for t-SNE.

    Returns:
        The axes.

    Raises:
        ValueError: For an unknown method or more than three highlighted classes.
    """
    Z = _np(X).reshape(len(X), -1)
    if method == "pca":
        from .geometry import pca

        V, ratio = pca(torch.from_numpy(Z), k=2)
        P = (Z - Z.mean(0)) @ _np(V).T
        labels = [f"PC1 ({100 * float(ratio[0]):.0f}%)", f"PC2 ({100 * float(ratio[1]):.0f}%)"]
    elif method == "tsne":
        from sklearn.manifold import TSNE

        P = TSNE(n_components=2, random_state=seed, init="pca").fit_transform(Z)
        labels = ["t-SNE 1", "t-SNE 2"]
    else:
        raise ValueError(f"method must be 'pca' or 'tsne', got {method!r}")
    ax = _ax(ax, figsize=(3.2, 3.0))
    if y is None:
        ax.scatter(P[:, 0], P[:, 1], s=6, color=color(0), alpha=0.8, linewidths=0)
    else:
        yy = _np(y).reshape(-1)
        classes = list(np.unique(yy))
        highlight = classes[:3] if highlight is None else list(highlight)
        if len(highlight) > 3:
            raise ValueError("color at most three classes; pass the ones the figure is about")
        rest = ~np.isin(yy, highlight)
        ax.scatter(P[rest, 0], P[rest, 1], s=4, color=INK["context"], linewidths=0,
                   label="other" if rest.any() else None)
        for i, c in enumerate(highlight):
            sel = yy == c
            name = (class_names[int(c)] if class_names is not None else str(int(c)))
            ax.scatter(P[sel, 0], P[sel, 1], s=7, color=color(i), linewidths=0.3, edgecolors=INK["surface"],
                       label=name)
        ax.legend(loc="best", markerscale=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    return ax


@styled
def distributions(
    groups: Mapping[str, Tensor],
    ax: Axes | None = None,
    bins: int = 30,
    xlabel: str = "",
    density: bool = True,
    title: str = "",
) -> Axes:
    """
    Distributions of a per-example quantity across groups, e.g. local
    complexity of correct vs. misclassified inputs.

    Args:
        groups: Name -> values; one outlined histogram each, in palette order.
        ax: Axes to draw on.
        bins: Number of bins (shared by all groups).
        xlabel: x-axis label.
        density: Normalise each histogram.
        title: Axes title.

    Returns:
        The axes.
    """
    ax = _ax(ax)
    data = {k: _np(v).reshape(-1) for k, v in groups.items()}
    allv = np.concatenate(list(data.values()))
    if np.all(allv == np.round(allv)):  # integer data (counts): bins centred on integers, no empty gaps
        step = max(1, int(np.ceil((allv.max() - allv.min() + 1) / bins)))
        edges = np.arange(allv.min() - 0.5, allv.max() + step + 0.5, step)
    else:
        edges = np.histogram_bin_edges(allv, bins=bins)
    for i, (name, v) in enumerate(data.items()):
        ax.hist(v, bins=edges, density=density, histtype="stepfilled", color=color(i), alpha=0.12)
        ax.hist(v, bins=edges, density=density, histtype="step", color=color(i), linewidth=1.2,
                label=f"{name} (median {np.median(v):.3g})")
    ax.set_ylim(top=ax.get_ylim()[1] * (1 + 0.12 * (len(data) + 1)))  # headroom for the legend
    ax.grid(axis="y")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density" if density else "count")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return ax


@styled
def scatter(
    x: Tensor | np.ndarray,
    y: Tensor | np.ndarray,
    ax: Axes | None = None,
    xlabel: str = "",
    ylabel: str = "",
    annotate: Sequence[str] | None = None,
    n_annotate: int = 5,
    title: str = "",
) -> Axes:
    """
    One quantity against another, e.g. variance explained vs. causal
    attribution of directions.

    Args:
        x: Values ``(k,)``.
        y: Values ``(k,)``.
        ax: Axes to draw on.
        xlabel: x-axis label.
        ylabel: y-axis label.
        annotate: Point names; the ``n_annotate`` points with the largest
            ``|y|`` are labelled.
        n_annotate: Number of labelled points.
        title: Axes title.

    Returns:
        The axes.
    """
    ax = _ax(ax)
    xv, yv = _np(x).reshape(-1), _np(y).reshape(-1)
    ax.scatter(xv, yv, s=14, color=color(0), linewidths=0.8, edgecolors=INK["surface"], zorder=3)
    if annotate is not None:
        for i in np.argsort(-np.abs(yv))[:n_annotate]:
            ax.annotate(annotate[i], (xv[i], yv[i]), xytext=(3, 3), textcoords="offset points",
                        fontsize=6.5, color=INK["secondary"])
    if (yv < 0).any() and (yv > 0).any():
        ax.axhline(0, color=INK["axis"], linewidth=0.8, zorder=0)
    ax.grid(axis="both")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


@styled
def partition(
    ids: Tensor,
    u: Tensor,
    v: Tensor,
    ax: Axes | None = None,
    fill: Tensor | None = None,
    fill_label: str = "",
    anchors: Sequence[str] = ("x₀", "x₁", "x₂"),
    title: str = "",
) -> Axes:
    """
    Region boundaries on a 2-d input slice, from ``spline.slice_partition``.

    Boundaries are drawn as thin ink lines; the three inputs spanning the
    slice are marked. An optional ``fill`` (e.g. the predicted-class
    probability on the grid) is shown underneath on the sequential scale.

    Args:
        ids: ``(res, res)`` region ids of one layer.
        u: ``(res, res)`` slice coordinates.
        v: ``(res, res)`` slice coordinates.
        ax: Axes to draw on.
        fill: Optional ``(res, res)`` values.
        fill_label: Color-bar label for ``fill``.
        anchors: Names of the inputs at ``(0, 0)``, ``(1, 0)``, ``(0, 1)``.
        title: Axes title; defaults to the number of regions.

    Returns:
        The axes.
    """
    from .spline import boundaries

    ax = _ax(ax, figsize=(3.0, 3.0))
    U, Vv = _np(u), _np(v)
    extent = (U.min(), U.max(), Vv.min(), Vv.max())
    if fill is not None:
        im = ax.imshow(_np(fill), origin="lower", extent=extent, cmap="innards_seq")
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.8, aspect=25, label=fill_label)
        cb.outline.set_visible(False)
    edge = boundaries(torch.as_tensor(ids)).float()
    if fill is not None:  # a white halo keeps the boundaries readable on dark fill values
        halo_px = F.max_pool2d(edge[None, None], 3, stride=1, padding=1)[0, 0]
        ax.imshow(_np(halo_px), origin="lower", extent=extent, interpolation="nearest", vmin=0, vmax=1,
                  cmap=ListedColormap([(0, 0, 0, 0), to_rgba(INK["surface"], 0.85)]))
    ink = to_rgba(INK["primary"], 0.9 if fill is not None else 0.45)
    ax.imshow(_np(edge), origin="lower", extent=extent, cmap=ListedColormap([(0, 0, 0, 0), ink]), vmin=0, vmax=1,
              interpolation="nearest")
    halo = [patheffects.withStroke(linewidth=2.5, foreground=INK["surface"])]
    for (a, b), name in zip([(0, 0), (1, 0), (0, 1)], anchors):
        ax.plot(a, b, "o", color=INK["primary"], markersize=5, markeredgecolor=INK["surface"], markeredgewidth=1)
        ax.annotate(name, (a, b), xytext=(4, 4), textcoords="offset points", color=INK["primary"], fontsize=7,
                    path_effects=halo)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(INK["grid"])
        spine.set_visible(True)
    ax.set_aspect("equal")
    ax.set_title(title or f"{int(_np(ids).max()) + 1} regions")
    return ax


def to_display_image(
    image: Tensor | np.ndarray,
    mean: Sequence[float] | None = IMAGENET_MEAN,
    std: Sequence[float] | None = IMAGENET_STD,
) -> np.ndarray:
    """
    Converts a normalized ``(3, H, W)`` tensor to an ``(H, W, 3)`` array in [0, 1].

    Args:
        image: Image tensor or array, channels first.
        mean: Normalisation mean to undo; ``None`` skips it.
        std: Normalisation std to undo; ``None`` skips it.

    Returns:
        Displayable image.
    """
    img = _np(image).astype(np.float32)
    if mean is not None and std is not None:
        img = img * np.asarray(std)[:, None, None] + np.asarray(mean)[:, None, None]
    return np.clip(np.transpose(img, (1, 2, 0)), 0, 1)


@styled
def overlay_heatmap(
    image: np.ndarray,
    heat: Tensor | np.ndarray,
    ax: Axes | None = None,
    alpha: float = 0.7,
    vmin: float | None = None,
    vmax: float | None = None,
    signed: bool = False,
) -> Axes:
    """
    An image with a (patch-level) heatmap upsampled on top.

    Opacity grows with the value, so low values leave the image visible.

    Args:
        image: ``(H, W, 3)`` image in [0, 1] (see :func:`to_display_image`).
        heat: Map ``(h, w)``, e.g. from ``interp.spatial_map``.
        ax: Axes to draw on.
        alpha: Maximum heatmap opacity.
        vmin: Value shown transparent (unsigned maps); ``None`` uses the map's minimum.
        vmax: Value at full intensity (for signed maps: the magnitude);
            ``None`` uses the 99.5th percentile, robust to a few extreme pixels.
        signed: Diverging colors (red positive, blue negative, transparent at
            0); otherwise the sequential scale from ``vmin`` to ``vmax``.

    Returns:
        The axes.
    """
    ax = _ax(ax, figsize=(1.8, 1.8))
    h = torch.as_tensor(_np(heat)).float()[None, None]
    h = F.interpolate(h, size=image.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
    if signed:
        vmax = vmax or max(float(np.percentile(np.abs(h), 99.5)), 1e-8)
        mag = np.clip(np.abs(h) / vmax, 0, 1)
        colours = plt.get_cmap("innards_div")(np.clip(0.5 + 0.5 * h / vmax, 0, 1))
    else:
        vmin = float(h.min()) if vmin is None else vmin
        vmax = float(np.percentile(h, 99.5)) if vmax is None else vmax
        mag = np.clip((h - vmin) / max(vmax - vmin, 1e-8), 0, 1)
        colours = plt.get_cmap("innards_seq")(0.3 + 0.4 * mag)  # mid ramp: highlights, never darkens to navy
    colours[..., 3] = alpha * mag
    ax.imshow(image)
    ax.imshow(colours)
    ax.axis("off")
    return ax


@styled
def images(
    images: Tensor | Sequence[np.ndarray],
    maps: Tensor | np.ndarray | None = None,
    titles: Sequence[str] | None = None,
    ncols: int = 4,
    signed: bool = False,
    shared_scale: bool = True,
    boxes: Sequence[tuple[float, float, float, float] | None] | None = None,
    mean: Sequence[float] | None = IMAGENET_MEAN,
    std: Sequence[float] | None = IMAGENET_STD,
    title: str = "",
) -> Figure:
    """
    Grid of images, optionally with heatmaps (spatial maps, attributions, attention).

    Args:
        images: Normalised ``(n, 3, H, W)`` tensors, or ``(H, W, 3)`` arrays in [0, 1].
        maps: Optional ``(n, h, w)`` maps overlaid on the images.
        titles: Optional titles per image.
        ncols: Columns.
        signed: Diverging colors for the maps.
        shared_scale: One color scale for all maps.
        boxes: Optional ``(x, y, w, h)`` rectangles (in image pixels) per image.
        mean: Normalisation mean to undo for tensors (the model's, e.g.
            ``model.pretrained_cfg["mean"]`` for timm); ``None`` skips it.
        std: Normalisation std to undo for tensors.
        title: Figure title.

    Returns:
        The figure.
    """
    imgs = [to_display_image(im, mean, std) if (isinstance(im, Tensor) or np.ndim(im) == 3 and np.shape(im)[0] == 3)
            else np.asarray(im) for im in images]
    n = len(imgs)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    w = style.width(2) if ncols > 3 else style.width(1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, w / ncols * nrows * 1.12 + 0.2), squeeze=False)
    vmin = vmax = None
    if maps is not None and shared_scale:
        m = _np(maps)
        vmax = float(np.percentile(np.abs(m) if signed else m, 99.5))
        vmin = None if signed else float(m.min())
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        if maps is not None:
            overlay_heatmap.__wrapped__(imgs[i], maps[i], ax=ax, vmin=vmin, vmax=vmax, signed=signed)
        else:
            ax.imshow(imgs[i])
            ax.axis("off")
        if boxes is not None and boxes[i] is not None:
            x, y, bw, bh = boxes[i]
            ax.add_patch(Rectangle((x, y), bw, bh, fill=False, edgecolor=INK["surface"], linewidth=2))
            ax.add_patch(Rectangle((x, y), bw, bh, fill=False, edgecolor=INK["primary"], linewidth=0.8))
        if titles is not None:
            ax.set_title(titles[i], fontweight="normal", color=INK["secondary"], fontsize=7)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=8, fontweight="semibold")
    return fig


def top_examples(
    inputs: Tensor,
    examples: Tensor,
    values: Tensor,
    positions: Tensor | None = None,
    grid: tuple[int, int] | None = None,
    exclude: int = 0,
    names: Sequence[str] | None = None,
    ncols: int = 4,
    title: str = "",
    mean: Sequence[float] | None = IMAGENET_MEAN,
    std: Sequence[float] | None = IMAGENET_STD,
) -> Figure:
    """
    The top-activating images of a direction (from ``interp.top_k``), with the patch marked.

    Args:
        inputs: All normalised images ``(N, 3, H, W)`` the activations came from.
        examples: Example indices ``(k,)``.
        values: Scores ``(k,)``.
        positions: Token positions ``(k,)``; with ``grid`` the patch is boxed.
        grid: Spatial ``(h, w)`` of the patch tokens.
        exclude: Number of leading non-spatial tokens (e.g. 1 for CLS).
        names: Optional class names per example, prefixed to the titles.
        ncols: Columns.
        title: Figure title.
        mean: Normalisation mean of the inputs (the model's, e.g.
            ``model.pretrained_cfg["mean"]`` for timm).
        std: Normalisation std of the inputs.

    Returns:
        The figure.
    """
    ex = _np(examples).astype(int)
    vals = _np(values)
    boxes = None
    if positions is not None and grid is not None:
        H, W = inputs.shape[-2:]
        ph, pw = H / grid[0], W / grid[1]
        boxes = []
        for p in _np(positions).astype(int) - exclude:
            boxes.append(None if p < 0 else ((p % grid[1]) * pw, (p // grid[1]) * ph, pw, ph))
    titles = [(f"{names[i]}  " if names is not None else "") + f"{v:.3g}" for i, v in enumerate(vals)]
    return images(inputs[torch.as_tensor(ex)], titles=titles, ncols=ncols, boxes=boxes, title=title, mean=mean, std=std)
