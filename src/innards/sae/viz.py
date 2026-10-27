"""
Plots and HTML views for SAE analysis (needs the ``viz`` extra).

Plotting functions take arrays or tensors computed by
:mod:`innards.sae.analysis` and draw on a matplotlib ``Axes`` (a new figure is
created when ``ax`` is ``None``), so they compose into custom figures. They
draw in the innards style (:mod:`innards.style`), like
:mod:`innards.analysis.plot`, whose image helpers they share. Text views
return HTML strings; :func:`show_html` displays them in Jupyter.

The latent dashboards follow the feature-browser format by
Bricken et al. (2023): activation density, top-activating examples and output
effects per latent. Always feel free to extend.

References:
    Bricken, T. et al. (2023). Towards Monosemanticity: Decomposing Language
        Models With Dictionary Learning. Transformer Circuits Thread.
"""

from __future__ import annotations

import html
from collections.abc import (
    Callable,
    Sequence
)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

from ..analysis.plot import (  # noqa: F401  (re-exported)
    IMAGENET_MEAN,
    IMAGENET_STD,
    images as _image_grid,
    overlay_heatmap,
    styled,
    to_display_image
)
from ..style import (
    DIVERGING,
    INK,
    color
)

POSITIVE, NEGATIVE = DIVERGING[8], DIVERGING[2]  # the diverging poles: red promotes, blue suppresses


def _np(x) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _ax(ax: Axes | None, figsize=None) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _label(labels: Sequence[str] | Callable[[int], str] | None, i: int) -> str:
    if labels is None:
        return str(i)
    return labels(i) if callable(labels) else labels[i]


@styled
def plot_latent_density(frequency: Tensor | np.ndarray, ax: Axes | None = None, bins: int = 60) -> Axes:
    """
    Histogram of log10 latent firing frequency.

    Dead latents (frequency 0) are reported in the title rather than plotted.

    Args:
        frequency: Per-latent firing frequency, e.g. ``LatentStats.frequency``.
        ax: Axes to draw on.
        bins: Number of histogram bins.

    Returns:
        The axes.
    """
    ax = _ax(ax)
    f = _np(frequency)
    alive = f[f > 0]
    ax.hist(np.log10(alive), bins=bins, color=color(0), edgecolor=INK["surface"], linewidth=0.4)
    ax.grid(axis="y")
    ax.set_xlabel("log10 firing frequency")
    ax.set_ylabel("latents")
    ax.set_title(f"Latent density ({(f == 0).sum()} dead of {f.size})")
    return ax


@styled
def plot_max_cosine_similarity(values: Tensor | np.ndarray, ax: Axes | None = None, bins: int = 60) -> Axes:
    """
    Histogram of each atom's highest cosine similarity to another atom.

    Args:
        values: Output of :func:`innards.sae.analysis.max_cosine_similarity`.
        ax: Axes to draw on.
        bins: Number of histogram bins.

    Returns:
        The axes.
    """
    ax = _ax(ax)
    ax.hist(_np(values), bins=bins, color=color(0), edgecolor=INK["surface"], linewidth=0.4)
    ax.grid(axis="y")
    ax.set_xlabel("max cosine similarity")
    ax.set_ylabel("latents")
    ax.set_title("Decoder redundancy")
    return ax


@styled
def plot_dictionary_map(
    coords: Tensor | np.ndarray,
    color: Tensor | np.ndarray | None = None,
    color_label: str = "log10 frequency",
    highlight: Sequence[int] = (),
    ax: Axes | None = None,
) -> Axes:
    """Scatter plot of a 2-D embedding of the decoder atoms.

    Args:
        coords: Output of :func:`innards.sae.analysis.dictionary_embedding`.
        color: Optional per-latent values for the colour scale, e.g.
            ``log10(frequency)``.
        color_label: Colour-bar label.
        highlight: Latents to annotate.
        ax: Axes to draw on.

    Returns:
        The axes.
    """
    ax = _ax(ax, figsize=(3.2, 3.0))
    c = _np(coords)
    if color is None:
        ax.scatter(c[:, 0], c[:, 1], s=3, color=INK["context"], linewidths=0)
    else:
        sc = ax.scatter(c[:, 0], c[:, 1], s=3, c=_np(color), cmap="innards_seq", linewidths=0)
        cb = ax.figure.colorbar(sc, ax=ax, shrink=0.8, aspect=25, label=color_label)
        cb.outline.set_visible(False)
    for i in highlight:
        ax.scatter(c[i, 0], c[i, 1], s=30, facecolors="none", edgecolors=INK["primary"], linewidths=0.8)
        ax.annotate(str(i), (c[i, 0], c[i, 1]), xytext=(3, 3), textcoords="offset points", fontsize=6.5,
                    color=INK["primary"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Dictionary map")
    return ax


@styled
def plot_similarity_matrix(matrix: Tensor | np.ndarray, latents: Sequence[int], ax: Axes | None = None) -> Axes:
    """
    Heatmap of pairwise decoder cosine similarities.

    Args:
        matrix: Output of :func:`innards.sae.analysis.decoder_similarity_matrix`.
        latents: Latent ids, for tick labels.
        ax: Axes to draw on.

    Returns:
        The axes.
    """
    ax = _ax(ax, figsize=(3.5, 3.0))
    im = ax.imshow(_np(matrix), cmap="innards_div", vmin=-1, vmax=1)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.8, aspect=25, label="cosine similarity")
    cb.outline.set_visible(False)
    ticks = range(len(latents))
    ax.set_xticks(ticks, [str(l) for l in latents], rotation=90)
    ax.set_yticks(ticks, [str(l) for l in latents])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return ax


@styled
def plot_output_effects(
    top_values: Tensor | np.ndarray,
    top_ids: Tensor | np.ndarray,
    bottom_values: Tensor | np.ndarray,
    bottom_ids: Tensor | np.ndarray,
    labels: Sequence[str] | Callable[[int], str] | None = None,
    title: str = "",
    ax: Axes | None = None,
) -> Axes:
    """
    Bar chart of the outputs a latent promotes and suppresses.

    Pass one row of :class:`innards.sae.analysis.OutputEffects`.

    Args:
        top_values: Largest effects, shape ``(n,)``.
        top_ids: Their output ids.
        bottom_values: Most negative effects, shape ``(n,)``.
        bottom_ids: Their output ids.
        labels: Output names (class names, decoded tokens) as a sequence or a
            function of the id.
        title: Axes title.
        ax: Axes to draw on.

    Returns:
        The axes.
    """
    ax = _ax(ax, figsize=(3.5, 3.2))
    vals = np.concatenate([_np(top_values), _np(bottom_values)[::-1]])
    ids = np.concatenate([_np(top_ids), _np(bottom_ids)[::-1]]).astype(int)
    names = [_label(labels, int(i)) for i in ids]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.7, color=[POSITIVE if v > 0 else NEGATIVE for v in vals])
    ax.set_yticks(y, names)
    ax.axvline(0, color=INK["axis"], linewidth=0.8)
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("direct effect")
    ax.set_title(title)
    return ax


@styled
def plot_attribution_bars(
    values: Tensor | np.ndarray,
    latents: Tensor | np.ndarray,
    error: float | None = None,
    descriptions: Callable[[int], str] | None = None,
    title: str = "Latent attribution",
    ax: Axes | None = None,
) -> Axes:
    """
    Bar chart of the latents with the largest attribution to a target.

    Args:
        values: Attribution values, e.g. from ``Attribution.top_latents``.
        latents: Latent ids.
        error: Optional total attribution of the SAE error term, shown as an
            extra bar.
        descriptions: Optional function giving a short label per latent.
        title: Axes title.
        ax: Axes to draw on.

    Returns:
        The axes.
    """
    ax = _ax(ax, figsize=(3.5, 3.2))
    v = list(_np(values))
    names = [f"#{int(l)}" + (f" {descriptions(int(l))}" if descriptions else "") for l in _np(latents)]
    if error is not None:
        v.append(float(error))
        names.append("SAE error")
    y = np.arange(len(v))[::-1]
    ax.barh(y, v, height=0.7, color=[POSITIVE if x > 0 else NEGATIVE for x in v])
    ax.set_yticks(y, names)
    ax.axvline(0, color=INK["axis"], linewidth=0.8)
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("attribution (first-order effect of ablation)")
    ax.set_title(title)
    return ax


@styled
def plot_top_images(
    images: Sequence[np.ndarray],
    heatmaps: Tensor | np.ndarray | None = None,
    values: Sequence[float] | None = None,
    titles: Sequence[str] | None = None,
    ncols: int = 4,
    suptitle: str = "",
    shared_scale: bool = True,
) -> Figure:
    """
    Grid of top-activating images with optional patch heatmaps.

    Args:
        images: ``(H, W, 3)`` images in [0, 1].
        heatmaps: Optional patch grids ``(n, h, w)``.
        values: Optional activation values for the titles.
        titles: Optional titles; override ``values``.
        ncols: Number of columns.
        suptitle: Figure title.
        shared_scale: Use one colour scale for all heatmaps, so intensities
            are comparable across images.

    Returns:
        The figure.
    """
    if titles is None and values is not None:
        titles = [f"{v:.3g}" for v in values]
    return _image_grid.__wrapped__(list(images), maps=heatmaps, titles=titles, ncols=ncols, shared_scale=shared_scale,
                              title=suptitle)


def _colour(value: float, vmax: float) -> str:
    a = min(abs(value) / vmax, 1.0) if vmax > 0 else 0.0
    rgb = (208, 59, 59) if value >= 0 else (42, 120, 214)  # the diverging poles
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{a:.3f})"


def token_activation_html(
    tokens: Sequence[str],
    values: Sequence[float] | Tensor,
    vmax: float | None = None,
    highlight: int | None = None,
) -> str:
    """
    Renders tokens with background colour proportional to a value.

    Positive values are red, negative blue. Hovering shows the value.

    Args:
        tokens: Decoded tokens (e.g. ``tokenizer.convert_ids_to_tokens`` with
            the byte-level markers replaced).
        values: One value per token, e.g. a latent's activation or a
            per-token attribution.
        vmax: Value mapped to full intensity; ``None`` uses the max magnitude.
        highlight: Optional position to underline (e.g. the max-activating
            token).

    Returns:
        An HTML string.
    """
    vals = [float(v) for v in _np(values)]
    vmax = vmax if vmax is not None else max([abs(v) for v in vals] + [1e-8])
    spans = []
    for i, (tok, v) in enumerate(zip(tokens, vals)):
        style = f"background:{_colour(v, vmax)};border-radius:3px;padding:0 1px;"
        if i == highlight:
            style += "border-bottom:2px solid black;"
        text = html.escape(tok).replace("\n", "↵")
        spans.append(f'<span title="{v:.4g}" style="{style}">{text}</span>')
    return ('<span style="font-family:monospace;line-height:1.9;white-space:pre-wrap">' + "".join(spans)
            + "</span>")


def text_examples_html(
    examples: Sequence[tuple[Sequence[str], Sequence[float] | Tensor, int]],
    title: str = "",
    context: int = 12,
    vmax: float | None = None,
) -> str:
    """
    Top-activating text snippets for one latent, one row per example.

    Args:
        examples: ``(tokens, values, position)`` per example, where
            ``position`` is the max-activating token.
        title: Heading.
        context: Tokens shown on each side of ``position``.
        vmax: Shared colour scale; ``None`` uses the max over all examples.

    Returns:
        An HTML string.
    """
    if vmax is None:
        vmax = max([float(np.abs(_np(v)).max()) for _, v, _ in examples] + [1e-8])
    rows = []
    for tokens, values, pos in examples:
        lo, hi = max(0, pos - context), min(len(tokens), pos + context + 1)
        v = _np(values)
        snippet = token_activation_html(tokens[lo:hi], v[lo:hi], vmax=vmax, highlight=pos - lo)
        rows.append(f'<div style="margin:2px 0"><span style="color:#898781;font-size:80%">'
                    f'{float(v[pos]):.3g}</span> {snippet}</div>')
    head = f"<h4 style='margin:6px 0'>{html.escape(title)}</h4>" if title else ""
    return head + "".join(rows)


def show_html(content: str) -> None:
    """Displays HTML in Jupyter, or prints it outside a notebook.

    Args:
        content: HTML string.
    """
    try:
        from IPython.display import HTML, display

        display(HTML(content))
    except ImportError:
        print(content)
