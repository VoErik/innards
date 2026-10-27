"""
Plots and HTML views for SAE analysis (needs the ``viz`` extra).

Plotting functions take arrays or tensors computed by
:mod:`innards.sae.analysis` and draw on a matplotlib ``Axes`` (a new figure is
created when ``ax`` is ``None``), so they compose into custom figures. Text
views return HTML strings; :func:`show_html` displays them in Jupyter.

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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# TODO: make colorscheme selectable, deliver custom mpl.style file

def _np(x) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _ax(ax: Axes | None, figsize=(5, 3.5)) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")
    return ax


def _label(labels: Sequence[str] | Callable[[int], str] | None, i: int) -> str:
    if labels is None:
        return str(i)
    return labels(i) if callable(labels) else labels[i]


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
    ax.hist(np.log10(alive), bins=bins, color="#4C72B0")
    ax.set_xlabel("log10 firing frequency")
    ax.set_ylabel("latents")
    ax.set_title(f"Latent density ({(f == 0).sum()} dead of {f.size})")
    return ax


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
    ax.hist(_np(values), bins=bins, color="#55A868")
    ax.set_xlabel("Max Cosine Similarity")
    ax.set_ylabel("SAE Dimension")
    ax.set_title("Decoder redundancy")
    return ax


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
    ax = _ax(ax, figsize=(5.5, 5))
    c = _np(coords)
    sc = ax.scatter(c[:, 0], c[:, 1], s=3, c=None if color is None else _np(color), cmap="viridis",
                    alpha=0.7, linewidths=0)
    if color is not None:
        ax.figure.colorbar(sc, ax=ax, label=color_label)
    for i in highlight:
        ax.scatter(c[i, 0], c[i, 1], s=40, facecolors="none", edgecolors="red")
        ax.annotate(str(i), (c[i, 0], c[i, 1]), fontsize=8, color="red")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Dictionary Map")
    return ax


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
    ax = _ax(ax, figsize=(5, 4.5))
    im = ax.imshow(_np(matrix), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.figure.colorbar(im, ax=ax, label="cosine similarity")
    ticks = range(len(latents))
    ax.set_xticks(ticks, [str(l) for l in latents], rotation=90, fontsize=7)
    ax.set_yticks(ticks, [str(l) for l in latents], fontsize=7)
    return ax


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
    ax = _ax(ax, figsize=(5, 4))
    vals = np.concatenate([_np(top_values), _np(bottom_values)[::-1]])
    ids = np.concatenate([_np(top_ids), _np(bottom_ids)[::-1]]).astype(int)
    names = [_label(labels, int(i)) for i in ids]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=["#C44E52" if v > 0 else "#4C72B0" for v in vals])
    ax.set_yticks(y, names, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Direct Effect")
    ax.set_title(title)
    return ax


def plot_attribution_bars(
    values: Tensor | np.ndarray,
    latents: Tensor | np.ndarray,
    error: float | None = None,
    descriptions: Callable[[int], str] | None = None,
    title: str = "Latent Attribution",
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
    ax = _ax(ax, figsize=(5, 4))
    v = list(_np(values))
    names = [f"#{int(l)}" + (f" {descriptions(int(l))}" if descriptions else "") for l in _np(latents)]
    if error is not None:
        v.append(float(error))
        names.append("SAE error")
    y = np.arange(len(v))[::-1]
    ax.barh(y, v, color=["#C44E52" if x > 0 else "#4C72B0" for x in v])
    ax.set_yticks(y, names, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Attribution (first-order effect of ablation)")
    ax.set_title(title)
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


def overlay_heatmap(
    image: np.ndarray,
    heat: Tensor | np.ndarray,
    ax: Axes | None = None,
    alpha: float = 0.55,
    cmap: str | None = None,
    vmax: float | None = None,
    signed: bool = False,
) -> Axes:
    """
    Draws an image with a patch-level heatmap upsampled on top.

    Args:
        image: ``(H, W, 3)`` image in [0, 1] (see :func:`to_display_image`).
        heat: Patch grid ``(h, w)``, e.g. from
            :func:`innards.sae.analysis.latent_patch_maps`.
        ax: Axes to draw on.
        alpha: Maximum heatmap opacity; opacity scales with the magnitude.
        cmap: Colour map; defaults to ``"inferno"``, or ``"RdBu_r"`` when
            ``signed``.
        vmax: Magnitude mapped to full intensity; ``None`` uses the max.
        signed: Keep the sign (red positive, blue negative), e.g. for
            attributions or AbsTopK activations. Otherwise magnitudes are
            shown.

    Returns:
        The axes.
    """
    ax = _ax(ax, figsize=(3, 3))
    h = torch.as_tensor(_np(heat)).float()[None, None]
    h = F.interpolate(h, size=image.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
    mag = np.abs(h)
    vmax = vmax or max(float(mag.max()), 1e-8)
    mag = np.clip(mag / vmax, 0, 1)
    if signed:
        colours = plt.get_cmap(cmap or "RdBu_r")(np.clip(0.5 + 0.5 * h / vmax, 0, 1))
    else:
        colours = plt.get_cmap(cmap or "inferno")(mag)
    colours[..., 3] = alpha * mag
    ax.imshow(image)
    ax.imshow(colours)
    ax.axis("off")
    return ax


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
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.5 * nrows), layout="constrained",
                             squeeze=False)
    vmax = float(np.abs(_np(heatmaps)).max()) if heatmaps is not None and shared_scale else None
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        if heatmaps is not None:
            overlay_heatmap(images[i], heatmaps[i], ax=ax, vmax=vmax)
        else:
            ax.imshow(images[i])
            ax.axis("off")
        if titles is not None:
            ax.set_title(titles[i], fontsize=8)
        elif values is not None:
            ax.set_title(f"{values[i]:.3g}", fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
    return fig


def _colour(value: float, vmax: float) -> str:
    a = min(abs(value) / vmax, 1.0) if vmax > 0 else 0.0
    rgb = (220, 60, 50) if value >= 0 else (50, 100, 220)
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
        rows.append(f'<div style="margin:2px 0"><span style="color:#888;font-size:80%">'
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
