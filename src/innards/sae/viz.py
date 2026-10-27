"""
HTML text views for SAE latents on token sequences (language models).

The text views return HTML strings; :func:`show_html` displays them in
Jupyter. Plots live in :mod:`innards.analysis.plot`: ``plot.histogram`` for
latent density (``log10=True``) and decoder redundancy, ``plot.embedding``
for dictionary maps (``values=``, ``annotate=``), ``plot.similarity_matrix``
(``signed=True``) for atom cosines, ``plot.bars`` for output effects and
attributions, ``plot.images`` for top images with heatmaps.

The latent dashboards follow the feature-browser format by Bricken et al.
(2023).

References:
    Bricken, T. et al. (2023). Towards Monosemanticity: Decomposing Language
        Models With Dictionary Learning. Transformer Circuits Thread.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

import numpy as np
from torch import Tensor


def _np(x) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


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
