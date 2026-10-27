"""
HTML views of units: a self-contained dashboard and inline text views.

* :func:`save_dashboard` writes a browsable feature dashboard
  (Bricken et al., 2023) for any units cached with
  :func:`innards.analysis.units.cache_units` (SAE latents, neurons, channels,
  directions): an index with a searchable, sortable table and one page per
  unit with statistics, an activation histogram, top examples, examples
  sampled across activation quantiles, nearest units and output effects. The
  file has inline CSS, JavaScript and data (no network access needed) and
  follows the system light / dark preference. How an example is shown is up
  to a *renderer*: :func:`text_renderer` for token sequences,
  :func:`image_renderer` for images with patch heatmaps, or your own
  ``fn(example, position, token_acts, vmax) -> html``.
* :func:`token_activation_html` and :func:`text_examples_html` shade tokens
  by a value in a notebook (:func:`show_html`).

Typical usage::

    path = cache_units(model, loader, "blocks.6", "units/b6", latent_fn(sae), exclude=(0,))
    save_dashboard("b6.html", path, image_renderer(images, n_prefix=1), directions=sae.decoder_weight(),
                   readout=head.weight, output_labels=class_names, noun="latent", per_example=True)

References:
    Bricken, T. et al. (2023). Towards Monosemanticity: Decomposing Language
        Models With Dictionary Learning. Transformer Circuits Thread.
"""

from __future__ import annotations

import base64
import html
import io
import json
from collections.abc import (
    Callable,
    Sequence
)
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .geometry import nearest
from .interp import (
    readout_effect,
    top_outputs
)
from .units import (
    _collect,
    cache_meta,
    cache_stats,
    token_index
)

Renderer = Callable[[int, int, dict[int, float], float], str]
"""``fn(example, position, token_acts, vmax) -> html``: one example of a unit.

``token_acts`` maps the example's token positions to the unit's activation
(stored entries only); ``position`` is the token being shown and ``vmax`` the
unit's largest absolute activation, for a shared colour scale."""

POSITIVE_RGB, NEGATIVE_RGB = (208, 59, 59), (42, 120, 214)  # the diverging poles of innards.style


def _np(x) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _shade(value: float, vmax: float) -> str:
    alpha = min(1.0, abs(value) / max(vmax, 1e-12))
    r, g, b = POSITIVE_RGB if value >= 0 else NEGATIVE_RGB
    return f"rgba({r},{g},{b},{alpha:.3f})"


# --------------------------------------------------------------------------- text views


def token_activation_html(
    tokens: Sequence[str],
    values: Sequence[float] | Tensor,
    vmax: float | None = None,
    highlight: int | None = None,
) -> str:
    """
    Tokens with a background proportional to a value (red positive, blue negative).

    Hovering shows the value.

    Args:
        tokens: Decoded tokens.
        values: One value per token, e.g. a unit's activation or a per-token attribution.
        vmax: Value mapped to full intensity; ``None`` uses the largest magnitude.
        highlight: Optional position to underline (e.g. the peak token).

    Returns:
        An HTML string.
    """
    vals = [float(v) for v in _np(values)]
    vmax = vmax if vmax is not None else max([abs(v) for v in vals] + [1e-8])
    spans = []
    for i, (tok, v) in enumerate(zip(tokens, vals)):
        style = f"background:{_shade(v, vmax)};border-radius:3px;padding:0 1px;"
        if i == highlight:
            style += "border-bottom:2px solid currentColor;"
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
    Text snippets around the peak token of each example, one row each.

    Args:
        examples: ``(tokens, values, position)`` per example.
        title: Heading.
        context: Tokens shown on each side of ``position``.
        vmax: Shared colour scale; ``None`` uses the largest magnitude over all examples.

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
    """
    Displays HTML in Jupyter, or prints it outside a notebook.

    Args:
        content: HTML string.
    """
    try:
        from IPython.display import HTML, display

        display(HTML(content))
    except ImportError:
        print(content)


# --------------------------------------------------------------------------- renderers


class TextRenderer:
    """
    Renders a window of tokens around the shown position, shaded by activation.

    In dashboards the window is stored as compact data and drawn by the page,
    which keeps files small; calling the renderer returns standalone HTML.

    Args:
        tokens: Token strings per example, in loader order (all positions,
            including excluded ones).
        context: Tokens shown on each side.
        sep: String between tokens: ``""`` for subword tokens that carry
            their own spaces, ``" "`` for word-level tokens.
    """

    def __init__(self, tokens: Sequence[Sequence[str]], context: int = 12, sep: str = ""):
        self.tokens, self.context, self.sep = tokens, context, sep

    def spec(self, example: int, position: int, acts: dict[int, float], vmax: float) -> dict[str, Any]:
        """The window as data: tokens, ``[index, value]`` of stored activations, current index, ellipses."""
        toks = self.tokens[example]
        lo, hi = max(0, position - self.context), min(len(toks), position + self.context + 1)
        return {
            "t": [str(t) for t in toks[lo:hi]],
            "a": [[p - lo, round(v, 3)] for p, v in sorted(acts.items()) if lo <= p < hi],
            "c": position - lo,
            "l": lo > 0,
            "r": hi < len(toks),
        }

    def __call__(self, example: int, position: int, acts: dict[int, float], vmax: float) -> str:
        sp = self.spec(example, position, acts, vmax)
        vals = dict((i, v) for i, v in sp["a"])
        spans = []
        for i, tok in enumerate(sp["t"]):
            v = vals.get(i, 0.0)
            style = f' style="background:{_shade(v, vmax)}"' if v else ""
            cls = "tok cur" if i == sp["c"] else "tok"
            spans.append(f'<span class="{cls}"{style} title="{v:.3g}">{html.escape(tok).replace(chr(10), "↵")}</span>')
        body = html.escape(self.sep).join(spans)
        return f'<div class="text">{"…" if sp["l"] else ""}{body}{"…" if sp["r"] else ""}</div>'


def text_renderer(tokens: Sequence[Sequence[str]], context: int = 12, sep: str = "") -> TextRenderer:
    """
    Text renderer for :func:`save_dashboard` (see :class:`TextRenderer`).

    Args:
        tokens: Token strings per example, in loader order.
        context: Tokens shown on each side of the shown position.
        sep: Separator between tokens (``" "`` for word-level tokens).

    Returns:
        The renderer.
    """
    return TextRenderer(tokens, context, sep)


def _image_data_uri(rgb: np.ndarray, quality: int = 85) -> str:
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    try:
        from PIL import Image

        Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
        kind = "jpeg"
    except ImportError:  # matplotlib (viz extra) as a fallback
        import matplotlib.pyplot as plt

        plt.imsave(buf, rgb, format="png")
        kind = "png"
    return f"data:image/{kind};base64," + base64.b64encode(buf.getvalue()).decode()


def image_renderer(
    images: Tensor | Callable[[int], Tensor],
    grid: tuple[int, int] | None = None,
    n_prefix: int = 0,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    size: int = 112,
) -> Renderer:
    """
    Renders an image with the unit's activation over its patch grid.

    Args:
        images: Images ``(N, 3, H, W)`` in loader order, or ``fn(example) -> (3, H, W)``.
        grid: Token grid ``(h, w)``; ``None`` assumes a square grid of the
            non-prefix tokens.
        n_prefix: Tokens before the patches (CLS, registers); token position
            ``n_prefix + i`` is patch ``i``.
        mean: Per-channel normalization mean to undo (e.g. ImageNet's).
        std: Per-channel normalization std to undo.
        size: Width in pixels; images are stored at this size (as JPEG), so
            dashboards with many examples stay small.

    Returns:
        A renderer; the shown patch is outlined.
    """

    def get(example: int) -> Tensor:
        return images(example) if callable(images) else images[example]

    def render(example: int, position: int, acts: dict[int, float], vmax: float) -> str:
        img = get(example).detach().float().cpu()
        if mean is not None and std is not None:
            img = img * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
        h_img, w_img = max(1, round(size * img.shape[1] / img.shape[2])), size
        img = F.interpolate(img[None], size=(h_img, w_img), mode="bilinear", antialias=True)[0]
        gh, gw = grid if grid is not None else (None, None)
        if gh is None:
            n = max([p - n_prefix + 1 for p in acts] + [position - n_prefix + 1, 1])
            gh = gw = int(np.ceil(np.sqrt(n)))
        heat = torch.zeros(gh * gw)
        for p, v in acts.items():
            i = p - n_prefix
            if 0 <= i < gh * gw:
                heat[i] = abs(v) / max(vmax, 1e-12)
        heat = F.interpolate(heat.view(1, 1, gh, gw), size=(h_img, w_img), mode="nearest")[0, 0]
        base = img.permute(1, 2, 0).clamp(0, 1).numpy()
        a = 0.65 * heat.numpy()[..., None]
        out = base * (1 - a) + np.array(POSITIVE_RGB) / 255 * a
        i = position - n_prefix
        if 0 <= i < gh * gw:  # outline the shown patch
            r, c = divmod(i, gw)
            y0, y1 = r * h_img // gh, (r + 1) * h_img // gh - 1
            x0, x1 = c * w_img // gw, (c + 1) * w_img // gw - 1
            out[y0, x0:x1 + 1] = out[y1, x0:x1 + 1] = out[y0:y1 + 1, x0] = out[y0:y1 + 1, x1] = 1.0
        return f'<img class="ex-img" width="{size}" src="{_image_data_uri(out)}" alt="example {example}">'

    return render


def _default_renderer(example: int, position: int, acts: dict[int, float], vmax: float) -> str:
    return f'<span class="muted">example {example}, position {position}</span>'


# --------------------------------------------------------------------------- dashboard


def _histogram(values: np.ndarray, bins: int = 32) -> dict[str, list[float]]:
    if values.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def dashboard_html(
    cache: str | Path,
    render: Renderer | None = None,
    units: Sequence[int] | Tensor | None = None,
    n_top: int = 12,
    per_example: bool = False,
    n_bins: int = 4,
    per_bin: int = 4,
    descriptions: dict[int, str] | None = None,
    directions: Tensor | None = None,
    readout: Tensor | None = None,
    output_labels: Sequence[str] | Callable[[int], str] | None = None,
    n_neighbors: int = 8,
    max_units: int = 2000,
    noun: str = "unit",
    title: str = "Units",
    seed: int = 0,
) -> str:
    """
    Builds a self-contained dashboard page for cached units.

    Args:
        cache: Cache directory from :func:`innards.analysis.units.cache_units`.
        render: Example renderer; ``None`` shows example and position only.
        units: Units to include; ``None`` takes the ``max_units`` most
            frequent live units.
        n_top: Top examples per unit.
        per_example: Top examples are distinct examples (their peak token),
            e.g. distinct images.
        n_bins: Activation-quantile bins per unit.
        per_bin: Examples sampled per bin.
        descriptions: Optional text per unit (e.g. auto-interp labels).
        directions: The units' directions ``(units, d)`` in the
            representation (an SAE's decoder, probe weights), for nearest
            units by cosine and output effects.
        readout: ``(n_outputs, d)`` readout for output effects (unembedding,
            classifier head); needs ``directions``.
        output_labels: Names of the outputs (list or ``fn(index) -> str``).
        n_neighbors: Nearest units shown per unit (needs ``directions``).
        max_units: Cap when ``units`` is ``None``; every unit's examples are
            embedded, so the file grows with it.
        noun: What a unit is called on the page, e.g. ``"latent"``, ``"channel"``.
        title: Page title.
        seed: Seed for the quantile sampling.

    Returns:
        The HTML document.
    """
    render = render or _default_renderer
    meta, stats = cache_meta(cache), cache_stats(cache)
    if units is None:
        alive = (stats.frequency > 0).nonzero().squeeze(-1)
        units = alive[stats.frequency[alive].argsort(descending=True)][:max_units]
    units = torch.as_tensor(units, dtype=torch.long).unique()
    data = _collect(cache, units) if units.numel() else {}
    g = torch.Generator().manual_seed(seed)

    neighbors: dict[int, list[list[float]]] = {}
    effects: dict[int, dict[str, list]] = {}
    if directions is not None and units.numel():
        atoms = F.normalize(directions.detach().float().cpu(), dim=-1)
        v, i = nearest(atoms, units, k=n_neighbors)
        for u, vs, is_ in zip(units.tolist(), v.tolist(), i.tolist()):
            neighbors[u] = [[j, round(s, 3)] for j, s in zip(is_, vs)]
        if readout is not None:
            def name(i: int) -> str:
                if output_labels is None:
                    return str(i)
                return output_labels(i) if callable(output_labels) else str(output_labels[i])

            raw = directions.detach().float().cpu()[units]
            top_v, top_i, bot_v, bot_i = top_outputs(readout_effect(raw, readout.detach().float().cpu()), n=8)
            for row, u in enumerate(units.tolist()):
                effects[u] = {
                    "top": [[name(i), round(v, 4)] for i, v in zip(top_i[row].tolist(), top_v[row].tolist())],
                    "bottom": [[name(i), round(v, 4)] for i, v in zip(bot_i[row].tolist(), bot_v[row].tolist())],
                }

    entries = []
    for u in units.tolist():
        tokens, values = data[u]
        if values.numel() == 0:
            continue
        example, position = token_index(cache, tokens)
        mags = values.abs()
        vmax = float(mags.max())
        top_idx = mags.argsort(descending=True)
        if per_example:  # first (strongest) token of each example
            seen: set[int] = set()
            top_idx = torch.tensor([i for i in top_idx.tolist()
                                    if not (int(example[i]) in seen or seen.add(int(example[i])))], dtype=torch.long)
        top_idx = top_idx[:n_top]
        bin_idx = []
        for part in torch.tensor_split(mags.argsort(), n_bins):
            pick = part[torch.randperm(part.numel(), generator=g)[:per_bin]]
            bin_idx.append(pick[mags[pick].argsort(descending=True)])
        shown = torch.cat([top_idx, *bin_idx]).unique()
        needed = example[shown].unique()
        per_ex: dict[int, dict[int, float]] = {e: {} for e in needed.tolist()}
        hit = torch.isin(example, needed)  # only the shown examples' activations
        for e, p, v in zip(example[hit].tolist(), position[hit].tolist(), values[hit].tolist()):
            per_ex[e][p] = v

        def show(i: int) -> dict[str, Any]:
            e, p, v = int(example[i]), int(position[i]), float(values[i])
            out = {"v": round(v, 4), "ex": e, "pos": p}
            if isinstance(render, TextRenderer):
                out["text"] = render.spec(e, p, per_ex[e], vmax)
            else:
                out["html"] = render(e, p, per_ex[e], vmax)
            return out

        entries.append({
            "id": u,
            "freq": float(stats.frequency[u]),
            "mean": float(stats.mean[u]),
            "max": float(stats.max[u]),
            "count": int(values.numel()),
            "desc": (descriptions or {}).get(u, ""),
            "hist": _histogram(values.numpy()),
            "top": [show(i) for i in top_idx.tolist()],
            "bins": [[show(i) for i in b.tolist()] for b in bin_idx][::-1],  # strongest bin first
            "neighbors": neighbors.get(u, []),
            "effects": effects.get(u),
        })

    freq = stats.frequency.numpy()
    payload = {
        "title": title,
        "noun": noun,
        "n_tokens": meta["n_tokens"],
        "n_units": meta["n_units"] or 0,
        "n_dead": int((freq == 0).sum()),
        "density": _histogram(np.log10(freq[freq > 0]), bins=48),
        "sep": render.sep if isinstance(render, TextRenderer) else "",
        "units": entries,
    }
    return page(title, _DASHBOARD_CSS, _DASHBOARD_JS, payload)


def save_dashboard(path: str | Path, cache: str | Path, render: Renderer | None = None, **kwargs: Any) -> Path:
    """
    Writes :func:`dashboard_html` to ``path``.

    Args:
        path: Output ``.html`` file.
        cache: Cache directory.
        render: Example renderer.
        **kwargs: Passed to :func:`dashboard_html`.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dashboard_html(cache, render, **kwargs), encoding="utf-8")
    return path


def page(title: str, css: str, js: str, payload: dict[str, Any]) -> str:
    """
    A self-contained HTML page: base styles plus ``css``, ``payload`` as JSON
    in ``<script id="data">``, and ``js`` that draws it into ``<div id="app">``.

    Args:
        title: Page title.
        css: Extra CSS.
        js: Page script.
        payload: JSON-serialisable data.

    Returns:
        The HTML document.
    """
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{_BASE_CSS}{css}</style>\n</head>\n"
        f"<body>\n<div id=\"app\"></div>\n<script id=\"data\" type=\"application/json\">{data}</script>\n"
        f"<script>{js}</script>\n</body>\n</html>\n"
    )


_BASE_CSS = """
:root {
  --bg: #fbfaf8; --panel: #ffffff; --text: #1d1d1f; --muted: #6b6b70; --line: #e4e2dd;
  --accent: #c2410c; --accent-soft: #fdebd9; --pos: #ea7a18; --neg: #2f74d8; --hover: #f3f1ec;
  --shadow: 0 1px 2px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.05);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #141416; --panel: #1c1c1f; --text: #ececee; --muted: #9a9aa2; --line: #2e2e33;
    --accent: #fb923c; --accent-soft: #3a2614; --pos: #f59e0b; --neg: #60a5fa; --hover: #26262a;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 4px 16px rgba(0,0,0,.3);
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 22px 0 8px; }
.muted { color: var(--muted); } .num { font-variant-numeric: tabular-nums; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 20px 16px 48px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; box-shadow: var(--shadow); }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
.chip { background: var(--panel); border: 1px solid var(--line); border-radius: 999px; padding: 3px 10px; font-size: 13px; }
.chip b { font-variant-numeric: tabular-nums; }
input[type=search], input[type=text] { background: var(--panel); color: var(--text); border: 1px solid var(--line);
  border-radius: 8px; padding: 7px 10px; font: inherit; width: 100%; max-width: 360px; }
button { background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 8px;
  padding: 5px 11px; font: inherit; cursor: pointer; } button:hover { background: var(--hover); }
"""

_DASHBOARD_CSS = """
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
td.desc { white-space: normal; color: var(--muted); }
th { font-weight: 600; cursor: pointer; user-select: none; position: sticky; top: 0; background: var(--panel); }
th .arrow { color: var(--muted); font-size: 11px; }
tbody tr { cursor: pointer; } tbody tr:hover { background: var(--hover); }
.table-wrap { max-height: 70vh; overflow: auto; border-radius: 10px; }
.bar { fill: var(--accent); opacity: .85; } .axis { fill: var(--muted); font-size: 11px; }
.hist { width: 100%; height: 150px; display: block; }
.examples { display: grid; gap: 6px; }
.example { display: grid; grid-template-columns: 70px 1fr; gap: 10px; align-items: center; padding: 6px 10px;
  border-bottom: 1px solid var(--line); }
.example:last-child { border-bottom: 0; }
.example .val { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; text-align: right; }
.examples.images { grid-template-columns: repeat(auto-fill, minmax(124px, 1fr)); }
.examples.images .example { grid-template-columns: 1fr; border: 0; justify-items: center; }
.examples.images .val { text-align: center; }
.text { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; line-height: 1.9;
  white-space: pre-wrap; word-break: break-word; }
.tok { border-radius: 3px; padding: 1px 0; } .tok.cur { outline: 1.5px solid var(--text); outline-offset: 1px; }
.ex-img { border-radius: 6px; display: block; image-rendering: pixelated; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 760px) { .grid2 { grid-template-columns: 1fr; } .example { grid-template-columns: 54px 1fr; } }
.list { margin: 0; padding: 0; list-style: none; } .list li { display: flex; justify-content: space-between; gap: 12px;
  padding: 3px 0; border-bottom: 1px dashed var(--line); font-variant-numeric: tabular-nums; }
.pad { padding: 12px 14px; }
.nav { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
.desc-line { font-size: 15px; margin: 4px 0 0; }
"""

_DASHBOARD_JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const app = document.getElementById('app');
const byId = new Map(D.units.map(l => [l.id, l]));
const N = D.noun, NC = N.charAt(0).toUpperCase() + N.slice(1);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = (v, d = 3) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)) ? v.toExponential(2) : (+v).toFixed(d);
const pct = v => (v * 100 < 0.01 && v > 0) ? (v * 100).toExponential(1) + '%' : (v * 100).toFixed(3) + '%';
let sortKey = 'freq', sortDir = -1, query = '', order = D.units.map(l => l.id);

function hist(h, xlabel) {
  if (!h.counts.length) return '<p class="muted">No activations.</p>';
  const W = 600, H = 150, P = 26, n = h.counts.length, max = Math.max(...h.counts);
  const bw = (W - 2 * P) / n;
  let bars = '';
  h.counts.forEach((c, i) => {
    const bh = max ? (H - 2 * P) * c / max : 0;
    bars += `<rect class="bar" x="${P + i * bw + 0.5}" y="${H - P - bh}" width="${Math.max(bw - 1, 1)}" height="${bh}"><title>${fmt(h.edges[i])} – ${fmt(h.edges[i + 1])}: ${c}</title></rect>`;
  });
  return `<svg class="hist" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${bars}
    <text class="axis" x="${P}" y="${H - 8}">${fmt(h.edges[0])}</text>
    <text class="axis" x="${W - P}" y="${H - 8}" text-anchor="end">${fmt(h.edges[n])}</text>
    <text class="axis" x="${W / 2}" y="${H - 8}" text-anchor="middle">${esc(xlabel)}</text>
    <text class="axis" x="${P}" y="14">${max}</text></svg>`;
}

function shade(v, vmax) {
  const a = Math.min(1, Math.abs(v) / Math.max(vmax, 1e-12)).toFixed(3);
  return v >= 0 ? `rgba(208,59,59,${a})` : `rgba(42,120,214,${a})`;
}
function textHtml(s, vmax) {
  const vals = new Map(s.a);
  const spans = s.t.map((t, i) => { const v = vals.get(i) || 0;
    return `<span class="tok${i === s.c ? ' cur' : ''}"${v ? ` style="background:${shade(v, vmax)}"` : ''} title="${fmt(v)}">${esc(t).replace(/\n/g, '↵')}</span>`; });
  return `<div class="text">${s.l ? '…' : ''}${spans.join(esc(D.sep))}${s.r ? '…' : ''}</div>`;
}
function examples(list, vmax) {
  if (!list.length) return '<p class="muted">None.</p>';
  const images = list.some(e => e.html && e.html.includes('<img'));
  return `<div class="card examples ${images ? 'images' : ''}">` + list.map(e =>
    `<div class="example"><div class="val" title="example ${e.ex}, position ${e.pos}">${fmt(e.v)}</div><div>${e.text ? textHtml(e.text, vmax) : e.html}</div></div>`).join('') + '</div>';
}

function index() {
  const rows = D.units.filter(l => !query || String(l.id) === query.trim() || l.desc.toLowerCase().includes(query.toLowerCase()))
    .sort((a, b) => sortDir * (typeof a[sortKey] === 'string' ? a[sortKey].localeCompare(b[sortKey]) : a[sortKey] - b[sortKey]));
  order = rows.map(l => l.id);
  const col = (k, label) => `<th data-k="${k}">${label} <span class="arrow">${sortKey === k ? (sortDir < 0 ? '▼' : '▲') : ''}</span></th>`;
  const hasDesc = D.units.some(l => l.desc);
  app.innerHTML = `<div class="wrap">
    <h1>${esc(D.title)}</h1>
    <div class="chips"><span class="chip"><b>${D.n_units.toLocaleString()}</b> ${N}s</span>
      <span class="chip"><b>${D.units.length.toLocaleString()}</b> shown</span>
      <span class="chip"><b>${D.n_dead.toLocaleString()}</b> dead</span>
      <span class="chip"><b>${D.n_tokens.toLocaleString()}</b> tokens</span></div>
    <div class="card pad"><h2 style="margin-top:0">Firing frequency (log10)</h2>${hist(D.density, 'log10 frequency')}</div>
    <h2>${NC}s</h2>
    <div class="nav"><input type="search" id="q" placeholder="Search by id${hasDesc ? ' or description' : ''}" value="${esc(query)}">
      <span class="muted">${rows.length.toLocaleString()} match</span></div>
    <div class="card table-wrap"><table><thead><tr>${col('id', NC)}${col('freq', 'Frequency')}${col('mean', 'Mean act.')}${col('max', 'Max act.')}${hasDesc ? col('desc', 'Description') : ''}</tr></thead>
    <tbody>${rows.map(l => `<tr data-id="${l.id}"><td>#${l.id}</td><td>${pct(l.freq)}</td><td>${fmt(l.mean)}</td><td>${fmt(l.max)}</td>${hasDesc ? `<td class="desc">${esc(l.desc)}</td>` : ''}</tr>`).join('')}</tbody></table></div>
  </div>`;
  const q = document.getElementById('q');
  q.addEventListener('input', () => { query = q.value; const pos = q.selectionStart; index(); const q2 = document.getElementById('q'); q2.focus(); q2.setSelectionRange(pos, pos); });
  app.querySelectorAll('th').forEach(th => th.addEventListener('click', () => {
    const k = th.dataset.k; sortDir = sortKey === k ? -sortDir : (k === 'id' || k === 'desc' ? 1 : -1); sortKey = k; index(); }));
  app.querySelectorAll('tbody tr').forEach(tr => tr.addEventListener('click', () => { location.hash = '#/unit/' + tr.dataset.id; }));
}

function unitPage(id) {
  const l = byId.get(id);
  if (!l) { app.innerHTML = `<div class="wrap"><p>${NC} #${id} is not in this dashboard. <a href="#/">Back</a></p></div>`; return; }
  const i = order.indexOf(id), prev = i > 0 ? order[i - 1] : null, next = i >= 0 && i < order.length - 1 ? order[i + 1] : null;
  const nb = l.neighbors.length ? `<ul class="list">${l.neighbors.map(([j, s]) =>
    `<li>${byId.has(j) ? `<a href="#/unit/${j}">#${j}</a>` : `<span class="muted">#${j}</span>`}<span>${fmt(s)}</span></li>`).join('')}</ul>` : '<p class="muted">Pass <code>directions=</code> to list them.</p>';
  const eff = l.effects ? `<div class="grid2"><div><div class="muted">Promotes</div><ul class="list">${l.effects.top.map(([n, v]) =>
      `<li><span>${esc(n)}</span><span>${fmt(v)}</span></li>`).join('')}</ul></div>
      <div><div class="muted">Suppresses</div><ul class="list">${l.effects.bottom.map(([n, v]) =>
      `<li><span>${esc(n)}</span><span>${fmt(v)}</span></li>`).join('')}</ul></div></div>` : '<p class="muted">Pass <code>directions=</code> and <code>readout=</code> to show them.</p>';
  const nBins = l.bins.length, vmax = Math.max(1e-12, ...l.top.map(e => Math.abs(e.v)));
  app.innerHTML = `<div class="wrap">
    <div class="nav"><a href="#/">← All ${N}s</a>
      ${prev !== null ? `<button data-go="${prev}">← #${prev}</button>` : ''}${next !== null ? `<button data-go="${next}">#${next} →</button>` : ''}</div>
    <h1>${NC} #${l.id}</h1>${l.desc ? `<p class="desc-line">${esc(l.desc)}</p>` : ''}
    <div class="chips"><span class="chip">frequency <b>${pct(l.freq)}</b></span><span class="chip">mean act. <b>${fmt(l.mean)}</b></span>
      <span class="chip">max act. <b>${fmt(l.max)}</b></span><span class="chip">fires on <b>${l.count.toLocaleString()}</b> tokens</span></div>
    <div class="card pad"><h2 style="margin-top:0">Activation distribution</h2>${hist(l.hist, 'activation')}</div>
    <h2>Top activations</h2>${examples(l.top, vmax)}
    ${l.bins.map((b, k) => `<h2>Quantile ${nBins - k} of ${nBins} <span class="muted">(${k === 0 ? 'strongest' : k === nBins - 1 ? 'weakest' : 'intermediate'} activations, random sample)</span></h2>${examples(b, vmax)}`).join('')}
    <div class="grid2"><div><h2>Nearest ${N}s (cosine)</h2><div class="card pad">${nb}</div></div>
      <div><h2>Output effects</h2><div class="card pad">${eff}</div></div></div>
  </div>`;
  app.querySelectorAll('[data-go]').forEach(b => b.addEventListener('click', () => { location.hash = '#/unit/' + b.dataset.go; }));
  window.scrollTo(0, 0);
}

function route() {
  const m = location.hash.match(/^#\/unit\/(\d+)/);
  if (m) unitPage(+m[1]); else index();
}
window.addEventListener('hashchange', route);
route();
"""
