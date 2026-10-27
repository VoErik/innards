"""
Self-contained HTML views: a latent dashboard and an attribution-graph viewer.

Both functions write a single HTML file with inline CSS, JavaScript and data
(no network access needed), so the result can be opened locally, attached or
hosted anywhere. Pages follow the system light / dark preference.

* :func:`save_latent_dashboard` builds a browsable feature dashboard from a
  :class:`~innards.sae.latent_cache.LatentCache` (Bricken et al., 2023): an index with
  a searchable, sortable table of latents and one page per latent with
  statistics, an activation histogram, top examples, examples sampled across
  activation quantiles, nearest latents and output effects. How an example is
  shown is up to a *renderer*: :func:`text_renderer` for token sequences,
  :func:`image_renderer` for images with patch heatmaps, or your own
  ``fn(example, position, token_acts, vmax) -> html``.
* :func:`save_graph_html` draws an :class:`~innards.sae.graph.AttributionGraph` with
  sites as rows and token positions as columns; edges can be filtered by
  strength and a clicked node shows its strongest inputs and outputs.

Typical usage::

    cache = LatentCache(cache_latents(sae, model, hook, val_loader, "latents/l3"))
    save_latent_dashboard("dash.html", cache, text_renderer(val_tokens), sae=sae,
                          readout=model.get_output_embeddings().weight, output_labels=vocab)
    save_graph_html(graph.prune(0.8), "graph.html", tokens=prompt_tokens)

References:
    Bricken, T. et al. (2023). Towards Monosemanticity: Decomposing Language
        Models With Dictionary Learning. Transformer Circuits Thread.
    Ameisen, E. et al. (2025). Circuit Tracing: Revealing Computational Graphs
        in Language Models. Transformer Circuits Thread.
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

from .graph import (
    AttributionGraph,
    Node
)
from .latent_cache import LatentCache

Renderer = Callable[[int, int, dict[int, float], float], str]
"""``fn(example, position, token_acts, vmax) -> html``: one example of a latent.

``token_acts`` maps the example's token positions to the latent's activation
(only non-zero entries); ``position`` is the token being shown and ``vmax`` the
latent's largest absolute activation, for a shared colour scale."""


# --------------------------------------------------------------------------- renderers


def _shade(value: float, vmax: float) -> str:
    alpha = min(1.0, abs(value) / max(vmax, 1e-12))
    rgb = "236,130,30" if value >= 0 else "50,120,225"
    return f"rgba({rgb},{alpha:.3f})"


class TextRenderer:
    """
    Renders a window of tokens around the shown position, shaded by activation.

    In dashboards the window is stored as compact data and drawn by the page,
    which keeps files small; calling the renderer returns standalone HTML.

    Args:
        tokens: Token strings per example, in the loader order the cache was
            built from (the same positions, including excluded ones).
        context: Tokens shown on each side.
        sep: String placed between tokens: ``""`` for subword tokenizers whose
            tokens carry their own spaces, ``" "`` for word-level tokens.
    """

    def __init__(self, tokens: Sequence[Sequence[str]], context: int = 12, sep: str = ""):
        self.tokens, self.context, self.sep = tokens, context, sep

    def spec(self, example: int, position: int, acts: dict[int, float], vmax: float) -> dict[str, Any]:
        """The window as data: tokens, ``[index, value]`` of non-zero activations, current index, ellipses."""
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
    Text renderer for :func:`save_latent_dashboard` (see :class:`TextRenderer`).

    Args:
        tokens: Token strings per example, in cache (loader) order.
        context: Tokens shown on each side of the shown position.
        sep: Separator between tokens (``" "`` for word-level tokens).

    Returns:
        The renderer.
    """
    return TextRenderer(tokens, context, sep)


def _png_data_uri(rgb: np.ndarray) -> str:
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    try:
        from PIL import Image

        Image.fromarray(rgb).save(buf, format="PNG")
    except ImportError:  # matplotlib (viz extra) as a fallback
        import matplotlib.pyplot as plt

        plt.imsave(buf, rgb, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def image_renderer(
    images: Tensor | Callable[[int], Tensor],
    grid: tuple[int, int] | None = None,
    n_prefix: int = 0,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    size: int = 112,
) -> Renderer:
    """
    Renders an image with the latent's activation over its patch grid.

    Args:
        images: Images ``(N, 3, H, W)`` in loader order, or ``fn(example) ->
            (3, H, W)``.
        grid: Patch grid ``(h, w)``; ``None`` assumes a square grid of the
            non-prefix tokens.
        n_prefix: Tokens before the patches (CLS, registers); token position
            ``n_prefix + i`` is patch ``i``.
        mean: Per-channel normalization mean to undo (e.g. ImageNet's).
        std: Per-channel normalization std to undo.
        size: Displayed width in pixels.

    Returns:
        A renderer; the shown patch is outlined.
    """

    def get(example: int) -> Tensor:
        return images(example) if callable(images) else images[example]

    def render(example: int, position: int, acts: dict[int, float], vmax: float) -> str:
        img = get(example).detach().float().cpu()
        if mean is not None and std is not None:
            img = img * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
        _, h_img, w_img = img.shape
        gh, gw = grid if grid is not None else (None, None)
        if gh is None:
            n = max([p - n_prefix + 1 for p in acts] + [position - n_prefix + 1, 1])
            side = int(np.ceil(np.sqrt(n)))
            gh = gw = side
        heat = torch.zeros(gh * gw)
        for p, v in acts.items():
            i = p - n_prefix
            if 0 <= i < gh * gw:
                heat[i] = abs(v) / max(vmax, 1e-12)
        heat = F.interpolate(heat.view(1, 1, gh, gw), size=(h_img, w_img), mode="nearest")[0, 0]
        base = img.permute(1, 2, 0).clamp(0, 1).numpy()
        color = np.array([0.93, 0.51, 0.12])
        a = 0.65 * heat.numpy()[..., None]
        out = base * (1 - a) + color * a
        i = position - n_prefix
        if 0 <= i < gh * gw:  # outline the shown patch
            r, c = divmod(i, gw)
            y0, y1 = r * h_img // gh, (r + 1) * h_img // gh - 1
            x0, x1 = c * w_img // gw, (c + 1) * w_img // gw - 1
            out[y0, x0:x1 + 1] = out[y1, x0:x1 + 1] = out[y0:y1 + 1, x0] = out[y0:y1 + 1, x1] = 1.0
        return f'<img class="ex-img" width="{size}" src="{_png_data_uri(out)}" alt="example {example}">'

    return render


def _default_renderer(example: int, position: int, acts: dict[int, float], vmax: float) -> str:
    return f'<span class="muted">example {example}, position {position}</span>'


# --------------------------------------------------------------------------- latent dashboard


def _histogram(values: np.ndarray, bins: int = 32) -> dict[str, list[float]]:
    if values.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def _collect(cache: LatentCache, latents: Tensor) -> dict[int, tuple[Tensor, Tensor]]:
    """All activations of the selected latents in one pass: latent -> (global tokens, values)."""
    lats, toks, vals = [], [], []
    for d in cache.shards():
        hit = torch.isin(d["latent"], latents)
        lats.append(d["latent"][hit])
        toks.append(d["token"][hit])
        vals.append(d["value"][hit])
    lat, tok, val = torch.cat(lats), torch.cat(toks), torch.cat(vals)
    order = lat.argsort(stable=True)
    lat, tok, val = lat[order], tok[order], val[order]
    counts = torch.bincount(lat, minlength=cache.d_sae)
    out, start = {}, 0
    for latent in latents.sort().values.tolist():
        n = int(counts[latent])
        out[latent] = (tok[start : start + n], val[start : start + n])
        start += n
    return out


def latent_dashboard_html(
    cache: LatentCache,
    render: Renderer | None = None,
    latents: Sequence[int] | Tensor | None = None,
    n_top: int = 12,
    n_bins: int = 4,
    per_bin: int = 4,
    descriptions: dict[int, str] | None = None,
    sae: Any = None,
    readout: Tensor | None = None,
    output_labels: Sequence[str] | Callable[[int], str] | None = None,
    n_neighbors: int = 8,
    max_latents: int = 2000,
    title: str = "Latent dashboard",
    seed: int = 0,
) -> str:
    """
    Builds a self-contained latent dashboard page.

    Args:
        cache: Latent cache (see :func:`innards.sae.latent_cache.cache_latents`).
        render: Example renderer; ``None`` shows example and position only.
        latents: Latents to include; ``None`` takes the ``max_latents`` most
            frequent live latents.
        n_top: Top-activating examples per latent.
        n_bins: Activation-quantile bins per latent.
        per_bin: Examples sampled per bin.
        descriptions: Optional text per latent (e.g. auto-interp labels).
        sae: The SAE, for nearest latents (decoder cosine) and output effects.
        readout: ``(n_outputs, d)`` readout for output effects (unembedding,
            classifier head); needs ``sae``.
        output_labels: Names of the outputs (list or ``fn(index) -> str``).
        n_neighbors: Nearest latents shown per latent (needs ``sae``).
        max_latents: Cap when ``latents`` is ``None``; every latent's examples
            are embedded, so the file grows with it.
        title: Page title.
        seed: Seed for the quantile sampling.

    Returns:
        The HTML document.
    """
    render = render or _default_renderer
    stats = cache.stats()
    if latents is None:
        alive = (stats.frequency > 0).nonzero().squeeze(-1)
        latents = alive[stats.frequency[alive].argsort(descending=True)][:max_latents]
    latents = torch.as_tensor(latents, dtype=torch.long).unique()
    data = _collect(cache, latents)
    g = torch.Generator().manual_seed(seed)

    neighbors: dict[int, list[list[float]]] = {}
    effects: dict[int, dict[str, list]] = {}
    if sae is not None:
        atoms = F.normalize(sae.decoder_weight().detach().float().cpu(), dim=-1)
        for chunk in latents.split(1024):
            sims = atoms[chunk] @ atoms.T
            sims[torch.arange(len(chunk)), chunk] = -2.0
            v, i = sims.topk(min(n_neighbors, atoms.shape[0] - 1), dim=-1)
            for lat, vs, is_ in zip(chunk.tolist(), v.tolist(), i.tolist()):
                neighbors[lat] = [[j, round(s, 3)] for j, s in zip(is_, vs)]
        if readout is not None:
            from .analysis import output_effects

            def name(i: int) -> str:
                if output_labels is None:
                    return str(i)
                return output_labels(i) if callable(output_labels) else str(output_labels[i])

            eff = output_effects(sae, readout, latents=latents, n=8)
            for row, lat in enumerate(eff.latents.tolist()):
                effects[lat] = {
                    "top": [[name(i), round(v, 4)] for i, v in zip(eff.top_ids[row].tolist(),
                                                                   eff.top_values[row].tolist())],
                    "bottom": [[name(i), round(v, 4)] for i, v in zip(eff.bottom_ids[row].tolist(),
                                                                      eff.bottom_values[row].tolist())],
                }

    entries = []
    for lat in latents.tolist():
        tokens, values = data[lat]
        if values.numel() == 0:
            continue
        mags = values.abs()
        vmax = float(mags.max())
        top_idx = mags.argsort(descending=True)[:n_top]
        order = mags.argsort()
        bin_idx = []
        for part in torch.tensor_split(order, n_bins):
            pick = part[torch.randperm(part.numel(), generator=g)[:per_bin]]
            bin_idx.append(pick[mags[pick].argsort(descending=True)])
        shown = torch.cat([top_idx, *bin_idx]).unique()
        example, position = cache.token_index(tokens)
        needed = example[shown].unique()
        per_example: dict[int, dict[int, float]] = {e: {} for e in needed.tolist()}
        hit = torch.isin(example, needed)  # only the shown examples' activations
        for e, p, v in zip(example[hit].tolist(), position[hit].tolist(), values[hit].tolist()):
            per_example[e][p] = v

        def show(i: int) -> dict[str, Any]:
            e, p, v = int(example[i]), int(position[i]), float(values[i])
            out = {"v": round(v, 4), "ex": e, "pos": p}
            if isinstance(render, TextRenderer):
                out["text"] = render.spec(e, p, per_example[e], vmax)
            else:
                out["html"] = render(e, p, per_example[e], vmax)
            return out

        entries.append({
            "id": lat,
            "freq": float(stats.frequency[lat]),
            "mean": float(stats.mean_activation[lat]),
            "max": float(stats.max_activation[lat]),
            "count": int(values.numel()),
            "desc": (descriptions or {}).get(lat, ""),
            "hist": _histogram(values.numpy()),
            "top": [show(i) for i in top_idx.tolist()],
            "bins": [[show(i) for i in b.tolist()] for b in bin_idx][::-1],  # strongest bin first
            "neighbors": neighbors.get(lat, []),
            "effects": effects.get(lat),
        })

    freq = stats.frequency.numpy()
    payload = {
        "title": title,
        "n_tokens": cache.n_tokens,
        "d_sae": cache.d_sae,
        "n_dead": int((freq == 0).sum()),
        "density": _histogram(np.log10(freq[freq > 0]), bins=48),
        "sep": render.sep if isinstance(render, TextRenderer) else "",
        "latents": entries,
    }
    return _page(title, _DASHBOARD_CSS, _DASHBOARD_JS, payload)


def save_latent_dashboard(path: str | Path, cache: LatentCache, render: Renderer | None = None, **kwargs: Any) -> Path:
    """
    Writes :func:`latent_dashboard_html` to ``path``.

    Args:
        path: Output ``.html`` file.
        cache: Latent cache.
        render: Example renderer.
        **kwargs: Passed to :func:`latent_dashboard_html`.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latent_dashboard_html(cache, render, **kwargs), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- graph viewer


def graph_html(
    graph: AttributionGraph,
    tokens: Sequence[str] | None = None,
    descriptions: dict[tuple[str, int], str] | Callable[[Node], str] | None = None,
    title: str = "Attribution graph",
    metric_label: str = "metric",
) -> str:
    """
    Builds a self-contained, interactive view of an attribution graph.

    Args:
        graph: The graph (usually pruned, e.g. ``graph.prune(0.8)``).
        tokens: Token strings of the (single) input, to label the columns.
        descriptions: Text per latent node, keyed by ``(site, latent)`` or a
            function of the node.
        title: Page title.
        metric_label: Name of the metric node (e.g. ``'logit(" Paris")'``).

    Returns:
        The HTML document.
    """
    d = graph.to_dict()
    sites: list[str] = []
    for n in graph.nodes:
        if n.site not in sites:
            sites.append(n.site)
    for node, nd in zip(graph.nodes, d["nodes"]):
        desc = ""
        if node.kind == "latent" and descriptions is not None:
            desc = descriptions(node) if callable(descriptions) else descriptions.get((node.site, node.latent), "")
        nd["desc"] = desc
    payload = {
        "title": title,
        "metric_label": metric_label,
        "metric": d["metric"],
        "tokens": list(tokens) if tokens is not None else None,
        "sites": sites,
        "nodes": d["nodes"],
        "edges": [[e["source"], e["target"], e["weight"]] for e in d["edges"]],
    }
    return _page(title, _GRAPH_CSS, _GRAPH_JS, payload)


def save_graph_html(graph: AttributionGraph, path: str | Path, **kwargs: Any) -> Path:
    """
    Writes :func:`graph_html` to ``path``.

    Args:
        graph: The graph.
        path: Output ``.html`` file.
        **kwargs: Passed to :func:`graph_html`.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph_html(graph, **kwargs), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- page assembly


def _page(title: str, css: str, js: str, payload: dict[str, Any]) -> str:
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
const byId = new Map(D.latents.map(l => [l.id, l]));
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = (v, d = 3) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)) ? v.toExponential(2) : (+v).toFixed(d);
const pct = v => (v * 100 < 0.01 && v > 0) ? (v * 100).toExponential(1) + '%' : (v * 100).toFixed(3) + '%';
let sortKey = 'freq', sortDir = -1, query = '', order = D.latents.map(l => l.id);

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
  return v >= 0 ? `rgba(236,130,30,${a})` : `rgba(50,120,225,${a})`;
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
  const rows = D.latents.filter(l => !query || String(l.id) === query.trim() || l.desc.toLowerCase().includes(query.toLowerCase()))
    .sort((a, b) => sortDir * (typeof a[sortKey] === 'string' ? a[sortKey].localeCompare(b[sortKey]) : a[sortKey] - b[sortKey]));
  order = rows.map(l => l.id);
  const col = (k, label) => `<th data-k="${k}">${label} <span class="arrow">${sortKey === k ? (sortDir < 0 ? '▼' : '▲') : ''}</span></th>`;
  const hasDesc = D.latents.some(l => l.desc);
  app.innerHTML = `<div class="wrap">
    <h1>${esc(D.title)}</h1>
    <div class="chips"><span class="chip"><b>${D.d_sae.toLocaleString()}</b> latents</span>
      <span class="chip"><b>${D.latents.length.toLocaleString()}</b> shown</span>
      <span class="chip"><b>${D.n_dead.toLocaleString()}</b> dead</span>
      <span class="chip"><b>${D.n_tokens.toLocaleString()}</b> tokens</span></div>
    <div class="card pad"><h2 style="margin-top:0">Firing frequency (log10)</h2>${hist(D.density, 'log10 frequency')}</div>
    <h2>Latents</h2>
    <div class="nav"><input type="search" id="q" placeholder="Search by id${hasDesc ? ' or description' : ''}" value="${esc(query)}">
      <span class="muted">${rows.length.toLocaleString()} match</span></div>
    <div class="card table-wrap"><table><thead><tr>${col('id', 'Latent')}${col('freq', 'Frequency')}${col('mean', 'Mean act.')}${col('max', 'Max act.')}${hasDesc ? col('desc', 'Description') : ''}</tr></thead>
    <tbody>${rows.map(l => `<tr data-id="${l.id}"><td>#${l.id}</td><td>${pct(l.freq)}</td><td>${fmt(l.mean)}</td><td>${fmt(l.max)}</td>${hasDesc ? `<td class="desc">${esc(l.desc)}</td>` : ''}</tr>`).join('')}</tbody></table></div>
  </div>`;
  const q = document.getElementById('q');
  q.addEventListener('input', () => { query = q.value; const pos = q.selectionStart; index(); const q2 = document.getElementById('q'); q2.focus(); q2.setSelectionRange(pos, pos); });
  app.querySelectorAll('th').forEach(th => th.addEventListener('click', () => {
    const k = th.dataset.k; sortDir = sortKey === k ? -sortDir : (k === 'id' || k === 'desc' ? 1 : -1); sortKey = k; index(); }));
  app.querySelectorAll('tbody tr').forEach(tr => tr.addEventListener('click', () => { location.hash = '#/latent/' + tr.dataset.id; }));
}

function latentPage(id) {
  const l = byId.get(id);
  if (!l) { app.innerHTML = `<div class="wrap"><p>Latent #${id} is not in this dashboard. <a href="#/">Back</a></p></div>`; return; }
  const i = order.indexOf(id), prev = i > 0 ? order[i - 1] : null, next = i >= 0 && i < order.length - 1 ? order[i + 1] : null;
  const nb = l.neighbors.length ? `<ul class="list">${l.neighbors.map(([j, s]) =>
    `<li>${byId.has(j) ? `<a href="#/latent/${j}">#${j}</a>` : `<span class="muted">#${j}</span>`}<span>${fmt(s)}</span></li>`).join('')}</ul>` : '<p class="muted">Pass <code>sae=</code> to list them.</p>';
  const eff = l.effects ? `<div class="grid2"><div><div class="muted">Promotes</div><ul class="list">${l.effects.top.map(([n, v]) =>
      `<li><span>${esc(n)}</span><span>${fmt(v)}</span></li>`).join('')}</ul></div>
      <div><div class="muted">Suppresses</div><ul class="list">${l.effects.bottom.map(([n, v]) =>
      `<li><span>${esc(n)}</span><span>${fmt(v)}</span></li>`).join('')}</ul></div></div>` : '<p class="muted">Pass <code>sae=</code> and <code>readout=</code> to show them.</p>';
  const nBins = l.bins.length, vmax = Math.max(1e-12, ...l.top.map(e => Math.abs(e.v)));
  app.innerHTML = `<div class="wrap">
    <div class="nav"><a href="#/">← All latents</a>
      ${prev !== null ? `<button data-go="${prev}">← #${prev}</button>` : ''}${next !== null ? `<button data-go="${next}">#${next} →</button>` : ''}</div>
    <h1>Latent #${l.id}</h1>${l.desc ? `<p class="desc-line">${esc(l.desc)}</p>` : ''}
    <div class="chips"><span class="chip">frequency <b>${pct(l.freq)}</b></span><span class="chip">mean act. <b>${fmt(l.mean)}</b></span>
      <span class="chip">max act. <b>${fmt(l.max)}</b></span><span class="chip">fires on <b>${l.count.toLocaleString()}</b> tokens</span></div>
    <div class="card pad"><h2 style="margin-top:0">Activation distribution</h2>${hist(l.hist, 'activation')}</div>
    <h2>Top activations</h2>${examples(l.top, vmax)}
    ${l.bins.map((b, k) => `<h2>Quantile ${nBins - k} of ${nBins} <span class="muted">(${k === 0 ? 'strongest' : k === nBins - 1 ? 'weakest' : 'intermediate'} activations, random sample)</span></h2>${examples(b, vmax)}`).join('')}
    <div class="grid2"><div><h2>Nearest latents (decoder cosine)</h2><div class="card pad">${nb}</div></div>
      <div><h2>Output effects</h2><div class="card pad">${eff}</div></div></div>
  </div>`;
  app.querySelectorAll('[data-go]').forEach(b => b.addEventListener('click', () => { location.hash = '#/latent/' + b.dataset.go; }));
  window.scrollTo(0, 0);
}

function route() {
  const m = location.hash.match(/^#\/latent\/(\d+)/);
  if (m) latentPage(+m[1]); else index();
}
window.addEventListener('hashchange', route);
route();
"""

_GRAPH_CSS = """
body { overflow: hidden; }
.layout { display: grid; grid-template-columns: 1fr 320px; height: 100vh; }
@media (max-width: 860px) { .layout { grid-template-columns: 1fr; grid-template-rows: 1fr 40vh; } }
.main { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.toolbar { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--line);
  background: var(--panel); }
.toolbar h1 { font-size: 16px; margin: 0; } .toolbar label { display: flex; gap: 6px; align-items: center; color: var(--muted); }
.canvas { flex: 1; overflow: auto; min-height: 0; }
svg.graph { display: block; }
.edge { fill: none; } .edge.dim { opacity: .06; }
.node { cursor: pointer; } .node.dim { opacity: .2; }
.node circle, .node rect { stroke: var(--panel); stroke-width: 1.5; }
.node.sel circle, .node.sel rect { stroke: var(--text); stroke-width: 2.5; }
.site-label { fill: var(--muted); font-size: 12px; paint-order: stroke; stroke: var(--bg); stroke-width: 4px; } .tok-label { fill: var(--text); font-size: 12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.gridline { stroke: var(--line); }
.side { border-left: 1px solid var(--line); background: var(--panel); overflow: auto; padding: 14px 16px; }
@media (max-width: 860px) { .side { border-left: 0; border-top: 1px solid var(--line); } }
.side h2 { margin: 14px 0 6px; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
.side .big { font-size: 17px; font-weight: 600; }
.list { margin: 0; padding: 0; list-style: none; } .list li { display: flex; justify-content: space-between; gap: 10px;
  padding: 4px 0; border-bottom: 1px dashed var(--line); font-variant-numeric: tabular-nums; cursor: pointer; }
.list li:hover { color: var(--accent); }
.pos { color: var(--pos); } .neg { color: var(--neg); }
.tip { position: fixed; pointer-events: none; background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  box-shadow: var(--shadow); padding: 7px 10px; font-size: 12.5px; max-width: 320px; display: none; z-index: 10; }
.legend { display: flex; gap: 12px; align-items: center; color: var(--muted); font-size: 12px; }
.sw { display: inline-block; width: 18px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 4px; }
"""

_GRAPH_JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const app = document.getElementById('app');
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = v => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)) ? v.toExponential(2) : (+v).toFixed(3);
const css = getComputedStyle(document.documentElement);
const POS = css.getPropertyValue('--pos').trim(), NEG = css.getPropertyValue('--neg').trim();

const N = D.nodes.length, M = N;  // index M = metric node
const edges = D.edges.map(([s, t, w]) => ({s, t, w}));
D.nodes.forEach((n, i) => { if (n.attribution) edges.push({s: i, t: M, w: n.attribution}); });
const maxW = Math.max(1e-12, ...edges.map(e => Math.abs(e.w)));
const ins = Array.from({length: N + 1}, () => []), outs = Array.from({length: N + 1}, () => []);
edges.forEach((e, k) => { outs[e.s].push(k); ins[e.t].push(k); });

let threshold = 0.02, showErrors = true, selected = null, scrollX = 0;

// layout: rows = sites (first at the bottom), columns = token positions
const positions = D.tokens ? D.tokens.map((_, i) => i) : [...new Set(D.nodes.map(n => n.position))].sort((a, b) => a - b);
const cells = new Map();
D.nodes.forEach((n, i) => { const k = n.site + '|' + n.position; if (!cells.has(k)) cells.set(k, []); cells.get(k).push(i); });
cells.forEach(list => list.sort((a, b) => (D.nodes[a].kind === 'error') - (D.nodes[b].kind === 'error') ||
  Math.abs(D.nodes[b].attribution) - Math.abs(D.nodes[a].attribution)));
const perCol = new Map(positions.map(p => [p, 0]));
cells.forEach((list, k) => { const p = +k.split('|').pop(); perCol.set(p, Math.max(perCol.get(p) || 0, list.length)); });
const STEP = 20, ROW = 92, LEFT = 110, TOP = 70, BOTTOM = 60;
const colX = new Map(); let x = LEFT;
positions.forEach(p => { const k = perCol.get(p) || 0; const w = k ? Math.max(90, k * STEP + 24) : 64; colX.set(p, [x, w]); x += w; });
const width = x + 30, nRows = D.sites.length, height = TOP + (nRows + 1) * ROW + BOTTOM;
const rowY = s => TOP + (nRows - D.sites.indexOf(s)) * ROW;  // first site at the bottom
const xy = new Array(N + 1);
cells.forEach((list, k) => {
  const site = k.slice(0, k.lastIndexOf('|')), p = +k.split('|').pop(), [cx, cw] = colX.get(p);
  const start = cx + cw / 2 - (list.length - 1) * STEP / 2;
  list.forEach((i, j) => { xy[i] = [start + j * STEP, rowY(site)]; });
});
const lastCol = colX.get(positions[positions.length - 1]);
xy[M] = [lastCol[0] + lastCol[1] / 2, TOP - 30];
const rNode = i => i === M ? 13 : 4 + 9 * Math.sqrt(Math.abs(D.nodes[i].attribution) / maxW);

function visible(i) { return i === M || showErrors || D.nodes[i].kind !== 'error'; }

function draw() {
  const lines = [];
  D.sites.forEach(s => { const y = rowY(s);
    lines.push(`<line class="gridline" x1="${LEFT - 10}" x2="${width - 20}" y1="${y}" y2="${y}"/>`,
               `<text class="site-label" x="${LEFT - 16 + scrollX}" y="${y + 4}" text-anchor="end">${esc(s)}</text>`); });
  positions.forEach(p => { const [cx, cw] = colX.get(p), t = D.tokens ? D.tokens[p] : null;
    lines.push(`<text class="tok-label" x="${cx + cw / 2}" y="${height - BOTTOM + 30}" text-anchor="middle">${esc(t !== null && t !== undefined ? JSON.stringify(t).slice(1, -1) : 'pos ' + p)}</text>`); });
  const thr = threshold * maxW;
  const live = selected === null ? null : new Set([...ins[selected], ...outs[selected]]);
  const edgeSvg = edges.map((e, k) => {
    if (Math.abs(e.w) < thr || !visible(e.s) || !visible(e.t)) return '';
    const [x1, y1] = xy[e.s], [x2, y2] = xy[e.t], my = (y1 + y2) / 2;
    const dim = live && !live.has(k) ? ' dim' : '';
    return `<path class="edge${dim}" d="M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}" stroke="${e.w > 0 ? POS : NEG}"
      stroke-width="${(0.6 + 5 * Math.abs(e.w) / maxW).toFixed(2)}" stroke-opacity="${(0.25 + 0.6 * Math.abs(e.w) / maxW).toFixed(2)}"/>`;
  }).join('');
  const near = selected === null ? null : new Set([selected, ...ins[selected].map(k => edges[k].s), ...outs[selected].map(k => edges[k].t)]);
  const nodeSvg = [...D.nodes.keys(), M].map(i => {
    if (!visible(i)) return '';
    const [cx, cy] = xy[i], r = rNode(i), cls = 'node' + (selected === i ? ' sel' : '') + (near && !near.has(i) ? ' dim' : '');
    if (i === M) return `<g class="${cls}" data-i="${i}"><circle cx="${cx}" cy="${cy}" r="${r}" fill="var(--text)"/>
      <text x="${cx - r - 8}" y="${cy + 4}" text-anchor="end" class="tok-label">${esc(D.metric_label)} = ${fmt(D.metric)}</text></g>`;
    const n = D.nodes[i], fill = n.attribution >= 0 ? POS : NEG;
    if (n.kind === 'error') return `<g class="${cls}" data-i="${i}"><rect x="${cx - 4.5}" y="${cy - 4.5}" width="9" height="9" rx="1.5" fill="var(--muted)"/></g>`;
    return `<g class="${cls}" data-i="${i}"><circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}"/></g>`;
  }).join('');
  document.getElementById('svg').innerHTML = lines.join('') + edgeSvg + nodeSvg;
}

function label(i) { return i === M ? D.metric_label : D.nodes[i].label; }

function side() {
  const el = document.getElementById('side');
  if (selected === null) {
    const top = [...D.nodes.keys()].filter(i => D.nodes[i].kind === 'latent').sort((a, b) =>
      Math.abs(D.nodes[b].attribution) - Math.abs(D.nodes[a].attribution)).slice(0, 15);
    el.innerHTML = `<div class="big">${esc(D.metric_label)} = ${fmt(D.metric)}</div>
      <p class="muted">${D.nodes.filter(n => n.kind === 'latent').length} latent and ${D.nodes.filter(n => n.kind === 'error').length} error nodes.
      Click a node to trace its inputs and outputs.</p>
      <h2>Largest attributions</h2><ul class="list">${top.map(i => item(i, D.nodes[i].attribution)).join('')}</ul>`;
  } else {
    const n = selected === M ? null : D.nodes[selected];
    const list = (ks, end) => ks.map(k => [edges[k][end], edges[k].w]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 15);
    el.innerHTML = `<div class="big">${esc(label(selected))}</div>
      ${n ? `<p>${n.desc ? esc(n.desc) + '<br>' : ''}<span class="muted">${n.kind === 'latent' ? 'activation' : 'error norm'}</span> ${fmt(n.activation)}
      · <span class="muted">attribution</span> <span class="${n.attribution >= 0 ? 'pos' : 'neg'}">${fmt(n.attribution)}</span></p>` : ''}
      <h2>Inputs (direct effect)</h2><ul class="list">${list(ins[selected], 's').map(([j, w]) => item(j, w)).join('') || '<li class="muted">none</li>'}</ul>
      <h2>Outputs</h2><ul class="list">${list(outs[selected], 't').map(([j, w]) => item(j, w)).join('') || '<li class="muted">none</li>'}</ul>
      <p><button id="clear">Clear selection</button></p>`;
    document.getElementById('clear').onclick = () => select(null);
  }
  el.querySelectorAll('li[data-i]').forEach(li => li.addEventListener('click', () => select(+li.dataset.i)));
}
function item(i, w) { return `<li data-i="${i}"><span>${esc(label(i))}</span><span class="${w >= 0 ? 'pos' : 'neg'}">${fmt(w)}</span></li>`; }
function select(i) { selected = i; draw(); side(); }

app.innerHTML = `<div class="layout"><div class="main">
  <div class="toolbar"><h1>${esc(D.title)}</h1>
    <label>edges ≥ <input id="thr" type="range" min="0" max="0.5" step="0.005" value="${threshold}"> <span id="thrv" class="num"></span></label>
    <label><input id="err" type="checkbox" checked> error nodes</label>
    <span class="legend"><span><span class="sw" style="background:${POS}"></span>positive</span><span><span class="sw" style="background:${NEG}"></span>negative</span></span>
  </div>
  <div class="canvas"><svg id="svg" class="graph" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"></svg></div>
  </div><aside class="side" id="side"></aside></div><div class="tip" id="tip"></div>`;
const thrEl = document.getElementById('thr'), thrv = document.getElementById('thrv');
const setThr = () => { threshold = +thrEl.value; thrv.textContent = (threshold * 100).toFixed(1) + '% of max'; draw(); };
thrEl.addEventListener('input', setThr);
document.getElementById('err').addEventListener('change', e => { showErrors = e.target.checked; draw(); });
const svg = document.getElementById('svg'), tip = document.getElementById('tip');
svg.addEventListener('click', e => { const g = e.target.closest('.node'); select(g ? +g.dataset.i : null); });
svg.addEventListener('mousemove', e => {
  const g = e.target.closest('.node'); if (!g) { tip.style.display = 'none'; return; }
  const i = +g.dataset.i, n = i === M ? null : D.nodes[i];
  tip.innerHTML = n ? `<b>${esc(n.label)}</b>${n.desc ? '<br>' + esc(n.desc) : ''}<br><span class="muted">${n.kind === 'latent' ? 'act' : 'error norm'}</span> ${fmt(n.activation)} · <span class="muted">attr</span> ${fmt(n.attribution)}`
    : `<b>${esc(D.metric_label)}</b> = ${fmt(D.metric)}`;
  tip.style.display = 'block'; tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 12) + 'px';
});
svg.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
setThr(); side();
const canvas = document.querySelector('.canvas');
canvas.addEventListener('scroll', () => {  // keep the row (site) labels at the visible left edge
  scrollX = canvas.scrollLeft;
  svg.querySelectorAll('.site-label').forEach(t => t.setAttribute('x', LEFT - 16 + scrollX));
});
canvas.scrollLeft = canvas.scrollWidth;  // open at the prediction end (last positions, metric node)
"""
