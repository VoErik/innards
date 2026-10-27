"""
Attribution-graph viewer: a self-contained, interactive HTML page.

:func:`save_graph_html` draws an :class:`~innards.sae.graph.AttributionGraph`
with sites as rows and token positions as columns; edges can be filtered by
strength and a clicked node shows its strongest inputs and outputs. The page
has inline CSS, JavaScript and data and follows the system light / dark
preference. Latent dashboards are general: :mod:`innards.analysis.dashboard`.

Typical usage::

    save_graph_html(graph.prune(0.8), "graph.html", tokens=prompt_tokens)

References:
    Ameisen, E. et al. (2025). Circuit Tracing: Revealing Computational Graphs
        in Language Models. Transformer Circuits Thread.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Sequence
)
from pathlib import Path
from typing import Any

from ..analysis.dashboard import page
from .graph import (
    AttributionGraph,
    Node
)


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
    return page(title, _GRAPH_CSS, _GRAPH_JS, payload)


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
