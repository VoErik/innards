"""
Attribution graphs: which latents drive a model output, and through which latents.

The model is run with dictionaries spliced in at one or more *sites*
(:func:`sae_site` for SAEs and transcoders, :func:`clt_site` for a
cross-layer transcoder). At every site the latent activations ``a`` and the
dictionary's error term ``e`` (the part of the activation it does not
explain) become leaves of the autograd graph, so the spliced model computes
exactly the clean output while gradients only flow *between* sites through
the model's own computation (residual stream, attention, skip connections).
This gives (Marks et al., 2024; Ameisen et al., 2025):

* **node attributions** ``a_u * dm/da_u`` (and ``e . dm/de`` for error nodes),
  the first-order effect of removing a node on the metric ``m``;
* **edges** ``a_u * d pre_d / d a_u``: the *direct* effect of node ``u`` on the
  pre-activation of a downstream latent ``d``, i.e. along paths that do not
  pass through other nodes (those are leaves, so they block the gradient).

Where the model is linear between sites (e.g. MLPs replaced by transcoders
and a linear readout), node attributions and edges are exact; elsewhere they
are the local linearization at this input. Ameisen et al. additionally freeze
attention patterns and normalization denominators to make the replacement
model linear; that requires model-specific hooks and is not done here.

Typical usage::

    sites = [sae_site(tc, model.blocks[i].ln2, model.blocks[i].mlp, name=f"tc{i}") for i in range(6)]
    graph = attribution_graph(model, sites, input_ids, lambda out, _: out.logits[0, -1, target_id])
    pruned = graph.prune(node_fraction=0.8)
    for node in pruned.top_nodes(10):
        print(node)

References:
    Marks, S. et al. (2024). Sparse Feature Circuits: Discovering and Editing
        Interpretable Causal Graphs in Language Models. arXiv:2403.19647.
    Dunefsky, J., Chlenski, P. & Nanda, N. (2024). Transcoders Find
        Interpretable LLM Feature Circuits. arXiv:2406.11944.
    Ameisen, E. et al. (2025). Circuit Tracing: Revealing Computational Graphs
        in Language Models. Transformer Circuits Thread.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterator,
    Sequence
)
from contextlib import (
    ExitStack,
    contextmanager
)
from dataclasses import (
    dataclass,
    field
)
from typing import Any

import torch
from einops import einsum
from torch import (
    Tensor,
    nn
)

from .analysis import _frozen
from .clt import CrossLayerTranscoder
from .data import (
    default_input_fn,
    from_tokens,
    replace_activation,
    to_tokens,
    token_keep_mask
)
from .model import SAE


@dataclass
class _LatentRecord:
    site: str
    keep: Tensor  # (batch, tokens) bool
    acts: Tensor  # leaf (n, d_sae)
    pre: Tensor  # (n, d_sae), differentiable w.r.t. upstream leaves


@dataclass
class _ErrorRecord:
    site: str
    keep: Tensor
    error: Tensor  # leaf (n, d)


class _Recorder:
    """Collects the leaves created by the site hooks, in forward order."""

    def __init__(self, layout: str, exclude_positions: Sequence[int], token_mask_fn, batch):
        self.layout, self.exclude, self.mask_fn, self.batch = layout, exclude_positions, token_mask_fn, batch
        self.records: list[_LatentRecord | _ErrorRecord] = []

    def keep(self, tokens: Tensor) -> Tensor:
        return token_keep_mask(tokens, self.exclude, self.mask_fn, self.batch)

    def patch(self, act: Tensor, keep: Tensor, new: Tensor) -> Tensor:
        tokens = to_tokens(act, self.layout)
        patched = tokens.clone()
        patched[keep] = new.to(tokens.dtype)
        return from_tokens(patched, self.layout, act)


@dataclass
class Site:
    """
    Where and how dictionaries are spliced in for :func:`attribution_graph`.

    Build with :func:`sae_site` or :func:`clt_site`.
    """

    name: str
    install: Callable[[_Recorder], Any] = field(repr=False)
    modules: list[nn.Module] = field(default_factory=list, repr=False)


def sae_site(sae: SAE, hook_module: nn.Module, target_module: nn.Module | None = None,
             name: str | None = None) -> Site:
    """
    An SAE (``target_module=None``) or transcoder site.

    An SAE replaces ``hook_module``'s output by ``decode(a) + e``; a
    transcoder reads ``hook_module`` and replaces ``target_module``'s output
    by ``decode(a) + skip(x) + e``. In both cases the latents ``a`` are
    computed from the (spliced) upstream model and become leaves.

    Args:
        sae: The SAE or transcoder.
        hook_module: Module whose output the dictionary reads.
        target_module: For transcoders, the module whose output is replaced.
        name: Site name used in node labels; defaults to the module's class.
    """
    name = name or type(hook_module).__name__

    @contextmanager
    def install(rec: _Recorder) -> Iterator[None]:
        state: dict[str, Tensor] = {}

        def read(act: Tensor) -> Tensor:
            tokens = to_tokens(act, rec.layout)
            keep = rec.keep(tokens)
            x = tokens[keep].to(sae.w_enc.dtype)
            pre = sae.pre_activations(x)
            acts = sae.encode(x).detach().requires_grad_(True)
            rec.records.append(_LatentRecord(name, keep, acts, pre))
            state.update(keep=keep, recon=sae.decode(acts) + sae.skip(x))
            return act

        def write(act: Tensor) -> Tensor:
            keep, recon = state.pop("keep"), state.pop("recon")
            y = to_tokens(act, rec.layout)[keep].to(recon.dtype)
            error = (y - recon).detach().requires_grad_(True)
            rec.records.append(_ErrorRecord(name, keep, error))
            return rec.patch(act, keep, recon + error)

        target = target_module if target_module is not None else hook_module
        with replace_activation(hook_module, read), replace_activation(target, write):
            yield

    return Site(name, install, [sae])


def clt_site(clt: CrossLayerTranscoder, inputs: Sequence[nn.Module], outputs: Sequence[nn.Module],
             name: str = "clt") -> Site:
    """
    A cross-layer transcoder replacing every MLP.

    Latents of layer ``l`` (site name ``"<name>.<l>"``) are leaves computed
    from layer ``l``'s MLP input; MLP ``m``'s output becomes the sum of the
    decoded contributions of layers ``<= m`` plus its error node.

    Args:
        clt: The cross-layer transcoder.
        inputs: Modules whose outputs are the MLP inputs, in layer order.
        outputs: The MLPs, in layer order.
        name: Site name prefix.
    """
    n, d_out = clt.cfg.n_layers, clt.cfg.d_out

    @contextmanager
    def install(rec: _Recorder) -> Iterator[None]:
        state: dict[int, tuple[Tensor, Tensor | float]] = {}

        def read(layer: int):
            def hook(act: Tensor) -> Tensor:
                tokens = to_tokens(act, rec.layout)
                keep = rec.keep(tokens)
                x_s = tokens[keep].to(clt.w_enc.dtype) * clt.input_scale[layer]
                pre = einsum(x_s, clt.w_enc[layer], "b d, d s -> b s") + clt.b_enc[layer]
                code = clt.activations[layer](pre.detach(), clt.k_current)
                acts = CrossLayerTranscoder._dense(code, clt.cfg.d_sae).detach().requires_grad_(True)
                rec.records.append(_LatentRecord(f"{name}.{layer}", keep, acts, pre))
                contrib = (acts @ clt.decoder_weight(layer)).view(-1, n - layer, d_out)
                skip = x_s @ clt.w_skip[layer] if clt.cfg.skip_connection else 0.0
                state[layer] = (contrib, skip)
                return act
            return hook

        def write(m: int):
            def hook(act: Tensor) -> Tensor:
                tokens = to_tokens(act, rec.layout)
                keep = rec.keep(tokens)
                pred = clt.b_dec[m] + sum(state[layer][0][:, m - layer] for layer in range(m + 1)) + state[m][1]
                recon = pred / clt.target_scale[m]
                error = (tokens[keep].to(recon.dtype) - recon).detach().requires_grad_(True)
                rec.records.append(_ErrorRecord(f"{name}.{m}", keep, error))
                return rec.patch(act, keep, recon + error)
            return hook

        with ExitStack() as stack:
            for layer in range(n):
                stack.enter_context(replace_activation(inputs[layer], read(layer)))
                stack.enter_context(replace_activation(outputs[layer], write(layer)))
            yield

    return Site(name, install, [clt])


@dataclass
class Node:
    """
    A node of an :class:`AttributionGraph`.

    Attributes:
        kind: ``"latent"`` or ``"error"``.
        site: Site (layer) name.
        example: Index in the batch.
        position: Token position (counting excluded positions).
        latent: Latent index (``None`` for error nodes).
        activation: Latent activation (error nodes: norm of the error).
        attribution: First-order effect on the metric, ``a * dm/da``.
    """

    kind: str
    site: str
    example: int
    position: int
    latent: int | None
    activation: float
    attribution: float

    def label(self) -> str:
        what = f"#{self.latent}" if self.kind == "latent" else " error"
        return f"{self.site}{what}@{self.position}" + (f"[{self.example}]" if self.example else "")


@dataclass
class AttributionGraph:
    """
    Result of :func:`attribution_graph`.

    Attributes:
        nodes: Latent nodes (the strongest by ``|attribution|``) and all error nodes.
        adjacency: ``(n_nodes, n_nodes)``; ``adjacency[d, u]`` is the direct
            effect of node ``u`` on latent ``d``'s pre-activation (rows of
            error nodes are zero: errors are sources).
        metric: The metric's value.
    """

    nodes: list[Node]
    adjacency: Tensor
    metric: float

    @property
    def attributions(self) -> Tensor:
        """Node attributions to the metric, ``(n_nodes,)``."""
        return torch.tensor([n.attribution for n in self.nodes])

    def influence(self) -> Tensor:
        """
        Total (direct + indirect) influence of every node on the metric.

        Following Ameisen et al. (2025), edges into each node and the metric's
        attributions are taken in absolute value and normalised to sum to 1;
        the influence is the metric's row of ``sum_k A^k = (I - A)^-1``.

        Returns:
            Non-negative influences, ``(n_nodes,)``.
        """
        a = self.adjacency.abs()
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-12)
        direct = self.attributions.abs()
        direct = direct / direct.sum().clamp_min(1e-12)
        eye = torch.eye(len(self.nodes))
        return torch.linalg.solve((eye - a).T, direct)

    def prune(self, node_fraction: float = 0.8) -> "AttributionGraph":
        """
        Keeps the most influential nodes covering ``node_fraction`` of the total influence.

        Args:
            node_fraction: Fraction of the summed influence to keep.

        Returns:
            The subgraph (edges among the kept nodes).
        """
        inf = self.influence()
        order = inf.argsort(descending=True)
        cum = inf[order].cumsum(0) / inf.sum().clamp_min(1e-12)
        n_keep = int((cum < node_fraction).sum()) + 1
        keep = order[:n_keep].sort().values
        return AttributionGraph([self.nodes[i] for i in keep.tolist()], self.adjacency[keep][:, keep], self.metric)

    def top_nodes(self, n: int = 10, kind: str | None = "latent") -> list[Node]:
        """The ``n`` nodes with the largest ``|attribution|`` (optionally of one kind)."""
        nodes = [x for x in self.nodes if kind is None or x.kind == kind]
        return sorted(nodes, key=lambda x: abs(x.attribution), reverse=True)[:n]

    def edges(self, threshold: float = 0.0) -> list[tuple[Node, Node, float]]:
        """
        Edges ``(source, target, weight)`` with ``|weight| > threshold``, strongest first.
        """
        d, u = (self.adjacency.abs() > threshold).nonzero(as_tuple=True)
        out = [(self.nodes[j], self.nodes[i], float(self.adjacency[i, j])) for i, j in zip(d.tolist(), u.tolist())]
        return sorted(out, key=lambda e: abs(e[2]), reverse=True)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (nodes, non-zero edges, metric)."""
        return {
            "metric": self.metric,
            "nodes": [dict(vars(n), label=n.label()) for n in self.nodes],
            "edges": [{"source": i_u, "target": i_d, "weight": float(self.adjacency[i_d, i_u])}
                      for i_d, i_u in self.adjacency.nonzero().tolist()],
        }


def attribution_graph(
    model: nn.Module,
    sites: Sequence[Site],
    batch: Any,
    target_fn: Callable[[Any, Any], Tensor],
    max_nodes: int = 200,
    layout: str = "btd",
    exclude_positions: Sequence[int] = (),
    token_mask_fn: Callable[[Any], Tensor] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
) -> AttributionGraph:
    """
    Builds the attribution graph of a scalar metric over the spliced sites.

    Runs one forward pass, one backward pass from the metric, and one
    backward pass per selected latent node (``max_nodes`` of them) for its
    incoming edges.

    Args:
        model: The model (frozen during the computation).
        sites: Dictionaries to splice in, from :func:`sae_site` / :func:`clt_site`.
        batch: Model inputs, already on the right device (typically one prompt).
        target_fn: ``fn(model_output, batch) -> scalar or (batch,)``; summed.
        max_nodes: Latent nodes kept, by ``|attribution|``. Error nodes are
            always kept.
        layout: Activation layout at the hooks.
        exclude_positions: Positions left unspliced (e.g. BOS / CLS).
        token_mask_fn: Optional ``fn(batch) -> bool (batch, tokens)`` mask.
        input_fn: Maps the batch to model ``(args, kwargs)``.

    Returns:
        An :class:`AttributionGraph`.
    """
    model.eval()
    dictionaries = [m.eval() for site in sites for m in site.modules]
    rec = _Recorder(layout, exclude_positions, token_mask_fn, batch)
    args, kwargs = input_fn(batch)
    with ExitStack() as stack:
        stack.enter_context(_frozen(model, *dictionaries))  # only the leaves need gradients
        for site in sites:
            stack.enter_context(site.install(rec))
        metric = target_fn(model(*args, **kwargs), batch).sum()

        records = rec.records
        leaves = [r.acts if isinstance(r, _LatentRecord) else r.error for r in records]
        grads = torch.autograd.grad(metric, leaves, retain_graph=True, allow_unused=True)

        # node list: latents (top max_nodes by |attribution|) and every error node
        candidates: list[tuple[int, int, int | None, float, float]] = []  # (record, row, latent, act, attr)
        for i, (r, g) in enumerate(zip(records, grads)):
            g = torch.zeros_like(leaves[i]) if g is None else g
            if isinstance(r, _LatentRecord):
                acts = r.acts.detach()
                attr = acts * g
                rows, lats = (acts != 0).nonzero(as_tuple=True)
                for row, lat in zip(rows.tolist(), lats.tolist()):
                    candidates.append((i, row, lat, float(acts[row, lat]), float(attr[row, lat])))
        candidates.sort(key=lambda c: abs(c[4]), reverse=True)
        chosen = candidates[:max_nodes]
        for i, (r, g) in enumerate(zip(records, grads)):
            if isinstance(r, _ErrorRecord):
                g = torch.zeros_like(r.error) if g is None else g
                err = r.error.detach()
                attr = (err * g).sum(-1)
                for row in range(err.shape[0]):
                    chosen.append((i, row, None, float(err[row].norm()), float(attr[row])))
        chosen.sort(key=lambda c: (c[0], c[1], -1 if c[2] is None else c[2]))

        nodes = []
        for i, row, lat, act, attr in chosen:
            r = records[i]
            ex, pos = r.keep.nonzero()[row].tolist()
            nodes.append(Node("latent" if lat is not None else "error", r.site, ex, pos, lat, act, attr))

        # incoming edges of each latent node: one backward pass from its pre-activation
        by_record: dict[int, list[tuple[int, int | None, int]]] = {}
        for k, (i, row, lat, _, _) in enumerate(chosen):
            by_record.setdefault(i, []).append((row, lat, k))
        adjacency = torch.zeros(len(chosen), len(chosen))
        for k, (i, row, lat, _, _) in enumerate(chosen):
            if lat is None or not records[i].pre.requires_grad:  # error nodes and first-site latents are sources
                continue
            upstream = list(range(i))
            gs = torch.autograd.grad(records[i].pre[row, lat], [leaves[j] for j in upstream], retain_graph=True,
                                     allow_unused=True)
            for j, g in zip(upstream, gs):
                if g is None:
                    continue
                for row_u, lat_u, k_u in by_record.get(j, []):
                    if lat_u is None:
                        w = (records[j].error[row_u].detach() * g[row_u]).sum()
                    else:
                        w = records[j].acts[row_u, lat_u].detach() * g[row_u, lat_u]
                    adjacency[k, k_u] = float(w)
    return AttributionGraph(nodes, adjacency, float(metric.detach()))

