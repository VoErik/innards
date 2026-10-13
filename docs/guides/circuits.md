# Circuits: attribution graphs

An attribution graph answers *which latents drive an output, and through which
other latents* (Marks et al., 2024; Ameisen et al., 2025).

## How it works

Dictionaries are spliced into the model at one or more *sites*. At every site
the latent activations $a$ and the error term $e$ (the part of the activation
the dictionary does not explain) become leaves of the autograd graph. The
spliced model computes exactly the clean output, but gradients only flow
*between* sites through the model's own computation (residual stream,
attention, skip connections).

- **Node attributions** $a_u\,\partial m / \partial a_u$ (and
  $e \cdot \partial m / \partial e$ for error nodes): the first-order effect
  on the metric $m$ of removing the node. One backward pass.
- **Edges** $a_u\,\partial \mathrm{pre}_d / \partial a_u$: the *direct*
  effect of node $u$ on the pre-activation of a downstream latent $d$, along
  paths that do not pass through other nodes (those are leaves and block the
  gradient). One backward pass per selected latent.

## Building a graph

```python
from sae import attribution_graph, clt_site, sae_site

# transcoders replacing each MLP
sites = [sae_site(tcs[i], ins[i], outs[i], name=f"tc{i}") for i in range(n_layers)]
# or a cross-layer transcoder:  sites = [clt_site(clt, ins, outs)]
# or residual-stream SAEs:      sites = [sae_site(sae_i, model.blocks[i], name=f"res{i}") ...]

target = tokenizer.encode(" Paris")[0]
graph = attribution_graph(model, sites, input_ids, lambda out, _: out.logits[0, -1, target],
                          max_nodes=200)
graph.top_nodes(10)                    # latents with the largest effect on the logit
pruned = graph.prune(node_fraction=0.8)
pruned.edges(threshold=0.01)           # (source, target, weight), strongest first
```

`max_nodes` latent nodes are kept (by |attribution|), plus all error nodes.
{meth}`~sae.graph.AttributionGraph.influence` combines direct and indirect
effects on the metric (normalized absolute edges, $(I - A)^{-1}$, as in
Ameisen et al.), and {meth}`~sae.graph.AttributionGraph.prune` keeps the
nodes covering a fraction of the total influence.

Residual-stream SAE sites cut the stream, so their graphs only link
consecutive sites. Transcoders and CLTs replace only the MLPs; upstream
information also flows around them through the residual stream and
attention, so edges can skip layers and positions.

## Viewing a graph

```python
from sae.dashboard import save_graph_html

save_graph_html(pruned, "graph.html", tokens=tokenizer.convert_ids_to_tokens(input_ids[0]),
                metric_label='logit(" Paris")',
                descriptions={("tc3", 1234): "capital cities"})
```

The viewer draws sites as rows and token positions as columns, with the
metric node above the predicted position. Filter edges by strength, toggle
error nodes, hover for details, and click a node to trace its strongest
inputs and outputs.

## Exactness and limitations

Where the model is linear between sites (MLPs replaced by transcoders or a
CLT, and a linear readout), node attributions and edges are exact. The tests
check this on such a model: attributions sum to $m - m(\text{all leaves} = 0)$,
and edges between adjacent sites equal the measured change of the downstream
pre-activation when the source is removed.

Through attention and layer norms the graph is the local linearization at
the input. Ameisen et al. additionally freeze attention patterns and
normalization denominators, which makes the replacement model linear; that
needs model-specific hooks and is on the [roadmap](../roadmap.md), together
with a direct check of edges between non-adjacent sites.
