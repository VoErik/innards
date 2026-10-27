# SAE

Sparse autoencoders (TopK, BatchTopK, JumpReLU, ReLU + L1; Matryoshka, Archetypal) for vision
and language models.

Documentation: <https://voerik.github.io/sparse-autoencoder/> (guides and API
reference; build locally with `uv run --group docs sphinx-build docs docs/_build/html`).

## Install

```bash
uv sync                       # core: torch, nnsight, einops, safetensors, numpy
uv sync --extra viz           # + matplotlib plots and HTML views
uv sync --extra notebooks     # everything the example notebooks need
uv sync --extra wandb         # or mlflow / tensorboard, for experiment logging
```

Extras: `wandb`, `mlflow`, `tensorboard`, `viz`, `umap`, `vision` (timm,
torchvision), `language` (transformers, datasets), `notebooks`. The `dev`
group adds pytest and notebook tooling (`uv sync --group dev`).

## Extracting activations

Any timm model, read through [nnsight](https://nnsight.net) (see the
[extraction guide](docs/guides/extraction.md)):

```python
from innards import extract, extract_to_disk, load_acts, load_model
from innards.reduce import token

model, preprocess = load_model("vit_base_patch16_224")
loader = ...  # any DataLoader over preprocess-ed images, e.g. (image, label) batches
acts, labels = extract(model, loader, ["blocks.6", "blocks.11"], reduce=token(0))
extract_to_disk(model, loader, ["blocks.6"], "acts/vit_b_val")   # each layer dir is an SAE cache
acts, labels = load_acts("acts/vit_b_val")
```

## Analysing representations

`innards.analysis` works on extracted activations (tensors) and directions
`V (k, d)`, so its modules combine freely (see the
[guide](docs/guides/representations.md)):

```python
from innards.analysis import causal, directions, geometry, metrics, spline
from innards.edits import ablate

W, b, classes = directions.linear_probe(acts["blocks.6"], labels)      # decodable ...
used = causal.effect(model, loader, {"blocks.6": ablate(W)}, metrics.change(metrics.correct()))  # ... and used?
V_grad, _ = causal.relevant_subspace(model, loader, "blocks.6", metrics.logit("label"), k=10)
geometry.subspace_angles(V_grad, geometry.pca(acts["blocks.6"], k=10)[0])  # variance vs. what matters
spline.local_complexity(model, images, spline.activation_layers(model))   # partition density near data
```

## Workflow

This assumes you have a *trained* (`torch`) vision or language model:

```python
import torch
from innards.sae import (SAE, ActivationStore, CachedActivations, SAETrainer,
                 TrainerConfig, cache_activations)

# 0. Initialize the model and dataloaders etc.
model = ...
loader = ...

# 1. Harvest activations at one layer: hook, drop CLS/registers, shuffle.
store = ActivationStore(
    model=model,
    hook_module=model.blocks[6], # the module the SAE shall be placed in
    input_loader=loader,
    exclude_positions=range(model.num_prefix_tokens),
    device="cuda",
    storage_dtype=torch.bfloat16
)

# 2. Optional: cache to disk.
cache_activations(store, "activations/model_block6")
acts = CachedActivations("activations/model_block6", batch_tokens=4096, device="cuda")

# 3. Train (or pass `store` directly to stream without caching; streaming
#    needs TrainerConfig(total_tokens=...) since a store has no length).
sae = SAE(
    d_in=model.embed_dim,
    d_sae=16 * model.embed_dim, # SAE expansion
    activation="batchtopk",     # see "Architectures" below
    k=32,                       # num active latents / sample
)
trainer = SAETrainer(sae, TrainerConfig(num_epochs=5, loggers=("tensorboard",)))
trainer.train(acts)

# 4. Optional: short KL fine-tune with the SAE spliced into the model.
trainer.finetune_kl(model, model.blocks[6], loader)

# 5. Save / load: one safetensors file, config in its metadata.
sae.save("sae.safetensors")
sae = SAE.load("sae.safetensors", device="cuda")
```

## Analysis and interventions

```python
import torch.nn.functional as F
from innards.sae import LatentCache, cache_latents
from innards.edits import rows, run, steer
from innards.sae.interventions import sae_edits, zero_ablate

# Encode a (deterministic) dataset once, then analyse without the model.
cache = LatentCache(cache_latents(sae, model, model.blocks[6], val_loader, "latents/block6",
                                  exclude_positions=(0,)))
stats = cache.stats()                       # frequency, mean / max activation
top = cache.top_examples(n_top=20)          # (value, example, position) per latent
bins = cache.quantile_examples(latent=42)   # samples across activation quantiles
co = cache.cooccurrence([42, 7, 1001])      # pairwise co-firing counts

# Causal edits through nnsight: the SAE error term is kept, so only the edit changes the model.
ablated_logits, _ = run(model, images, sae_edits(sae, "blocks.6", zero_ablate([42]), exclude_positions=(0,)))
steered_logits, _ = run(model, images, {"blocks.6": rows(steer(sae.decoder_weight()[42], 8.0), exclude=(0,))})

# Downstream fidelity: KL, top-1 agreement and fraction of loss recovered.
trainer.evaluate_downstream(model, model.blocks[6], val_loader, kl_cfg,
                            loss_fn=lambda out, batch: F.cross_entropy(out, batch[1]))
```

`innards.sae.analysis` covers the rest (single-pass `analyze_dataset`, dictionary
geometry, output effects / logit lens, attribution patching, patch maps),
and `innards.sae.viz` plots it.

## Architectures

An `SAE` is assembled from independent parts, chosen in `SAEConfig`:

| Part | Options | Module |
|---|---|---|
| `activation` | `"topk"`, `"batchtopk"` (both with optional `abs_topk`), `"jumprelu"`, `"relu"` (L1) | `innards.sae.activations` |
| `decoder` | `"linear"` (unit-norm atoms), `"archetypal"` | `innards.sae.decoders` |
| loss terms | AuxK, `matryoshka`, `multi_topk` (TopK variants) | `innards.sae.model` |
| target | the input (SAE), or another activation with `transcoder=True` (+ `skip_connection`) | `innards.sae.model` |

```python
SAE(d_in=768, d_sae=16384, activation="topk", k=64, matryoshka=True)
SAE(d_in=768, d_sae=16384, activation="jumprelu", sparsity_coeff=1e-2)  # + TrainerConfig(sparsity_warmup_steps=...)
SAE(d_in=768, d_sae=16384, activation="batchtopk", k=64, decoder="archetypal")
```

To add an architecture, subclass `Activation` and register it; the model,
trainer, saving/loading and analysis code need no changes:

```python
from innards.sae import Activation, register_activation
from innards.sae.activations import dense_code

@register_activation("my_act")
class MyActivation(Activation):
    def forward(self, pre, k):               # (n, d_sae) pre-activations -> SparseCode
        return dense_code(torch.relu(pre))

    def penalty(self, pre, code, decoder_weight):  # optional sparsity loss
        return self.cfg.sparsity_coeff * code.dense.sum(-1).mean()
```

Override `update_stats` for running statistics and `fold_` for thresholds
that must be rescaled when saving. New decoder parametrizations subclass
`Decoder` and go into `innards.sae.decoders.DECODERS`.

## Many layers at once

Harvesting is usually the expensive part, so capture several hook points in
one pass and train one SAE per hook in the same loop:

```python
from innards.sae import SAEGroup

hooks = {f"block{i}": model.blocks[i] for i in range(12)}
store = ActivationStore(model, hooks, loader, exclude_positions=(0,))  # dict batches
group = SAEGroup({name: SAE(d_in=768, d_sae=16384, k=32) for name in hooks})
trainer = SAETrainer(group, TrainerConfig(total_tokens=200_000_000))
trainer.train(store)              # or cache_activations(store, ...) first
trainer.evaluate(val_store)       # eval/block0/fve, ...
group["block3"].save("block3.safetensors")
```

Tokens from all hooks are shuffled jointly, so every batch holds the same
tokens at every layer (cross-layer models rely on this).

## Dashboards

Self-contained HTML files (no server, no network): a latent browser built on
the latent cache, and an interactive viewer for attribution graphs.

```python
from innards.sae.dashboard import image_renderer, save_graph_html, save_latent_dashboard, text_renderer

save_latent_dashboard("dashboard.html", cache, text_renderer(val_tokens),  # or image_renderer(images, n_prefix=1)
                      sae=sae, readout=unembed, output_labels=vocab)       # nearest latents, output effects
save_graph_html(graph.prune(0.8), "graph.html", tokens=prompt_tokens, metric_label='logit(" Paris")')
```

The dashboard lists every latent (searchable, sortable) and shows per latent
its activation histogram, top examples, examples sampled across activation
quantiles, nearest latents by decoder direction and output effects. The
graph viewer draws sites as rows and token positions as columns; filter
edges by strength, hover for details, click a node to trace its inputs and
outputs.

## Transcoders

A transcoder reads one activation and predicts another, e.g. an MLP's output
from its input, so its latents explain what the MLP computes. Pass the
input and output modules; batches become `(input, target)` pairs, and
everything that splices the model takes a `target_module`:

```python
mlp = model.gpt_neox.layers[3].mlp
ln = model.gpt_neox.layers[3].post_attention_layernorm   # the MLP's input
store = ActivationStore(model, ln, loader, target_module=mlp)  # yields (x, y)
tc = SAE(d_in=512, d_sae=8192, activation="topk", k=32, transcoder=True,
         skip_connection=True)                              # optional linear skip
trainer = SAETrainer(tc, TrainerConfig(total_tokens=50_000_000))
trainer.train(store)
trainer.evaluate_downstream(model, ln, loader, kl_cfg, target_module=mlp)
```

## Cross-layer transcoders

A CLT replaces every MLP at once: latents read layer `l`'s MLP input and
write to the MLP outputs of layers `l, l+1, ...` (Ameisen et al., 2025), so
a feature computed once can explain the same output across layers.

```python
from innards.edits import run
from innards.sae import CrossLayerTranscoder, clt_edits, clt_hooks

layers = model.gpt_neox.layers
ins, outs = [l.post_attention_layernorm for l in layers], [l.mlp for l in layers]
store = ActivationStore(model, clt_hooks(ins, outs), loader)       # aligned inputs/outputs of all MLPs
clt = CrossLayerTranscoder(n_layers=len(layers), d_in=512, d_sae=4096, activation="topk", k=32)
trainer = SAETrainer(clt, TrainerConfig(total_tokens=100_000_000))
trainer.train(store)
trainer.evaluate(val_store)                                         # eval/fve, eval/layer3/fve, ...
trainer.evaluate_downstream(model, ins, loader, kl_cfg, target_module=outs, loss_fn=...)
logits, _ = run(model, inputs, clt_edits(clt, ins, outs, model=model))   # the replacement model
```

## Attribution graphs

Which latents drive an output, and through which other latents? Splice
dictionaries in at several sites; latents and error terms become graph
nodes, edges are direct effects between them (Marks et al., 2024; Ameisen et
al., 2025):

```python
from innards.sae import attribution_graph, clt_site, sae_site

sites = [sae_site(tcs[i], ins[i], outs[i], name=f"tc{i}") for i in range(n_layers)]  # transcoders
# or [clt_site(clt, ins, outs)], or residual SAEs: sae_site(sae, model.blocks[i])
graph = attribution_graph(model, sites, input_ids, lambda out, _: out.logits[0, -1, target_id])
graph.top_nodes(10)                 # latents with the largest effect on the logit
pruned = graph.prune(node_fraction=0.8)
pruned.edges(threshold=0.01)        # (source, target, weight), strongest first
json.dump(pruned.to_dict(), f)      # for a graph viewer
```

Where the model is linear between sites (MLPs replaced by transcoders or a
CLT and a linear readout) attributions and edges are exact; through
attention and layer norms they are the local linearization at the input
(Ameisen et al. additionally freeze attention patterns and norms).

## SAELens interop

```python
from innards.sae.interop import load_saelens, save_saelens

save_saelens(sae, "export/blocks.6", metadata={"model_name": "vit_tiny", "hook_name": "blocks.6"})
gemma_scope_sae, cfg = load_saelens("path/to/saelens/sae")  # then use innards.sae.analysis etc. on it
```

Supported: TopK, AbsTopK, BatchTopK (exported as JumpReLU with a constant
threshold), JumpReLU and ReLU SAEs; ReLU / JumpReLU / BatchTopK transcoders
with or without skip connection. `tests/test_interop.py` cross-checks both
directions against the `sae_lens` package when it is installed
(`uv run --group dev --with sae-lens pytest tests/test_interop.py`).

## Notebooks

* `notebooks/vision_vit.ipynb`: ViT-Tiny (timm) on Imagenette.
* `notebooks/language_pythia.ipynb`: Pythia-70m on the Pile (10k documents).

Each runs the full workflow: harvest → cache → train → evaluate → KL
fine-tune → save/load → dictionary analysis → latent dashboards →
attribution.

## Roadmap

Open work (frozen-attention graphs, KL fine-tuning for CLTs, ...) is tracked in
[TODO.md](TODO.md).

## Tests

```bash
uv run --group dev pytest
```
