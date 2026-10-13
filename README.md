# SAE

Sparse autoencoders (TopK, BatchTopK, JumpReLU, ReLU + L1; Matryoshka, Archetypal) for vision
and language models.

## Install

```bash
uv sync                       # core: torch, einops, safetensors, numpy
uv sync --extra viz           # + matplotlib plots and HTML views
uv sync --extra notebooks     # everything the example notebooks need
uv sync --extra wandb         # or mlflow / tensorboard, for experiment logging
```

Extras: `wandb`, `mlflow`, `tensorboard`, `viz`, `umap`, `vision` (timm,
torchvision), `language` (transformers, datasets), `notebooks`. The `dev`
group adds pytest and notebook tooling (`uv sync --group dev`).

## Workflow

This assumes you have a *trained* (`torch`) vision or language model:

```python
import torch
from sae import (SAE, ActivationStore, CachedActivations, SAETrainer,
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
from sae import LatentCache, cache_latents
from sae.interventions import clamp, splice_sae, steer, zero_ablate

# Encode a (deterministic) dataset once, then analyse without the model.
cache = LatentCache(cache_latents(sae, model, model.blocks[6], val_loader, "latents/block6",
                                  exclude_positions=(0,)))
stats = cache.stats()                       # frequency, mean / max activation
top = cache.top_examples(n_top=20)          # (value, example, position) per latent
bins = cache.quantile_examples(latent=42)   # samples across activation quantiles
co = cache.cooccurrence([42, 7, 1001])      # pairwise co-firing counts

# Causal edits: the SAE error term is kept, so only the edit changes the model.
with splice_sae(sae, model.blocks[6], zero_ablate([42]), exclude_positions=(0,)):
    ablated_logits = model(images)
with steer(model.blocks[6], sae.decoder_weight()[42], alpha=8.0):
    steered_logits = model(images)

# Downstream fidelity: KL, top-1 agreement and fraction of loss recovered.
trainer.evaluate_downstream(model, model.blocks[6], val_loader, kl_cfg,
                            loss_fn=lambda out, batch: F.cross_entropy(out, batch[1]))
```

`sae.analysis` covers the rest (single-pass `analyze_dataset`, dictionary
geometry, output effects / logit lens, attribution patching, patch maps),
and `sae.viz` plots it.

## Architectures

An `SAE` is assembled from independent parts, chosen in `SAEConfig`:

| Part | Options | Module |
|---|---|---|
| `activation` | `"topk"`, `"batchtopk"` (both with optional `abs_topk`), `"jumprelu"`, `"relu"` (L1) | `sae.activations` |
| `decoder` | `"linear"` (unit-norm atoms), `"archetypal"` | `sae.decoders` |
| loss terms | AuxK, `matryoshka`, `multi_topk` (TopK variants) | `sae.model` |
| target | the input (SAE), or another activation with `transcoder=True` (+ `skip_connection`) | `sae.model` |

```python
SAE(d_in=768, d_sae=16384, activation="topk", k=64, matryoshka=True)
SAE(d_in=768, d_sae=16384, activation="jumprelu", sparsity_coeff=1e-2)  # + TrainerConfig(sparsity_warmup_steps=...)
SAE(d_in=768, d_sae=16384, activation="batchtopk", k=64, decoder="archetypal")
```

To add an architecture, subclass `Activation` and register it; the model,
trainer, saving/loading and analysis code need no changes:

```python
from sae import Activation, register_activation
from sae.activations import dense_code

@register_activation("my_act")
class MyActivation(Activation):
    def forward(self, pre, k):               # (n, d_sae) pre-activations -> SparseCode
        return dense_code(torch.relu(pre))

    def penalty(self, pre, code, decoder_weight):  # optional sparsity loss
        return self.cfg.sparsity_coeff * code.dense.sum(-1).mean()
```

Override `update_stats` for running statistics and `fold_` for thresholds
that must be rescaled when saving. New decoder parametrizations subclass
`Decoder` and go into `sae.decoders.DECODERS`.

## Many layers at once

Harvesting is usually the expensive part, so capture several hook points in
one pass and train one SAE per hook in the same loop:

```python
from sae import SAEGroup

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
from sae import CrossLayerTranscoder, clt_hooks, splice_clt

layers = model.gpt_neox.layers
ins, outs = [l.post_attention_layernorm for l in layers], [l.mlp for l in layers]
store = ActivationStore(model, clt_hooks(ins, outs), loader)       # aligned inputs/outputs of all MLPs
clt = CrossLayerTranscoder(n_layers=len(layers), d_in=512, d_sae=4096, activation="topk", k=32)
trainer = SAETrainer(clt, TrainerConfig(total_tokens=100_000_000))
trainer.train(store)
trainer.evaluate(val_store)                                         # eval/fve, eval/layer3/fve, ...
trainer.evaluate_downstream(model, ins, loader, kl_cfg, target_module=outs, loss_fn=...)
with splice_clt(clt, ins, outs):                                    # run the replacement model
    logits = model(**inputs)
```

## Attribution graphs

Which latents drive an output, and through which other latents? Splice
dictionaries in at several sites; latents and error terms become graph
nodes, edges are direct effects between them (Marks et al., 2024; Ameisen et
al., 2025):

```python
from sae import attribution_graph, clt_site, sae_site

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
from sae.interop import load_saelens, save_saelens

save_saelens(sae, "export/blocks.6", metadata={"model_name": "vit_tiny", "hook_name": "blocks.6"})
gemma_scope_sae, cfg = load_saelens("path/to/saelens/sae")  # then use sae.analysis etc. on it
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

## Tests

```bash
uv run --group dev pytest
```
