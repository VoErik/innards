# innards

Interpretability for vision (and language) models, built on
[nnsight](https://nnsight.net): read and edit any layer of any
[timm](https://github.com/huggingface/pytorch-image-models) model, store
activations, analyse their geometry, causal role and spline partition, and
train sparse autoencoders on them.

Everything is a function over tensors. Representations are `(n, d)` or
`(n, tokens, d)` tensors, sets of directions are `(k, d)`, interventions are
functions on a layer's tokens, so the pieces compose: directions found by a
probe, PCA, a gradient analysis or an SAE can be ablated, steered,
attributed, plotted and compared with each other.

Documentation: <https://voerik.github.io/innards/> (guides and API reference;
build locally with `uv run --group docs sphinx-build docs docs/_build/html`).

## Install

```bash
uv sync                          # core: torch, nnsight, scikit-learn, scipy, einops, safetensors
uv sync --extra vision           # + timm, torchvision (timm models, the examples)
uv sync --extra viz              # + matplotlib (plots, figure style)
uv sync --extra notebooks        # everything the notebooks need
```

Other extras: `wandb`, `mlflow`, `tensorboard` (SAE training logs), `umap`,
`language` (transformers, datasets). The `dev` group adds the test tooling.

## Overview

| Module | What it does |
|---|---|
| `innards.models` | load timm models wrapped for nnsight (`load_model`), find layers (`stage_names`, `module_names`) |
| `innards.extract`, `innards.reduce` | extract activations of any layers over any `DataLoader`, in memory, to disk, or as a shuffled stream of token rows (`RowStream`); CLS token, pooling, token selection |
| `innards.store` | run directories of activations; row caches and `RowLoader` (shuffled token rows, e.g. for SAE training) |
| `innards.edits` | interventions (`steer`, `ablate`, `patch`, `zero`, ...) applied in one nnsight trace (`run`) |
| `innards.analysis.geometry`, `.similarity` | dimensionality, spectra, intrinsic dimension, neural collapse; CKA, RSA, Procrustes, SVCCA, mutual kNN |
| `innards.analysis.directions` | class means, probes, LDA, INLP |
| `innards.analysis.causal`, `.metrics` | effects and sweeps with per-example values and bootstrap CIs, activation / attribution patching, relevant subspaces |
| `innards.analysis.spline` | the spline view of ReLU networks: activation regions, local complexity, partitions on input slices, Jacobian spectra |
| `innards.analysis.interp` | top examples, spatial maps, integrated gradients, logit lens, attention maps and rollout |
| `innards.analysis.units`, `.dashboard` | what each unit (neuron, channel, direction, SAE latent) responds to over a dataset: statistics, top examples, a sparse cache, HTML dashboards |
| `innards.analysis.plot`, `innards.style` | publication-style figures for all of the above, one matplotlib style |
| `innards.sae` | sparse autoencoders, transcoders and cross-layer transcoders: training, analysis, interventions, attribution graphs, dashboards |

## Extract

```python
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from innards import extract, extract_to_disk, load_acts, load_model, stage_names
from innards.reduce import token

model, preprocess = load_model("vit_small_patch16_224.augreg_in21k_ft_in1k")   # nnsight model + eval transform
loader = DataLoader(ImageFolder("imagenette2-160/val", preprocess), batch_size=64)
stage_names(model)                                    # ['blocks.0', ..., 'blocks.11']

acts, labels = extract(model, loader, ["blocks.6", "blocks.11"], reduce=token(0))   # CLS tokens, (n, d)
extract_to_disk(model, loader, "blocks.6", "acts/vit_s_val")                       # all tokens, streamed
acts, labels = load_acts("acts/vit_s_val")                                        # (n, tokens, d)
```

Any timm model works, whatever its layout: tokens (ViT), `bchw` feature maps
(ResNet, ConvNeXt) or channels-last maps (Swin, MambaOut) all come back as
tokens, and `describe` reports each layer's layout and spatial grid.

## Intervene

```python
from innards.edits import ablate, edit_output, patch, rows, run, steer

logits, _ = run(model, images, {"blocks.6": ablate(V)})                 # project out directions
logits, reads = run(model, images, {"blocks.6": steer(v, 4.0)}, read=["blocks.9"])
_, clean = run(model, clean_images, read=["blocks.6"])
patched, _ = run(model, corrupted_images, {"blocks.6": patch(clean["blocks.6"], mask=positions)})

with model.trace(images):                                               # or inside your own trace
    edit_output(model.blocks[6], rows(steer(v), exclude=(0,)))         # leave the CLS token alone
    logits = model.output.save()
```

## Analyse

```python
from innards.analysis import causal, directions, geometry, metrics, similarity, spline

X = acts["blocks.6"][:, 0]                                              # CLS tokens
directions.probe_layers(acts, labels)                                   # held-out probe accuracy per layer
geometry.effective_rank(X), geometry.twonn_dim(X), similarity.cka_matrix(acts)

W, b, classes = directions.linear_probe(X, labels)                      # decodable ...
causal.effect(model, loader, {"blocks.6": ablate(W)}, metrics.change(metrics.correct()))   # ... but used?
V_used, energy = causal.relevant_subspace(model, loader, "blocks.6", metrics.logit("label"), k=10)
geometry.subspace_angles(V_used, geometry.pca(X, k=10)[0])              # variance vs. what the output uses
spline.local_complexity(resnet, images, spline.activation_layers(resnet))   # partition density near inputs
```

The [guide](docs/guides/representations.md) lists the tools and the
combinations they were built for (decodable vs. used, erasure, variance vs.
importance, partitions vs. concepts, steering across regions).

## Plot

```python
from innards import style
from innards.analysis import interp, plot

style.use()                                                             # for your own figures too
ax = plot.layer_curve({"probe": directions.probe_layers(acts, labels)}, ylabel="accuracy", reference=0.1)
plot.similarity_matrix(similarity.cka_matrix(acts), labels=list(acts))
plot.sweep(causal.sweep(model, loader, lambda a: {"blocks.6": steer(v, a)}, alphas, metrics.kl()), alphas)
fig = plot.images(images, maps=interp.spatial_map(tokens, v, grid=(14, 14), exclude=(0,)))
plot.save(fig, "figures/maps.pdf")
```

The style (`src/innards/innards.mplstyle`) is set for print: single-column
3.5 in figures, 7-8 pt text, hairline axes, thin marks, embedded fonts. Its
colors are a palette validated for color-vision deficiencies, assigned in a
fixed order; sequential (`innards_seq`) and diverging (`innards_div`,
lightness-matched arms) colormaps cover magnitudes and signed values.

## Examples

[`examples/`](examples/README.md) has six short scripts on Imagenette with
pretrained timm models: extraction, geometry across layers, causal analysis,
the spline partition of a ResNet, interpretation of a direction, and an SAE
end to end. Each writes its figures in the innards style.

```bash
uv sync --extra vision --extra viz
uv run python examples/03_causal.py
```

## Sparse autoencoders

`innards.sae` trains and analyses SAEs, transcoders and cross-layer
transcoders. Training rows come from the general machinery: a layer of
`extract_to_disk` read by a `RowLoader`, or a `RowStream` from the live model:

```python
from innards.reduce import drop
from innards import RowLoader, RowStream
from innards.sae import SAE, SAETrainer, TrainerConfig

extract_to_disk(model, loader, "blocks.6", "acts/run", reduce=drop([0]))    # patch tokens (no CLS)
sae = SAE(d_in=384, d_sae=16 * 384, activation="batchtopk", k=32)
trainer = SAETrainer(sae, TrainerConfig(num_epochs=5, loggers=("tensorboard",)))
trainer.train(RowLoader("acts/run/blocks.6", batch_rows=4096))
# or stream from the model: trainer.train(RowStream(model, loader, "blocks.6", exclude=(0,)))  (+ total_tokens)

trainer.finetune_kl(model, "blocks.6", loader)                          # optional KL fine-tuning
trainer.evaluate_downstream(model, "blocks.6", val_loader, loss_fn=...) # KL, top-1 agreement, loss recovered
sae.save("sae.safetensors")
```

Analysis and interventions use the same machinery as the rest of innards:

```python
from innards.analysis import units
from innards.analysis.dashboard import image_renderer, save_dashboard
from innards.sae import latent_fn
from innards.edits import zero
from innards.sae.interventions import sae_edits

path = units.cache_units(model, val_loader, "blocks.6", "latents/b6", latent_fn(sae), exclude=(0,))  # latents are units
units.cache_stats(path), units.cache_top(path, n_top=20, per_example=True), units.quantile_examples(path, 42)
save_dashboard("b6.html", path, image_renderer(images, n_prefix=1), directions=sae.decoder_weight(), noun="latent")
logits, _ = run(model, images, sae_edits(sae, "blocks.6", zero([42]), exclude=(0,)))
V = sae.decoder_weight()                    # the dictionary is a set of directions for innards.analysis
```

* **Architectures** (`SAEConfig`): TopK, BatchTopK (both with AbsTopK),
  JumpReLU, ReLU + L1; Matryoshka, Multi-TopK, Archetypal decoders; your own
  activations via `register_activation`.
  ([guide](docs/guides/architectures.md))
* **Many layers at once**: a dict of layers in one `RowStream` and an
  `SAEGroup` train one SAE per layer from a single pass.
  ([guide](docs/guides/training.md))
* **Transcoders and cross-layer transcoders**: `SAE(..., transcoder=True)`
  with a `target` layer; `CrossLayerTranscoder` with `clt_hooks`, and
  `clt_edits` to run the replacement model.
  ([guide](docs/guides/transcoders.md))
* **Attribution graphs** over spliced SAEs, transcoders or CLTs
  (`attribution_graph`, `sae_site`, `clt_site`), with influence-based
  pruning. ([guide](docs/guides/circuits.md))
* **Dashboards**: self-contained HTML latent browsers
  (`innards.analysis.dashboard`, for any units) and attribution-graph viewers
  (`innards.sae.dashboard`). ([guide](docs/guides/analysis.md))
* **SAELens interop**: `save_saelens` / `load_saelens`.
  ([guide](docs/guides/interop.md))

## Notebooks

* `notebooks/vision_vit.ipynb`: an SAE on ViT-Tiny (timm) on Imagenette.
* `notebooks/language_pythia.ipynb`: an SAE on Pythia-70m on the Pile (10k documents).

## Roadmap and tests

Open work is tracked in [TODO.md](TODO.md).

```bash
uv run --group dev pytest
```
