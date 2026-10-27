# innards

Interpretability for vision (and language) models, built on
[nnsight](https://nnsight.net). Everything is a function over tensors, so the
pieces compose: directions found by a probe, PCA, a gradient analysis or an
SAE can be ablated, steered, attributed, plotted and compared.

- **Model access**: load any timm model wrapped for nnsight, extract
  activations of any layers (tokens, feature maps, channels-last maps) over
  any `DataLoader`, in memory or streamed to disk.
- **Interventions**: steering, ablation, patching and SAE splicing as plain
  edit functions, applied in one nnsight trace.
- **Analysis**: representation geometry and similarity, probes and concept
  erasure, causal effects with bootstrap intervals, activation and
  attribution patching, the spline partition of ReLU networks, and
  interpretation (top examples, spatial maps, integrated gradients, logit
  lens, attention).
- **Figures**: publication-style plots for all of it, from one matplotlib
  style with a palette validated for color-vision deficiencies.
- **Sparse autoencoders**: TopK, BatchTopK, JumpReLU and ReLU SAEs,
  transcoders and cross-layer transcoders; training with literature
  defaults, KL fine-tuning, latent analysis, attribution graphs, HTML
  dashboards and SAELens interop.

```python
from innards import extract, load_model
from innards.analysis import directions, plot
from innards.reduce import token

model, preprocess = load_model("vit_small_patch16_224.augreg_in21k_ft_in1k")
acts, labels = extract(model, loader, [f"blocks.{i}" for i in range(12)], reduce=token(0))
plot.layer_curve({"probe": directions.probe_layers(acts, labels)}, ylabel="held-out accuracy")
```

## Where to start

- [Extracting activations](guides/extraction.md), [Interventions](guides/interventions.md) and
  [Analysing representations](guides/representations.md): the core workflow.
- [Figures](guides/figures.md): the plot functions and the figure style.
- [Getting started with SAEs](getting-started.md): install, then harvest, train, evaluate and save an SAE.
- [Architectures](guides/architectures.md): activations, decoders, loss terms, and adding your own.
- [Training](guides/training.md): defaults, schedules, checkpoints, several layers at once, KL fine-tuning.
- [Transcoders](guides/transcoders.md): transcoders, skip connections and cross-layer transcoders.
- [Analysis](guides/analysis.md): statistics, examples, interventions and dashboards.
- [Circuits](guides/circuits.md): attribution graphs and the graph viewer.
- [SAELens interop](guides/interop.md) and the [API reference](api/index.md).

```{toctree}
:maxdepth: 2
:caption: Guide
:hidden:

getting-started
guides/extraction
guides/interventions
guides/representations
guides/figures
guides/architectures
guides/training
guides/transcoders
guides/analysis
guides/circuits
guides/interop
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api/index
roadmap
```
