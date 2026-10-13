# sae

Sparse autoencoders and transcoders for interpretability research on vision
and language models, in plain PyTorch.

- **Architectures**: TopK, BatchTopK, AbsTopK, JumpReLU and ReLU (L1) SAEs;
  Matryoshka, Multi-TopK and Archetypal variants; transcoders (with skip
  connections) and cross-layer transcoders. One model class, pluggable parts,
  and a registry for your own activations.
- **Training**: literature defaults (learning-rate scaling, warmup/decay,
  AuxK, input normalization, geometric-median init, EMA, k annealing, sparsity
  warmup), several layers from one forward pass, resumable checkpoints, KL
  fine-tuning, and wandb / MLflow / TensorBoard logging.
- **Analysis**: latent statistics, top and quantile examples, dictionary
  geometry, output effects (logit lens), attribution patching, ablation /
  clamping / steering interventions, and loss-recovered metrics.
- **Circuits**: attribution graphs over spliced SAEs, transcoders or
  cross-layer transcoders, with influence-based pruning.
- **Viewing**: self-contained HTML latent dashboards and graph viewers.
- **Interop**: read and write the SAELens format.

```python
from sae import SAE, ActivationStore, SAETrainer, TrainerConfig

store = ActivationStore(model, model.blocks[6], loader, exclude_positions=(0,))
sae = SAE(d_in=768, d_sae=16384, activation="batchtopk", k=32)
SAETrainer(sae, TrainerConfig(total_tokens=50_000_000)).train(store)
sae.save("sae.safetensors")
```

## Where to start

- [Getting started](getting-started.md): install, then harvest, train, evaluate and save an SAE.
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
