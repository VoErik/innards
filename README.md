# SAE

TopK / BatchTopK sparse autoencoders for vision and  language models.

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
from sae import (ActivationStore, CachedActivations, SAETrainer, TopKSAE,
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
sae = TopKSAE(
    d_in=model.embed_dim, 
    d_sae=16 * model.embed_dim, # SAE expansion
    k=32, # num active latents / sample
    as_batchtopk=True
)
trainer = SAETrainer(sae, TrainerConfig(num_epochs=5, loggers=("tensorboard",)))
trainer.train(acts)

# 4. Optional: short KL fine-tune with the SAE spliced into the model.
trainer.finetune_kl(model, model.blocks[6], loader)

# 5. Save / load: one safetensors file, config in its metadata.
sae.save("sae.safetensors")
sae = TopKSAE.load("sae.safetensors", device="cuda")
```

Then analyze it with `sae.analysis` (latent statistics, top-activating  examples, dictionary geometry, 
output effects, attributions) and plot with `sae.viz`.

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
