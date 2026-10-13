# topk-sae

TopK / BatchTopK sparse autoencoders built around einops, for vision and
language models. Every training trick is cited in the docstring of the code
that implements it.

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

The model you study stays **frozen**: you take a trained model (your own or a
pretrained one) and train the SAE to explain its activations at one layer.

```python
from sae import (ActivationStore, CachedActivations, SAETrainer, TopKSAE,
                 TrainerConfig, cache_activations)

# 1. Harvest activations at one layer: hook, drop CLS/registers, shuffle.
store = ActivationStore(model, model.blocks[6], image_loader,
                        exclude_positions=range(model.num_prefix_tokens),
                        device="cuda", storage_dtype=torch.bfloat16)

# 2. Optional: cache to disk (worth it for several epochs or sweeps).
cache_activations(store, "activations/vit_block6")
acts = CachedActivations("activations/vit_block6", batch_tokens=4096, device="cuda")

# 3. Train (or pass `store` directly to stream without caching).
sae = TopKSAE(d_in=model.embed_dim, d_sae=16 * model.embed_dim, k=32, as_batchtopk=True)
trainer = SAETrainer(sae, TrainerConfig(num_epochs=5, loggers=("tensorboard",)))
trainer.train(acts)

# 4. Optional: short KL fine-tune with the SAE spliced into the model.
trainer.finetune_kl(model, model.blocks[6], image_loader)

# 5. Save / load: one safetensors file, config in its metadata.
sae.save("sae.safetensors")
sae = TopKSAE.load("sae.safetensors", device="cuda")
```

Then analyse it with `topk_sae.analysis` (latent statistics, top-activating
examples, dictionary geometry, output effects, attributions) and plot with
`topk_sae.viz`.

## Notebooks

* `notebooks/vision_vit.ipynb`: ViT-Tiny (timm) on Imagenette.
* `notebooks/language_pythia.ipynb`: Pythia-70m on the Pile (10k documents).

Each runs the full workflow: harvest → cache → train → evaluate → KL
fine-tune → save/load → dictionary analysis → latent dashboards →
attribution. With `SAE_SMOKE_TEST=1`, every cell runs offline on
random-weight models and fake data (for CI).

The notebooks are paired with `.py` files in jupytext percent format, which
diff cleanly in git; edit either and sync with `jupytext --sync`.

## Tests

```bash
uv run --group dev pytest
```
