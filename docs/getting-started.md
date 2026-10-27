# Getting started

## Install

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                    # core: torch, nnsight, einops, safetensors, numpy
uv sync --extra viz        # + matplotlib plots
uv sync --extra notebooks  # everything the example notebooks need
uv sync --extra wandb      # or mlflow / tensorboard, for experiment logging
```

Extras: `wandb`, `mlflow`, `tensorboard`, `viz`, `umap`, `vision` (timm,
torchvision), `language` (transformers, datasets), `notebooks`. The `dev`
group adds the test tooling, the `docs` group this documentation.

## The workflow

You need a trained PyTorch model and a loader of its inputs. Everything below
works the same for vision transformers, CNNs and language models (layouts are
inferred); only the layer and the tokens to exclude change.

### 1. Token rows to train on

{class}`~innards.extract.RowStream` runs the frozen model, reads one layer
with nnsight, drops excluded tokens and serves shuffled batches of token
rows from a large buffer:

```python
import torch
from innards import RowStream

stream = RowStream(
    model,
    loader,
    "blocks.6",                                   # the SAE reads this layer's output
    exclude=range(model.num_prefix_tokens),       # e.g. CLS / registers, or BOS
    dtype=torch.bfloat16,
)
```

For language models, pass `mask_fn=lambda batch: batch["attention_mask"]`
(or a mask from token ids) to skip padding and special tokens; the forward
pass stops right after the layer.

### 2. Optionally store them

Streaming re-runs the model every epoch. When the same activations are used
more than once (several epochs, sweeps, several SAEs on one layer), write
them to a row cache and read it with {class}`~innards.store.RowLoader`:

```python
from innards import RowLoader, write_rows

write_rows(stream, "activations/block6", dtype=torch.bfloat16, metadata={"layer": "blocks.6"})
acts = RowLoader("activations/block6", batch_rows=4096, device="cuda")
```

A layer folder of {func}`~innards.extract.extract_to_disk` is a row cache
too (in example order), so extracted activations train directly.

### 3. Train

```python
from innards.sae import SAE, SAETrainer, TrainerConfig

sae = SAE(d_in=768, d_sae=16 * 768, activation="batchtopk", k=32)
trainer = SAETrainer(sae, TrainerConfig(num_epochs=5, loggers=("tensorboard",)))
trainer.train(acts)        # or trainer.train(stream) with TrainerConfig(total_tokens=...)
print(trainer.evaluate(val_acts))   # {"eval/nmse": ..., "eval/fve": ..., "eval/l0": ...}
```

The trainer initializes the SAE from data (input scale, geometric-median
decoder bias, encoder scale), uses the learning rate and schedule from the
literature unless you override them, and writes `checkpoints/sae.safetensors`
at the end. See [Training](guides/training.md) for the knobs.

### 4. Check the model still works with the SAE spliced in

```python
import torch.nn.functional as F
from innards.sae import KLFinetuneConfig

def next_token_loss(out, batch):        # e.g. for a Hugging Face causal LM
    logits, ids = out.logits[:, :-1], batch["input_ids"][:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1))

kl_cfg = KLFinetuneConfig(exclude=(0,), output_fn=lambda out: out.logits)
trainer.evaluate_downstream(model, "blocks.6", val_loader, kl_cfg, loss_fn=next_token_loss)
# {"downstream/kl", "downstream/top1_agreement", "downstream/loss_recovered", ...}
trainer.finetune_kl(model, "blocks.6", train_loader, kl_cfg)  # optional, a few % of the budget
```

### 5. Save, load, analyse

```python
sae.save("sae.safetensors")             # input scale folded into the weights
sae = SAE.load("sae.safetensors", device="cuda")
acts = sae.encode(x)                    # (..., d_sae) in input units
```

From here, [Analysis](guides/analysis.md) covers latent statistics, examples,
interventions and dashboards, and [Circuits](guides/circuits.md) attribution
graphs.

## Example notebooks

- `notebooks/vision_vit.ipynb`: ViT-Tiny (timm) on Imagenette.
- `notebooks/language_pythia.ipynb`: Pythia-70m on the Pile (10k documents).

Each runs the full workflow: harvest, cache, train, evaluate, KL fine-tune,
save / load, dictionary analysis, latent dashboards and attribution.

## Running the tests

```bash
uv run --group dev pytest
uv run --group dev --with sae-lens pytest tests/test_interop.py   # cross-check against SAELens
```
