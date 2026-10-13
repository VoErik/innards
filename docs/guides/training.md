# Training

{class}`~sae.trainer.SAETrainer` trains any model that follows its small
protocol: single SAEs and transcoders, {class}`~sae.group.SAEGroup` and
{class}`~sae.clt.CrossLayerTranscoder`.

```python
trainer = SAETrainer(sae, TrainerConfig(total_tokens=100_000_000, loggers=("wandb",)))
trainer.train(store_or_cache)
```

## What happens by default

| Setting | Default | Source |
|---|---|---|
| Optimizer | Adam, betas (0.9, 0.999), eps 6.25e-10, no weight decay | Gao et al. (2024) |
| Learning rate | `2e-4 / sqrt(d_sae / 2**14)` | Gao et al. (2024), `sparsify` |
| Schedule | linear warmup (≤ 5% of steps), constant, linear decay over the last 20% | Conerly et al. (2024) |
| Gradient clipping | norm 1.0 | Conerly et al. (2024) |
| Input scale | $E\|x\| = \sqrt{d_{in}}$ | Conerly et al. (2024) |
| Decoder bias init | geometric median of a sample | Gao et al. (2024) |
| Encoder init | transpose of the decoder, rescaled to match the input | Gao et al. (2024) |
| Outlier filter | drop tokens with norm > 10× the median | |
| Mixed precision | bf16 autocast on CUDA | |

Optimizers `"muon"` and `"signum"` are available (as in EleutherAI's
`sparsify`) with their own learning-rate defaults.

## Common settings

`total_tokens` / `num_epochs`
: Stream for a token budget (needed for an {class}`~sae.data.ActivationStore`,
  which has no length) or run whole epochs over a cache or `DataLoader`.

`k_decay_steps`, `k_initial`
: Anneal `k` linearly from a large value (TopK-style activations).

`sparsity_warmup_steps`
: Ramp the JumpReLU / ReLU penalty in from 0.

`ema_decay`
: Keep an exponential moving average of the weights; evaluation and saving
  use the EMA weights.

`loggers`, `project`, `run_name`, `log_every`
: Any of `"wandb"`, `"mlflow"`, `"tensorboard"`; metrics are also printed.
  Logged: loss, normalized MSE, FVE, L0, AuxK loss, dead / dense fractions,
  gradient norm, learning rate, and latent-density histograms.

## Checkpoints and resuming

{meth}`~sae.trainer.SAETrainer.save` writes the exported SAE (input scale
folded, EMA weights if enabled). For resuming, `checkpoint_every` writes a
full training checkpoint (weights, optimizer and EMA state, counters, RNG):

```python
trainer = SAETrainer(sae, TrainerConfig(total_tokens=..., checkpoint_every=5000))
...  # interrupted
trainer = SAETrainer.from_checkpoint("checkpoints/checkpoint.pt")
trainer.train(store)   # continues at the saved step; the LR schedule picks up where it was
```

The data stream starts again from its beginning on resume.

## Several layers from one pass

Harvesting is usually the expensive part. Pass a dict of modules to the store
to capture several hook points in one forward pass (tokens are shuffled
jointly, so every batch holds the same tokens at every hook) and train one
SAE per hook in the same loop:

```python
from sae import SAEGroup

hooks = {f"block{i}": model.blocks[i] for i in range(12)}
store = ActivationStore(model, hooks, loader, exclude_positions=(0,))
group = SAEGroup({name: SAE(d_in=768, d_sae=16384, k=32) for name in hooks})
trainer = SAETrainer(group, TrainerConfig(total_tokens=200_000_000))
trainer.train(store)
trainer.evaluate(val_store)              # eval/block0/fve, eval/block1/fve, ...
group["block3"].save("block3.safetensors")
```

Members train independently (they share no parameters); the learning rate
default follows the largest member.

## Evaluation

`trainer.evaluate(data)`
: Normalized MSE, FVE and L0 over a dataset, computed from global sums (per
  member / per layer for groups and cross-layer transcoders).

`trainer.evaluate_downstream(model, hook_module, loader, kl_cfg, loss_fn=...)`
: Splices the SAE into the model and reports the KL to the clean model,
  top-1 agreement and, with a `loss_fn`, the loss clean / spliced / ablated
  and the fraction of loss recovered,
  $(L_{ablated} - L_{SAE}) / (L_{ablated} - L_{clean})$.

## KL fine-tuning

{meth}`~sae.trainer.SAETrainer.finetune_kl` trains the SAE spliced into the
model on $\tfrac12(L_{SAE} + \alpha\,KL)$ with $\alpha$ balancing the two terms,
which shrinks the downstream loss gap at a few percent of the training budget
(Karvonen, 2025). Configure it with
{class}`~sae.trainer.KLFinetuneConfig` (`output_fn` maps the model output to
logits; `exclude_positions` and `token_mask_fn` should match training).
