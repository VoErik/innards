# Architectures

Most SAE variants differ in only a few places. An {class}`~sae.model.SAE` is
assembled from three independent parts, all chosen in
{class}`~sae.model.SAEConfig`:

| Part | Options | Module |
|---|---|---|
| `activation` | `"topk"`, `"batchtopk"` (both with optional `abs_topk`), `"jumprelu"`, `"relu"` | {mod}`sae.activations` |
| `decoder` | `"linear"` (unit-norm atoms), `"archetypal"` | {mod}`sae.decoders` |
| loss terms | AuxK, `matryoshka`, `multi_topk` | {mod}`sae.model` |
| target | the input (SAE), or another activation (`transcoder=True`) | {mod}`sae.model` |

The encoder is $f(x) = \sigma(W_{enc}^\top (x - b_{dec}) + b_{enc})$ with the
pre-encoder bias of Bricken et al. (2023); the decoder is
$\hat x = f(x)\,D + b_{dec}$.

## Activations

`topk`
: The `k` largest pre-activations per token (Makhzani & Frey, 2013; Gao et
  al., 2024). Fixed sparsity, no penalty to tune, few dead latents with AuxK.
  The default choice.

`batchtopk`
: The `k * batch_size` largest activations over the whole batch, so `k` is
  the *average* number of active latents (Bussmann, Leask & Nanda, 2024).
  Tokens that need more latents get more. At inference a learned global
  threshold replaces the batch selection, so a token's code does not depend
  on the rest of the batch.

`abs_topk=True`
: For either TopK variant: select by absolute value and keep the sign, so one
  latent can represent both directions of a feature (Zhu, Khalili & Zhu, 2025).

`jumprelu`
: Per-latent learned thresholds, $f(z) = z\,H(z - \theta)$, trained with
  straight-through pseudo-gradients and an L0 penalty (Rajamanoharan et al.,
  2024). Sparsity is set by `sparsity_coeff`, which you tune for the L0 you
  want; ramp it in with `TrainerConfig(sparsity_warmup_steps=...)`.

`relu`
: ReLU with an L1 penalty weighted by decoder norms (Conerly et al., 2024).

```python
SAE(d_in=768, d_sae=16384, activation="topk", k=64)
SAE(d_in=768, d_sae=16384, activation="batchtopk", k=64, abs_topk=True)
SAE(d_in=768, d_sae=16384, activation="jumprelu", sparsity_coeff=1e-2)
```

```{note}
The JumpReLU defaults (initial threshold 0.1, bandwidth 0.05, in the
model's normalized units) are larger than the paper's 0.001: with
pre-activations of order 1 the paper's values leave the thresholds in the
dense regime for a long time on short runs.
```

## Decoders

`linear`
: Free atoms, projected to unit norm after every step, with the gradient
  component parallel to each atom removed (Gao et al., 2024).

`archetypal`
: Atoms constrained to (a relaxation of) the convex hull of k-means centroids
  of the data, $D = (W C + \Lambda)\,e^s$ with row-stochastic $W$ and
  $\|\Lambda_i\| \le \delta$ (RA-SAE; Fel et al., 2025). Makes dictionaries
  more stable across seeds. `archetypal_n_candidates` controls the number of
  centroids; the initialization sample must contain at least four times as
  many tokens (the trainer takes care of this).

## Loss terms

The objective is the normalized MSE (fraction of variance unexplained) plus,
in training mode:

- **AuxK** (`aux_coeff`, default 1/32 for TopK variants): the top `k_aux`
  *dead* latents reconstruct the residual, which revives them (Gao et al.,
  2024). A latent is dead after `dead_tokens_threshold` tokens without firing.
- **Matryoshka** (`matryoshka=True`, `matryoshka_prefixes`): nested prefixes
  of the dictionary must reconstruct the input on their own, which reduces
  feature absorption (Bussmann et al., 2025; Chanin et al., 2024).
- **Multi-TopK** (`multi_topk=True`): also reconstruct with `4k` latents, so
  the SAE degrades gracefully when evaluated at other `k` (Gao et al., 2024).
- the activation's sparsity penalty (JumpReLU, ReLU), times the warmup scale.

## Units, normalization and saving

Internally the model works on `x * input_scale`, a scalar chosen at
initialization so that $E\|x\| = \sqrt{d_{in}}$ (Conerly et al., 2024); all
public methods take and return activations in input units.
{meth}`~sae.model.SAE.save` folds the scale into the weights (as done for
Gemma Scope), so a saved SAE works on raw activations. Files are a single
safetensors file with the config in its metadata.

## Adding an architecture

A new activation is a subclass of {class}`~sae.activations.Activation`,
registered under a name. The model, trainer, saving / loading and the
analysis code need no changes:

```python
import torch
from sae import Activation, register_activation
from sae.activations import dense_code

@register_activation("my_act")
class MyActivation(Activation):
    def forward(self, pre, k):            # (n, d_sae) pre-activations -> code
        return dense_code(torch.relu(pre))

    def penalty(self, pre, code, decoder_weight):   # optional sparsity loss
        return self.cfg.sparsity_coeff * code.dense.sum(-1).mean()

sae = SAE(d_in=768, d_sae=16384, activation="my_act")
```

Return a sparse code (rows / cols / values, as the TopK activations do) when
only few latents are active; return {func}`~sae.activations.dense_code` when
gradients must reach latents that did not fire (JumpReLU does this). Override
{meth}`~sae.activations.Activation.update_stats` for running statistics and
{meth}`~sae.activations.Activation.fold_` for thresholds that must be
rescaled on save. New decoders subclass {class}`~sae.decoders.Decoder` and
are added to {data}`sae.decoders.DECODERS`.
