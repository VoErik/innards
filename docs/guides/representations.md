# Analysing representations

{mod}`innards.analysis` asks three kinds of questions about a model's
internals, and is built so they combine:

- **Geometry**: what the representation looks like ({mod}`~innards.analysis.geometry`,
  {mod}`~innards.analysis.similarity`, and the spline partition in
  {mod}`~innards.analysis.spline`).
- **Causality**: what the model uses ({mod}`~innards.analysis.causal` with
  {mod}`~innards.analysis.metrics`).
- **Interpretation**: what a direction means ({mod}`~innards.analysis.interp`,
  {mod}`~innards.analysis.directions`).

Three conventions make them compose:

1. Representations are tensors `X (n, d)` or tokens `(n, t, d)`, from
   `extract` / `load_acts`.
2. Directions are `V (k, d)`, however they were found: a probe, `pca`, `lda`,
   `inlp`, `relevant_subspace`, or an SAE dictionary (`sae.decoder_weight()`).
3. Causal tools take a model, any iterable of batches, edits from
   {mod}`innards.edits` and a metric, and return one value per example
   (summarise with `.mean()` or `metrics.bootstrap_ci`).

Multi-value results are plain tuples, documented in each docstring.

## Geometry

```python
from innards.analysis import geometry as G, similarity as S

V, ratio = G.pca(X, k=20)
G.effective_rank(X), G.participation_ratio(X), G.spectrum_decay(X)   # linear dimensionality
G.twonn_dim(X), G.local_dim(X).mean()                               # intrinsic dimension
nc1, nc2_norm, nc2_angle, nc3, nc4 = G.neural_collapse(X, y, W=head.weight, b=head.bias)
G.outlier_dims(tokens), G.high_norm_tokens(tokens)                  # massive activations, artifact tokens

S.cka_matrix(acts)                       # layer x layer similarity
S.knn_overlap(acts_a["head"], acts_b["head"], k=10)
```

## The spline partition

Networks with ReLU-like activations cut their input space into regions,
each with one affine map. {mod}`~innards.analysis.spline` reads the
activation codes (which units are on) with nnsight:

```python
from innards.analysis import spline as SP

layers = SP.activation_layers(model)
codes = SP.codes(model, loader, layers)
within, between = SP.code_agreement(codes[layers[-1]], labels)   # does the partition follow the labels?
lc = SP.local_complexity(model, images, layers, radius=0.05)     # boundaries near each input
flips = SP.path_crossings(model, img_a, img_b, layers)           # regions crossed between two images
ids, (u, v) = SP.slice_partition(model, img_a, img_b, img_c, layers[-1:], res=128)
sv = SP.jacobian_spectrum(model, images, "layer4", src="layer2", k=10)
```

For GELU / SiLU models (ViTs, ConvNeXts) codes and Jacobians are
approximations; `SP.is_piecewise_linear(model)` says which case applies.

## Causal effects

```python
from innards.analysis import causal as C, metrics as M
from innards.edits import ablate, steer

C.effect(model, loader, {"blocks.6": ablate(V)}, M.change(M.correct()))        # (n,) accuracy change
C.sweep(model, loader, lambda a: {"blocks.6": steer(v, a)}, [0, 2, 4, 8], M.kl())  # (n, 4)
C.patch_sweep(model, clean, corrupt, ["blocks.2", "blocks.6"], M.change(M.logit("label")), positions=True)
C.attribution_patching(model, clean, corrupt, ["blocks.2", "blocks.6"], M.logit("label"))
C.direction_attribution(model, loader, "blocks.6", V, M.logit("label"))      # (n, k)
V_used, energy = C.relevant_subspace(model, loader, "blocks.6", M.logit("label"), k=10)
```

## Interpretation

```python
from innards.analysis import interp as I

values, examples, positions = I.top_k(tokens, v, k=16)
I.spatial_map(tokens, V, grid, exclude=(0,), size=(224, 224))
I.input_attribution(model, images, "blocks.6", v, method="integrated_gradients")
I.readout_effect(V, head.weight).topk(5)            # classes each direction promotes
I.logit_lens(model, tokens)                          # ViTs: what a block already encodes
I.attention_rollout(I.attention_maps(model, images))[:, 0, 1:]
```

`v` can also be a function of the tokens (`fn(tokens) -> scores`), e.g. an
SAE latent (`innards.sae.latent_fn(sae, 42)`).

## Units over a dataset

{mod}`~innards.analysis.units` answers "what does each unit respond to?" over
a whole dataset, streamed from the model. A unit is one output of an encode
function on a layer's tokens: the neurons or channels themselves (the
default), projections on directions, probe logits or SAE latents.

```python
from innards.analysis import units as U
from innards.analysis.dashboard import image_renderer, save_dashboard

stats, top = U.unit_stats(model, loader, "stages.2", threshold=0.5, per_example=True)  # channels of a CNN
stats, top = U.unit_stats(model, loader, "blocks.6", lambda t: t @ V.T, exclude=(0,))   # directions
U.entries(top, 3)                                    # [(value, example, position), ...], best first

path = U.cache_units(model, loader, "blocks.6", "units/b6", lambda t: t @ V.T, threshold=1.0, exclude=(0,))
U.quantile_examples(path, 3), U.cooccurrence(path, [0, 1, 2])   # no model needed any more
save_dashboard("b6.html", path, image_renderer(images, n_prefix=1), directions=V, readout=head.weight)
```

`per_example=True` ranks examples by their peak token, so the top entries
are distinct images (the top *tokens* of a unit often all come from one
image). A unit is active where `|activation| > threshold`; dense units need a
threshold for frequencies to mean something, and for the cache to stay
sparse.

## Combining them

- **Decodable vs. used.** A probe finds a concept; ablating its direction
  tests whether the model relies on it:
  `W, b, c = D.linear_probe(X, y)`, then `C.effect(model, loader, {L: ablate(W)}, M.change(M.correct()))`.
- **Erasure.** `V, accs = D.inlp(X, y)` removes the concept until it is no
  longer decodable; the effect of `ablate(V)` is what the model loses.
- **Variance vs. importance.** `G.subspace_angles(C.relevant_subspace(...)[0], G.pca(X, k)[0])`:
  are the directions the output depends on the ones with most variance?
- **Partition vs. concepts.** `SP.code_agreement` per layer shows where the
  partition starts to follow the labels; `SP.local_complexity` near
  misclassified inputs or the probe's decision boundary shows where the
  partition is dense.
- **Steering across regions.** Steer along `v` with growing `alpha` and
  measure how many downstream units switch,
  `SP.hamming(SP.codes(model, x, layers), SP.codes(model, x, layers, edits={L: steer(v, a)}))`,
  next to `C.sweep(..., M.flip())`: does the edit move inputs into other
  regions before it changes the prediction?
- **Spread of an edit.** `run(model, x, {L: steer(v, a)}, read=[L + k])` with
  and without the edit, then `S.cka` or `M.rep_shift` on the reads.
- **Meaning of what matters.** Rank directions with
  `C.direction_attribution`, then look at the top ones with `I.top_k` and
  `I.spatial_map`.
- **Extrapolation.** `SP.in_hull(X_train, X_test)` per example, against
  effect sizes or local complexity.
