# Analysis

Everything here is modality-agnostic: the same functions work for ViTs, CNNs
(`layout="bchw"`) and language models.

## Latent statistics and examples

For a single pass over a (deterministic) loader,
{func}`~sae.analysis.analyze_dataset` returns per-latent statistics and the
top-activating tokens:

```python
from sae import analysis as A

res = A.analyze_dataset(sae, model, model.blocks[6], val_loader, n_top=20, exclude_positions=(0,))
res.stats.frequency, res.stats.dead, res.stats.dense(0.1)
res.top.unique_examples(latent)          # [(value, example index, token position), ...]
```

For repeated analysis, **cache the latents once** and query the cache without
running the model again. {func}`~sae.latent_cache.cache_latents` stores the
sparse codes and each token's (example, position):

```python
from sae import LatentCache, cache_latents

cache = LatentCache(cache_latents(sae, model, model.blocks[6], val_loader, "latents/block6",
                                  exclude_positions=(0,)))
stats = cache.stats()
top = cache.top_examples(n_top=20)
bins = cache.quantile_examples(latent=42, n_bins=5)   # examples across activation strengths
co = cache.cooccurrence([42, 7, 1001])                # pairwise co-firing counts
```

Top examples show a latent's most extreme behaviour; sampling each
activation quantile shows what it responds to at typical strengths, which is
where polysemanticity tends to show up (Bricken et al., 2023).

## Dictionary geometry and output effects

```python
max_cos, _ = A.max_cosine_similarity(sae)          # near 1: duplicated / split latents
sims, ids = A.nearest_latents(sae, latent=42)
cos = A.encoder_decoder_cosine(sae)                # low: latents suppressing related ones
coords = A.dictionary_embedding(sae, "umap")       # or "pca"
eff = A.output_effects(sae, model.get_output_embeddings().weight, latents=[42], n=10)  # logit lens
```

For a vision classifier, pass the classification head as the readout; it is
exact only for the final residual stream.

## Attribution

{func}`~sae.analysis.latent_attribution` splices the SAE in with its error
term kept (so the model is unchanged) and returns, per position and latent,
the attribution-patching estimate of how much a scalar target drops if the
latent is zero-ablated (Nanda, 2023; Syed et al., 2023; Marks et al., 2024).
{func}`~sae.analysis.input_attribution` gives a pixel- or token-level saliency
map of one latent, and {func}`~sae.analysis.latent_patch_maps` activation maps
over the patch grid.

## Interventions

{mod}`sae.interventions` makes causal edits with the error term kept, so only
the edit changes the model:

```python
from sae.interventions import clamp, mean_ablate, scale, splice_sae, steer, zero_ablate

with splice_sae(sae, model.blocks[6], zero_ablate([42]), exclude_positions=(0,)):
    ablated = model(images)
with splice_sae(sae, model.blocks[6], clamp(42, value=5 * stats.max_activation[42])):
    clamped = model(images)
with steer(model.blocks[6], sae.decoder_weight()[42], alpha=8.0):   # no SAE needed; works in generation
    out = model.generate(**inputs)
```

## Dashboards

{mod}`sae.dashboard` writes self-contained HTML pages (no server, no network,
light and dark themes):

```python
from sae.dashboard import image_renderer, save_latent_dashboard, text_renderer

save_latent_dashboard("dashboard.html", cache, text_renderer(val_tokens),
                      sae=sae, readout=unembed, output_labels=vocab)
save_latent_dashboard("vit.html", cache, image_renderer(val_images, n_prefix=1,
                      mean=IMAGENET_MEAN, std=IMAGENET_STD))
```

The index lists every latent (searchable and sortable, with a firing-frequency
histogram). Each latent's page shows its statistics, activation histogram, top
examples, examples per activation quantile, nearest latents by decoder
direction and output effects. A renderer decides how an example looks:
`text_renderer` shades a window of tokens, `image_renderer` overlays the
patch activations on the image, and any `fn(example, position, token_acts,
vmax) -> html` works. Every latent's examples are embedded, so files grow
with `max_latents`.

## Plots

{mod}`sae.viz` (needs the `viz` extra) draws latent density, max cosine
similarity, dictionary maps, similarity matrices, output effects,
attribution bars, image heatmap overlays and top-image grids on matplotlib
axes, and HTML token views for notebooks.
