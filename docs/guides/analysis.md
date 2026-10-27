# Analysis

Everything here is modality-agnostic: the same functions work for ViTs, CNNs
and language models (layouts are inferred). Layers are given by name or as
modules.

## Latent statistics and examples

Latents are units like any other, so statistics, top examples, the latent
cache and the dashboards are the general tools of
{mod}`innards.analysis.units` and {mod}`innards.analysis.dashboard` with
{func}`~innards.sae.analysis.latent_fn` as the encode function. One pass over a
(deterministic) loader:

```python
from innards.analysis import units as U
from innards.sae import latent_fn

stats, top = U.unit_stats(model, val_loader, "blocks.6", latent_fn(sae), n_top=20, exclude=(0,))
stats.frequency, (stats.frequency == 0).nonzero()      # firing frequency, dead latents
U.entries(top, latent)                                 # [(value, example index, token position), ...]
U.unit_stats(model, val_loader, "blocks.6", latent_fn(sae), per_example=True)   # top images, not top patches
```

For repeated analysis, **cache the latents once** and query the cache without
running the model again. {func}`~innards.analysis.units.cache_units` stores the
sparse codes and each token's (example, position):

```python
path = U.cache_units(model, val_loader, "blocks.6", "latents/block6", latent_fn(sae), exclude=(0,),
                     metadata={"sae": "block6_16k"})
stats, top = U.cache_stats(path), U.cache_top(path, n_top=20)
bins = U.quantile_examples(path, 42, n_bins=5)      # examples across activation strengths
co = U.cooccurrence(path, [42, 7, 1001])            # pairwise co-firing counts
```

Top examples show a latent's most extreme behaviour; sampling each
activation quantile shows what it responds to at typical strengths, which is
where polysemanticity tends to show up (Bricken et al., 2023).

## Dictionary geometry and output effects

The dictionary is a set of directions, so the general geometry and readout
tools apply to it:

```python
import torch.nn.functional as F
from innards.analysis import geometry, interp, plot

atoms = F.normalize(sae.decoder_weight().detach(), dim=-1)
max_cos, _ = geometry.nearest(atoms, k=1)               # near 1: duplicated / split latents
sims, ids = geometry.nearest(atoms, 42, k=10)           # the latents closest to latent 42
cos = A.encoder_decoder_cosine(sae)                     # low: latents suppressing related ones
plot.embedding(atoms, method="umap", metric="cosine", values=stats.frequency.log10())   # dictionary map
effects = interp.readout_effect(atoms[[42]], model.get_output_embeddings().weight)    # logit lens
top_v, top_i, bot_v, bot_i = interp.top_outputs(effects, n=10)
```

For a vision classifier, pass the classification head as the readout; it is
exact only for the final residual stream.

## Attribution

{func}`~innards.sae.analysis.latent_attribution` splices the SAE in with its error
term kept (so the model is unchanged) and returns, per position and latent,
the attribution-patching estimate of how much a scalar target drops if the
latent is zero-ablated (Nanda, 2023; Syed et al., 2023; Marks et al., 2024).
{func}`~innards.sae.analysis.latent_fn` turns latents into a function of the
layer's tokens, which every tool that scores a direction accepts:
{func}`innards.analysis.interp.spatial_map` for activation maps over the patch
grid, {func}`~innards.analysis.interp.top_k` for top tokens and
{func}`~innards.analysis.interp.input_attribution` (with ``pre=True``) for a
pixel- or token-level saliency map of one latent:

```python
tokens = layer_tokens(model, images, "blocks.6")
maps = interp.spatial_map(tokens, latent_fn(sae, 42), grid=(14, 14), exclude=(0,))
sal = interp.input_attribution(model, images, "blocks.6", latent_fn(sae, 42, pre=True), exclude=(0,))
```

## Interventions

{mod}`innards.sae.interventions` makes causal edits with the error term kept, so only
the edit changes the model. The edits on latents are the general unit edits of
{mod}`innards.edits` (``zero``, ``scale``, ``clamp``, ``mean_ablate``), applied to the
SAE's code instead of the layer's tokens. {func}`~innards.sae.interventions.sae_edits` returns edits
for {func}`innards.edits.run` (see [Interventions](interventions.md)), which applies them
in one nnsight trace:

```python
from innards.edits import clamp, rows, run, steer, zero
from innards.sae.interventions import sae_edits

ablated, _ = run(model, images, sae_edits(sae, "blocks.6", zero([42]), exclude=(0,)))
clamped, _ = run(model, images, sae_edits(sae, "blocks.6", clamp(42, value=5 * stats.max_activation[42])))
steered, _ = run(model, images, {"blocks.6": rows(steer(sae.decoder_weight()[42], 8.0), exclude=(0,))})
```

For a transcoder pass ``target=`` (the layer whose output it replaces).
Modules work in place of layer names when ``model=`` is given, e.g.
``sae_edits(sae, model.blocks[6], model=model)``.

## Dashboards

{mod}`innards.analysis.dashboard` writes self-contained HTML pages for a
cache (no server, no network, light and dark themes):

```python
from innards.analysis.dashboard import image_renderer, save_dashboard, text_renderer

save_dashboard("dashboard.html", path, text_renderer(val_tokens), noun="latent",
               directions=sae.decoder_weight(), readout=unembed, output_labels=vocab)
save_dashboard("vit.html", path, image_renderer(val_images, n_prefix=1, mean=IMAGENET_MEAN,
               std=IMAGENET_STD), noun="latent", per_example=True)
```

The index lists every latent (searchable and sortable, with a firing-frequency
histogram). Each latent's page shows its statistics, activation histogram, top
examples, examples per activation quantile, nearest latents by decoder
direction and output effects (both need `directions=`). A renderer decides
how an example looks: `text_renderer` shades a window of tokens,
`image_renderer` overlays the patch activations on the image, and any
`fn(example, position, token_acts, vmax) -> html` works. Every latent's
examples are embedded, so files grow with `max_units`.
{mod}`innards.sae.dashboard` draws attribution graphs (see [Circuits](circuits.md)).

## Plots

SAE results are plotted with {mod}`innards.analysis.plot`, like everything
else: ``plot.histogram`` for latent density (``log10=True``) and decoder
redundancy, ``plot.embedding`` for dictionary maps, ``plot.similarity_matrix``
(``signed=True``) for atom cosines, ``plot.bars`` for output effects and
attributions, ``plot.images`` for top images with activation maps.
For language models, {mod}`innards.analysis.dashboard` also has inline token
views for notebooks (``text_examples_html``, ``token_activation_html``,
``show_html``).
