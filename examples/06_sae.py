"""
Sparse autoencoders: train on extracted activations, find what a latent responds to, test what it does.

    python examples/06_sae.py --model vit_tiny_patch16_224.augreg_in21k_ft_in1k
"""

import torch
from common import NAME, figure_path, normalisation, parse, setup, stack

from innards import RowLoader, extract_to_disk, unwrap
from innards.analysis import causal, interp, metrics, plot
from innards.analysis.dashboard import image_renderer, save_dashboard
from innards.analysis.units import cache_units, entries, unit_stats
from innards.edits import zero
from innards.extract import layer_tokens
from innards.reduce import drop
from innards.sae import SAE, SAETrainer, TrainerConfig, latent_fn
from innards.sae.interventions import sae_edits

args = parse(__doc__, "vit_tiny_patch16_224.augreg_in21k_ft_in1k", n=500)
model, data, loader = setup(args)
vit = unwrap(model)
prefix = range(vit.num_prefix_tokens)
L = "blocks.6"

# 1. Extract patch tokens to disk; the layer folder is a row cache to train on.
run = extract_to_disk(model, loader, L, f"{args.out}/acts_sae", reduce=drop(prefix))
d = vit.embed_dim
sae = SAE(d_in=d, d_sae=8 * d, activation="batchtopk", k=16)
trainer = SAETrainer(sae, TrainerConfig(num_epochs=5, output_dir=f"{args.out}/sae", device=args.device,
                                        warmup_steps=100, log_every=100))
trainer.train(RowLoader(run / L, batch_rows=1024))
print("held-in fve:", round(trainer.evaluate(RowLoader(run / L, batch_rows=1024))["eval/fve"], 3))
sae = trainer.export_model().eval()

# 2. Which latents fire, and on what? One pass over the data through the live model; the latents are
#    units like any other (neurons, directions), and per_example ranks images by their peak patch.
stats, top = unit_stats(model, loader, L, latent_fn(sae), n_top=8, per_example=True, exclude=prefix)
ax = plot.histogram(stats.frequency, log10=True, xlabel="log10 firing frequency", ylabel="latents",
                    title="latent density")
plot.save(ax.figure, figure_path(args, "sae_latent_density"))
latent = int(torch.where(stats.frequency < 0.05, stats.frequency, 0).argmax())   # most frequent non-dense latent
images, labels = stack(data, [e for _, e, _ in entries(top, latent)])
with torch.no_grad():   # the latent as a function of the layer's tokens: any direction tool takes it
    tokens = layer_tokens(model, images.to(args.device), L)
    maps = interp.spatial_map(tokens, latent_fn(sae, latent), tuple(vit.patch_embed.grid_size), exclude=prefix)
fig = plot.images(images, maps=maps.cpu(), titles=[NAME[int(l)] for l in labels],
                  title=f"SAE latent {latent} at {L}: top images", **normalisation(model))
plot.save(fig, figure_path(args, "sae_latent_top_images"))

# 3. What does the latent do? Ablate it (the SAE error is kept, so only the latent changes).
edits = sae_edits(sae, L, zero([latent]), exclude=prefix)
change = causal.effect(model, loader, edits, metrics.change(metrics.logit("label")))
mean, lo, hi = metrics.bootstrap_ci(change)
print(f"ablating latent {latent}: label logit change {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")

# 4. A browsable dashboard of the 100 most frequent latents (cache once, then no model needed).
cache = cache_units(model, loader, L, f"{args.out}/latents_sae", latent_fn(sae), exclude=prefix)
render = image_renderer(lambda e: data[e][0], tuple(vit.patch_embed.grid_size), n_prefix=len(prefix),
                        **normalisation(model))
page = save_dashboard(figure_path(args, "sae_dashboard").with_suffix(".html"), cache, render, per_example=True,
                      directions=sae.decoder_weight(), readout=vit.head.weight, output_labels=lambda i: NAME.get(i, f"class {i}"),
                      noun="latent", max_units=100, n_top=8, n_bins=2, title=f"SAE latents at {L}")
print("dashboard:", page)
