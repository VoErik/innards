"""
Sparse autoencoders: train on extracted activations, find what a latent responds to, test what it does.

    python examples/06_sae.py --model vit_tiny_patch16_224.augreg_in21k_ft_in1k
"""

import torch
from common import NAME, figure_path, normalisation, parse, setup, stack

from innards import RowLoader, extract_to_disk, unwrap
from innards.analysis import causal, interp, metrics, plot
from innards.edits import zero
from innards.extract import layer_tokens
from innards.reduce import drop
from innards.sae import SAE, SAETrainer, TrainerConfig, analyze_dataset, encode_inputs, latent_fn
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

# 2. Which latents fire, and on what? (one pass over the data through the live model)
analysis = analyze_dataset(sae, model, L, loader, exclude=prefix)
ax = plot.histogram(analysis.stats.frequency, log10=True, xlabel="log10 firing frequency", ylabel="latents",
                    title="latent density")
plot.save(ax.figure, figure_path(args, "sae_latent_density"))
freq = analysis.stats.frequency
latent = int(torch.where(freq < 0.05, freq, torch.zeros_like(freq)).argmax())   # the most frequent non-dense latent
# top images = highest maximum over each image's patches (top *tokens* can all come from one image)
with torch.no_grad():
    peak = torch.cat([encode_inputs(sae, model, L, x.to(args.device))[:, len(prefix):, latent].amax(1).cpu()
                      for x, _ in loader])
examples = peak.topk(8).indices.tolist()
images, labels = stack(data, examples)
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
