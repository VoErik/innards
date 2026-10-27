"""
Interpretation: what a direction responds to, where in the image, and what it does to the output.

    python examples/05_interpret.py --model vit_tiny_patch16_224.augreg_in21k_ft_in1k
"""

import torch
from common import IMAGENET_IDX, NAME, NAMES, dense, figure_path, normalisation, parse, setup, stack

from innards import capture, extract, stage_names, unwrap
from innards.analysis import directions, interp, plot
from innards.reduce import drop

args = parse(__doc__, "vit_tiny_patch16_224.augreg_in21k_ft_in1k", n=200)
model, data, loader = setup(args)
vit = unwrap(model)
norm = normalisation(model)                                   # to show inputs as images
grid, prefix = tuple(vit.patch_embed.grid_size), vit.num_prefix_tokens
L = "blocks.9"

# A "church" direction: difference of the mean patch token of church images and of the rest.
acts, labels = extract(model, loader, L, reduce=drop(range(prefix)))       # patch tokens only
patches, y = acts[L], dense(labels)
church = NAMES.index("church")
v = directions.diff_of_means(patches.mean(1), y, positive=church).float()

# 1. Which patches respond most? (each image at most once, so the grid shows different images)
values, examples, positions = interp.top_k(patches, v, k=200)
first = {}
for val, ex, pos in zip(values.tolist(), examples.tolist(), positions.tolist()):
    first.setdefault(ex, (val, pos))
top = list(first.items())[:8]
images, labels = stack(data, [ex for ex, _ in top])
fig = plot.top_examples(images, torch.arange(len(top)), torch.tensor([val for _, (val, _) in top]),
                        positions=torch.tensor([pos for _, (_, pos) in top]), grid=grid,
                        names=[NAME[int(l)] for l in labels], title=f"Top patches for the church direction ({L})",
                        **norm)
plot.save(fig, figure_path(args, "interp_top_patches"))

# 2. Where does it fire, where does the CLS token look, and which pixels drive it?
maps = interp.spatial_map(patches[torch.tensor([ex for ex, _ in top])], v, grid)
fig = plot.images(images, maps=maps, title="Church direction, spatial map", **norm)
plot.save(fig, figure_path(args, "interp_spatial_map"))

rollout = interp.attention_rollout(interp.attention_maps(model, images.to(args.device)))
fig = plot.images(images, maps=rollout[:, 0, prefix:].reshape(-1, *grid), title="Attention rollout from CLS",
                  **norm)
plot.save(fig, figure_path(args, "interp_rollout"))

ig = interp.input_attribution(model, images[:4].to(args.device), L, v, method="integrated_gradients",
                              exclude=range(prefix), steps=16)
fig = plot.images(images[:4], maps=ig.sum(1).cpu(), signed=True, title="Integrated gradients of the direction",
                  **norm)
plot.save(fig, figure_path(args, "interp_integrated_gradients"))

# 3. What does each block already encode about the label? (logit lens, accuracy among the 10 classes)
blocks = stage_names(model)
correct = {b: 0 for b in blocks}
for x, labels in loader:
    with torch.no_grad():
        out = capture(model, x.to(args.device), blocks)
    for b in blocks:
        logits = interp.logit_lens(model, out[b])[:, IMAGENET_IDX]
        correct[b] += int((logits.argmax(-1).cpu() == dense(labels)).sum())
ax = plot.layer_curve({b.replace("blocks.", ""): correct[b] / len(data) for b in blocks}, xlabel="block",
                      ylabel="accuracy", reference=1 / len(NAMES), reference_label="chance", title="Logit lens")
plot.save(ax.figure, figure_path(args, "interp_logit_lens"))

# 4. What does the direction do through the head (ignoring the final norm)?
effect = interp.readout_effect(v, vit.head.weight)[0]
print("classes the church direction promotes:", [NAME.get(int(i), int(i)) for i in effect.topk(5).indices])
