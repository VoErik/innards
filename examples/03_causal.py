"""
Causal analysis: steering, ablating what a probe finds, activation patching, and which directions matter.

    python examples/03_causal.py --model vit_tiny_patch16_224.augreg_in21k_ft_in1k
"""

import matplotlib.pyplot as plt
import torch
from common import IMAGENET_IDX, NAMES, dense, figure_path, parse, setup, stack

from innards import extract, unwrap
from innards.analysis import causal, directions, geometry, metrics, plot
from innards.edits import ablate, steer
from innards.reduce import token

args = parse(__doc__, "vit_tiny_patch16_224.augreg_in21k_ft_in1k", n=200)
model, data, loader = setup(args)
L = "blocks.6"
acts, labels = extract(model, loader, L, reduce=token(0))
X, y = acts[L], dense(labels)

# 1. Steering: add a class direction (difference of means, CLS tokens) to every token of block 6.
v = directions.diff_of_means(X, y, positive=0).float()            # "tench" vs. the rest
v = v / v.norm() * X.norm(dim=1).mean()                            # alpha = 1 adds one typical CLS norm
rand = torch.randn_like(v)
rand = rand / rand.norm() * v.norm()
alphas = [0.0, 0.25, 0.5, 1.0, 2.0]
target = metrics.change(metrics.logit(IMAGENET_IDX[0]))            # change of the tench logit
swept = {name: causal.sweep(model, loader, lambda a, d=d: {L: steer(d, a)}, alphas, target)
         for name, d in [("tench direction", v), ("random direction", rand)]}
ax = plot.sweep(swept, alphas, xlabel="steering strength α", ylabel="Δ tench logit", title=f"Steering at {L}")
plot.save(ax.figure, figure_path(args, "causal_steering"))

# 2. Decodable vs. used: erase what linear probes find (INLP) and see whether predictions change.
V, accs = directions.inlp(X, y, n_iter=3)
print(f"INLP: probe accuracy {accs.tolist()} -> {V.shape[0]} directions removed")
random_V = torch.randn_like(V)
for name, U in [("INLP directions", V), ("random directions", random_V)]:
    change = causal.effect(model, loader, {L: ablate(U)}, metrics.change(metrics.correct()))
    mean, lo, hi = metrics.bootstrap_ci(change)
    print(f"ablating {name}: accuracy change {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")

# 3. Activation patching: put clean activations into a run on an image of another class, per position.
targets = torch.tensor(data.dataset.targets)[data.indices]      # dataset classes 0..9
clean = stack(data, (targets == 0).nonzero().squeeze(1)[:4])      # tench
corrupt = stack(data, (targets == 4).nonzero().squeeze(1)[:4])    # church
layers = ["blocks.2", "blocks.6", "blocks.10"]
restored = metrics.change(metrics.logit("label"))                 # how much of the clean label's logit comes back
exact = causal.patch_sweep(model, clean, corrupt, layers, restored, positions=True)
fast = causal.attribution_patching(model, clean, corrupt, layers, metrics.logit("label"))
vit = unwrap(model)
grid, prefix = tuple(vit.patch_embed.grid_size), range(vit.num_prefix_tokens)   # patch grid; CLS (+ registers)
fig = plot.patching(exact, grid=grid, exclude=prefix, label="Δ tench logit", title="Activation patching")
plot.save(fig, figure_path(args, "causal_patching"))
fig = plot.patching(fast, grid=grid, exclude=prefix, label="Δ tench logit (linear estimate)",
                    title="Attribution patching")
plot.save(fig, figure_path(args, "causal_attribution_patching"))

# 4. Variance vs. importance: which directions does the output depend on, and do they carry much variance?
V_used, energy = causal.relevant_subspace(model, loader, L, metrics.logit("label"), k=10)
V_pca, ratio = geometry.pca(X, k=10)
print("principal angles relevant subspace vs. PCA (deg):",
      [round(float(a), 1) for a in torch.rad2deg(geometry.subspace_angles(V_used, V_pca))])
attr = causal.direction_attribution(model, loader, L, V_pca, metrics.logit("label")).mean(0)
ax = plot.scatter(ratio, attr, xlabel="variance explained (CLS)", ylabel="attribution to the label logit",
                  annotate=[f"PC{i + 1}" for i in range(10)], title=f"PCA directions at {L}")
plot.save(ax.figure, figure_path(args, "causal_variance_vs_attribution"))
