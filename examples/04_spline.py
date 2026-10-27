"""
Spline geometry of a ReLU network: activation regions, local complexity, a 2-d slice of the partition.

A ReLU network is piecewise affine: its units cut the input space into regions with one affine map
each. On a ResNet this view is exact (on GELU models such as ViTs it is an approximation).

    python examples/04_spline.py --model resnet18.a1_in1k
"""

import torch
from common import NAMES, dense, figure_path, parse, setup, stack

from innards import unwrap
from innards.analysis import plot, spline

args = parse(__doc__, "resnet18.a1_in1k", n=100)
args.batch_size = min(args.batch_size, 16)
model, data, loader = setup(args)
net = unwrap(model)                                   # the plain timm module, for ordinary forward passes
print("piecewise linear:", spline.is_piecewise_linear(model))

# Activation layers: the last ReLU of every stage.
layers = [l for l in spline.activation_layers(model) if l.endswith(".1.act2")]
codes = spline.codes(model, loader, layers)
labels = torch.cat([y for _, y in loader])
y = dense(labels)

# 1. Does the partition follow the classes? Code distance within vs. between classes, per stage.
agree = {l: spline.code_agreement(codes[l], y) for l in layers}
short = {l: l.split(".")[0] for l in layers}
ax = plot.layer_curve({"within class": {short[l]: w for l, (w, b) in agree.items()},
                       "between classes": {short[l]: b for l, (w, b) in agree.items()}},
                      ylabel="fraction of units that differ", xlabel="stage", title="Activation codes vs. classes")
plot.save(ax.figure, figure_path(args, "spline_codes"))

# 2. Local complexity: how many unit boundaries pass near correctly vs. wrongly classified inputs.
images, labels = stack(data, range(min(len(data), 40)))
with torch.no_grad():
    correct = net(images.to(args.device)).argmax(-1).cpu() == labels
lc = spline.local_complexity(model, images.to(args.device), layers[-1], radius=0.5, n_dirs=4, chunk=32)[layers[-1]]
groups = {"correct": lc[correct], "misclassified": lc[~correct]}
ax = plot.distributions({k: v for k, v in groups.items() if len(v)}, xlabel="units cut near the input",
                        title=f"Local complexity ({short[layers[-1]]})")
plot.save(ax.figure, figure_path(args, "spline_local_complexity"))

# 3. Regions crossed on the straight line between two images, and the local stretching between stages.
a, b = images[:1].to(args.device), images[-1:].to(args.device)
flips = spline.path_crossings(model, a, b, layers, steps=48, chunk=16)
print("units flipping along the line:", {short[l]: int(f.sum()) for l, f in flips.items()})
sv = spline.jacobian_spectrum(model, images[:2].to(args.device), "layer4", src="layer2", k=5)
print("top singular values layer2 -> layer4:", [[round(float(s), 2) for s in row] for row in sv])

# 4. A 2-d slice through three images: decision boundaries (lines) over the density of the partition.
#    A full CNN layer has so many units that every grid point lies in its own region, so the fill shows
#    how many units switch between neighbouring grid points instead of individual regions.
x0, x1, x2 = images[0], images[len(images) // 2], images[-1]
res = 48
t = torch.linspace(-0.25, 1.25, res)
u, v = torch.meshgrid(t, t, indexing="xy")                        # x0 at (0, 0), x1 at (1, 0), x2 at (0, 1)
plane = x0 + u.reshape(-1, 1, 1, 1) * (x1 - x0) + v.reshape(-1, 1, 1, 1) * (x2 - x0)
grid_codes = spline.codes(model, plane.split(32), layers[-1])[layers[-1]]
density = spline.flip_density(grid_codes, res)
with torch.no_grad():
    decision = torch.cat([net(p.to(args.device)).argmax(-1).cpu() for p in plane.split(32)]).reshape(res, res)
names = [NAMES[int(i)] for i in dense(torch.stack([labels[0], labels[len(images) // 2], labels[-1]]))]
ax = plot.partition(decision, u, v, fill=density, fill_label=f"units switching ({short[layers[-1]]})",
                    anchors=names, title="Decision boundaries over partition density")
plot.save(ax.figure, figure_path(args, "spline_slice"))
