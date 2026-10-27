"""
Representation geometry across layers: decodability, dimensionality, similarity, class structure.

    python examples/02_geometry.py --model vit_tiny_patch16_224.augreg_in21k_ft_in1k
"""

import matplotlib.pyplot as plt
from common import IMAGENET_IDX, NAMES, dense, figure_path, parse, setup

from innards import extract, stage_names, unwrap
from innards.analysis import directions, geometry, plot, similarity
from innards.reduce import token

args = parse(__doc__, "vit_tiny_patch16_224.augreg_in21k_ft_in1k")
model, data, loader = setup(args)

layers = stage_names(model)                          # blocks.0 ... blocks.11
acts, labels = extract(model, loader, layers + ["norm"], reduce=token(0))   # CLS token per block
y = dense(labels)
head_features = acts.pop("norm")                     # what the classifier head reads

# How decodable are the classes, and how high-dimensional is the representation, per layer?
probe = directions.probe_layers(acts, y)
dims = {
    "effective rank": {l: geometry.effective_rank(X, center=True) for l, X in acts.items()},
    "participation ratio": {l: geometry.participation_ratio(X) for l, X in acts.items()},
    "TwoNN dimension": {l: geometry.twonn_dim(X) for l, X in acts.items()},
}
short = {l: l.replace("blocks.", "") for l in layers}
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
plot.layer_curve({short[l]: v for l, v in probe.items()}, ax=a, ylabel="held-out accuracy", xlabel="block",
                 reference=1 / len(NAMES), reference_label="chance", title="Linear probe (10 classes)")
plot.layer_curve({k: {short[l]: v for l, v in d.items()} for k, d in dims.items()}, ax=b, ylabel="dimension",
                 xlabel="block", title="Dimensionality")
plot.save(fig, figure_path(args, "geometry_layers"))

# How similar are the layers to each other?
ax = plot.similarity_matrix(similarity.cka_matrix(acts), labels=list(short.values()),
                            title="Linear CKA between blocks")
plot.save(ax.figure, figure_path(args, "geometry_cka"))

# Eigenspectra of an early, a middle and the last block.
picks = [layers[1], layers[len(layers) // 2], layers[-1]]
ax = plot.spectrum({l: acts[l] for l in picks}, title="Covariance spectra")
plot.save(ax.figure, figure_path(args, "geometry_spectra"))

# Class structure at the head: embedding and neural collapse.
ax = plot.embedding(head_features, y, class_names=NAMES, title="Head input (PCA)")
plot.save(ax.figure, figure_path(args, "geometry_embedding"))
head = unwrap(model).head
nc1, nc2_norm, nc2_angle, nc3, nc4 = geometry.neural_collapse(
    head_features, y, W=head.weight[IMAGENET_IDX], b=head.bias[IMAGENET_IDX])
print(f"neural collapse: NC1 {nc1:.3f}  NC2 norms {nc2_norm:.3f} angles {nc2_angle:.3f}  NC3 {nc3:.3f}  NC4 {nc4:.3f}")
print(f"anisotropy at the head: {geometry.anisotropy(head_features):.3f}")
print("probe accuracy:", {short[l]: round(v, 3) for l, v in probe.items()})
