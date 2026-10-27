"""
Extracting activations: find layers, extract them in memory or to disk, load them back.

    python examples/01_extract.py --model resnet18.a1_in1k
    python examples/01_extract.py --model vit_tiny_patch16_224.augreg_in21k_ft_in1k
"""

import torch
from common import parse, setup

from innards import describe, extract, extract_to_disk, load_acts, load_info, stage_names
from innards.reduce import mean_pool, token
from innards.sae import CachedActivations

args = parse(__doc__, "vit_tiny_patch16_224.augreg_in21k_ft_in1k", n=200)
model, data, loader = setup(args)

# 1. Which layers? timm's feature stages are a good default for any architecture.
layers = stage_names(model)
images, labels = next(iter(loader))
info = describe(model, images.to(args.device), layers)
for name, i in info.items():
    print(f"{name:12s} layout={i['layout']:5s} shape={i['shape']} grid={i['grid']}")

# 2. In memory: one vector per image (the CLS token for ViTs, the spatial mean for CNNs).
is_vit = info[layers[-1]]["layout"] == "btd"
acts, targets = extract(model, loader, layers[-3:], reduce=token(0) if is_vit else mean_pool())
print({k: tuple(v.shape) for k, v in acts.items()}, "targets", tuple(targets.shape))

# 3. To disk: all tokens of one layer, streamed in shards (here in float16).
run = extract_to_disk(model, loader, layers[-2], f"{args.out}/acts", dtype=torch.float16,
                      metadata={"model": args.model, "dataset": "imagenette-val"})
info = load_info(run)
print("stored:", info["n_examples"], "examples,", info["layers"][layers[-2]])

# 4. Load back, as tensors (n, tokens, d) - or the layer folder is directly an SAE training cache.
acts, targets = load_acts(run)
print("loaded:", tuple(acts[layers[-2]].shape), "labels:", targets[:5].tolist())
batch = next(iter(CachedActivations(run / layers[-2], batch_tokens=1024)))
print("as an SAE cache:", tuple(batch.shape))
