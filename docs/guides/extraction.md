# Extracting activations

`innards` reads model internals with [nnsight](https://nnsight.net). Any
timm model works, whatever its activation layout: tokens (ViT, DeiT, EVA),
`bchw` feature maps (ResNet, ConvNeXt, EfficientNet) or channels-last `bhwc`
maps (Swin, MambaOut).

## Load a model and pick layers

```python
from innards import load_model, module_names, stage_names

model, preprocess = load_model("convnext_tiny")   # nnsight model + timm eval transform
stage_names(model)                                # ['stages.0', ..., 'stages.3']
module_names(model, pattern=r"stages\.2\.blocks\.\d+$")
```

Any `nn.Module` can be wrapped with `innards.wrap` instead; every function
also accepts a plain module and wraps it itself.

## Extract

Batches come from any iterable: a `DataLoader`, a list of tensors, or dicts.
`(images, labels)` batches give the targets too.

```python
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from innards import extract
from innards.reduce import mean_pool, token

loader = DataLoader(ImageFolder("imagenet/val", preprocess), batch_size=128, num_workers=8)
acts, labels = extract(model, loader, ["stages.1", "stages.3"], reduce=mean_pool())
acts["stages.3"].shape          # (n_images, 768)
```

Activations come back as tokens `(n, tokens, d)` (a feature map has one
token per spatial position, row-major) or, after a pooling reduction, as
`(n, d)`. Reductions are small functions in `innards.reduce` (`token(0)`,
`keep`, `drop`, `mean_pool`, `max_pool`), or your own `fn(tokens) -> tensor`;
pass a dict to reduce each layer differently.

`describe(model, batch, layers)` reports each layer's layout, shape and
spatial grid. Layouts are inferred from the shapes and the input image size;
pass `layouts=` if a guess is wrong.

## Store and load

For datasets that do not fit in memory, stream to disk:

```python
import torch
from innards import extract_to_disk, load_acts, load_info

extract_to_disk(model, loader, ["stages.1", "stages.3"], "acts/convnext_val",
                dtype=torch.float16, metadata={"model": "convnext_tiny"})
acts, labels = load_acts("acts/convnext_val", layers=["stages.3"])
load_info("acts/convnext_val")["layers"]["stages.3"]   # layout, grid, shapes
```

Rows are stored in dataset order, so do not shuffle the loader. Each layer
directory is also an activation cache for SAE training:

```python
from innards.sae import CachedActivations, SAETrainer

SAETrainer(sae, cfg).train(CachedActivations("acts/convnext_val/stages.3", batch_tokens=4096))
```

Loaded activations are tensors, so `TensorDataset(acts["stages.3"], labels)`
turns them into a `DataLoader` again, e.g. for probes.
