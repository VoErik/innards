# Examples

Small, self-contained scripts on [Imagenette](https://github.com/fastai/imagenette)
(ten ImageNet classes, 160 px, downloaded on first use, ~100 MB) with
ImageNet-pretrained timm models. Each writes its figures to `figures/` in the
innards figure style.

```bash
uv sync --extra vision --extra viz
uv run python examples/02_geometry.py                    # defaults: ViT-Tiny, 500 images
uv run python examples/04_spline.py --model resnet50.a1_in1k --device cuda
uv run python examples/03_causal.py --n 100 --format png
```

| Script | What it shows |
|---|---|
| `01_extract.py` | finding layers and their layouts; extracting in memory and to disk; loading back; the stored layer as an SAE cache |
| `02_geometry.py` | linear probes, effective rank, participation ratio and TwoNN dimension per block; CKA between blocks; eigenspectra; class embedding; neural collapse |
| `03_causal.py` | steering sweep along a class direction vs. a random one; erasing what probes decode (INLP) and its effect; activation vs. attribution patching per position; relevant subspace vs. PCA, variance vs. attribution |
| `04_spline.py` | activation codes vs. classes per stage; local complexity of correct vs. misclassified images; regions crossed between images; Jacobian spectra; decision boundaries over partition density on a slice through three images |
| `05_interpret.py` | top patches of a direction; spatial maps, attention rollout and integrated gradients on images; logit lens per block; classes a direction promotes |
| `06_sae.py` | training an SAE on extracted tokens; latent density; a latent's top images and where it fires; the effect of ablating it |

Common options: `--model` (any timm model; scripts 02, 03, 05 and 06 use ViT
specifics such as the patch grid), `--n` (images), `--batch-size`, `--device`,
`--data`, `--out`, `--format` (`pdf`, `svg`, `png`), and `--random-weights`
for a quick smoke test without downloading weights. `04_spline.py` is the
slowest (the slice needs one forward pass per grid point); all run on a
laptop CPU with the defaults.
