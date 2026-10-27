# Figures

{mod}`innards.analysis.plot` draws the results of the analysis tools, and
{mod}`innards.style` gives every figure in innards (including
{mod}`innards.sae.viz`) the same look.

## The style

```python
from innards import style

style.use()                 # globally, for your own figures too
with style.context():       # or for a block
    fig, ax = plt.subplots(figsize=(style.width(1), 2.6))
```

The style file `innards.mplstyle` is set for print:

- single-column figures 3.5 in wide (`style.width(2)` = 7.2 in for the full
  page), 7-8 pt text in one sans-serif family, embedded fonts in PDF/SVG;
- hairline axes on the left and bottom only, solid hairline grids switched
  on per plot, thin lines (1.5 pt) and small markers with a white ring;
- text, axes and grids in ink colors ({data}`innards.style.INK`), never in a
  data color.

Colors carry one job each:

- **Identity** (series, classes): {data}`innards.style.CATEGORICAL`, eight
  hues in a fixed order, validated for color-vision deficiencies on white.
  Assign them in order, not by rank; scatter plots color at most three
  classes and show the rest in gray.
- **Magnitude**: `innards_seq`, one blue hue from near-white to dark.
- **Sign**: `innards_div`, blue - gray - red with lightness-matched arms,
  centred on zero.

## The plots

| Function | For |
|---|---|
| `layer_curve` | a quantity across layers (probe accuracy, dimension, effect), several series, per-example values as mean + 95% bootstrap band |
| `similarity_matrix` | CKA or cosine matrices |
| `spectrum` | eigenspectra on log-log axes with the power-law exponent |
| `sweep` | effects across steering strengths or other swept values |
| `patching` | activation / attribution patching per layer, per position, or as spatial maps |
| `embedding` | PCA / t-SNE of a representation colored by class |
| `distributions` | a per-example quantity across groups (e.g. local complexity of correct vs. wrong) |
| `scatter` | one quantity against another, e.g. variance vs. attribution of directions |
| `partition` | region or decision boundaries on a 2-d input slice, over a fill value |
| `images`, `top_examples`, `overlay_heatmap` | image grids with spatial maps, attributions or attention, and boxed patches |

Every function takes an optional `ax` and returns it (image grids return the
figure), so they combine into multi-panel figures:

```python
fig, (a, b) = plt.subplots(1, 2, figsize=(style.width(2), 2.6))
plot.layer_curve({"probe": probe}, ax=a, ylabel="accuracy", reference=0.1, reference_label="chance")
plot.layer_curve({"effective rank": rank, "TwoNN": twonn}, ax=b, ylabel="dimension")
plot.save(fig, "figures/layers.pdf")       # tight bounding box, 300 dpi, embedded fonts
```

The [examples](https://github.com/VoErik/innards/tree/main/examples) produce
one figure of each kind.
