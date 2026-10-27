# Interventions

{mod}`innards.edits` changes activations inside a forward pass with
[nnsight](https://nnsight.net). An edit is a plain function on a layer's tokens
`(batch, tokens, d)` returning tokens of the same shape, so the same edit works
for ViTs, CNNs (`bchw`) and channels-last models (`bhwc`).

## Running with edits

```python
from innards.edits import ablate, run, steer

logits, _ = run(model, images, {"blocks.6": ablate(v)})                     # project out v
logits, reads = run(model, images, {"blocks.6": steer(v, 4.0)}, read=["blocks.9"])
reads["blocks.9"]               # downstream tokens under the edit
```

`run` returns the model output and the tokens of the `read` layers (after any
edit at that layer). Gradients flow through the edits unless you wrap the
call in `torch.no_grad`, so the output can be backpropagated.

## Edits

| Edit | Effect |
|---|---|
| `steer(v, alpha)` | add `alpha * v` |
| `ablate(V)` | project out the span of `V` (`(d,)` or `(k, d)`, any basis) |
| `zero()`, `mean_ablate(mu)` | replace by zero / by `mu` |
| `patch(source, exclude, mask)` | replace by tokens from another run (activation patching) |
| `rows(fn, exclude, mask)` | apply `fn` to some tokens only, e.g. not the CLS token |
| `cross(source, target, fn)` | replace `target`'s tokens by `fn(x, y)` with `x` read at `source` |
| `compose(*edits)` | apply edits in order |

Any `fn(tokens) -> tokens` is an edit too. Activation patching, for example:

```python
from innards.edits import patch

_, clean = run(model, clean_images, read=["layer3"])
one = torch.zeros(clean["layer3"].shape[:2], dtype=torch.bool)
one[:, 10] = True                                   # one spatial position
patched, _ = run(model, corrupted_images, {"layer3": patch(clean["layer3"], mask=one)})
```

## Inside your own trace

{func}`~innards.edits.edit_output` applies an edit inside a trace you write
yourself, so edits mix with anything else nnsight offers:

```python
from innards import wrap
from innards.edits import edit_output

model = wrap(timm_model)
with model.trace(images):
    edit_output(model.blocks[6], ablate(v))
    logits = model.output.save()
```

## SAEs

{func}`innards.sae.interventions.sae_edits` splices an SAE (or transcoder) into a
layer as edits for `run`; see [Analysis](analysis.md#interventions).

## Layer order

nnsight reads layers in execution order. `run` finds that order (and each
layer's layout) with one partial forward pass the first time a set of layers
is used with a model, and caches it ({func}`innards.extract.plan`).
