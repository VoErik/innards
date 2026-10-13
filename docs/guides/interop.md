# SAELens interop

[SAELens](https://github.com/decoderesearch/SAELens) stores an SAE as a
directory with `cfg.json` and `sae_weights.safetensors`. Exporting lets SAEs
trained here be used with its tooling, Neuronpedia and SAEBench; importing
lets the analysis code here run on published SAEs. The SAELens package itself
is not needed.

```python
from sae.interop import load_saelens, save_saelens

save_saelens(sae, "export/blocks.6", metadata={"model_name": "vit_tiny", "hook_name": "blocks.6"})
sae, cfg = load_saelens("path/to/saelens/sae")   # cfg["metadata"]["hook_name"], ...
```

| Here | SAELens |
|---|---|
| `topk` | `topk` |
| `topk` + `abs_topk` | `abstopk` |
| `batchtopk` | `jumprelu` with a constant threshold (SAELens' own convention) |
| `jumprelu` | `jumprelu` |
| `relu` | `standard` |
| `relu` / `jumprelu` / `batchtopk` transcoders | `transcoder`, `skip_transcoder`, `jumprelu_transcoder`, `jumprelu_skip_transcoder` |

TopK transcoders have no SAELens equivalent. On import,
`apply_b_dec_to_input=False` is absorbed into the encoder bias, and ReLU /
JumpReLU decoders are rescaled to unit norm with the encoder and thresholds
compensating (as SAELens' `fold_W_dec_norm`); input normalization and
`rescale_acts_by_decoder_norm` are not supported.

`tests/test_interop.py` cross-checks both directions against the `sae_lens`
package when it is installed:

```bash
uv run --group dev --with sae-lens pytest tests/test_interop.py
```
