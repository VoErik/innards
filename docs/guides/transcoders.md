# Transcoders and cross-layer transcoders

## Transcoders

A transcoder reads one activation and predicts another, typically an MLP's
output from its input (Dunefsky, Chlenski & Nanda, 2024). Its latents then
explain what the MLP *computes*, and replacing the MLP by the transcoder gives
a model whose MLPs are sparse and linear given the active latents, which is
what makes [attribution graphs](circuits.md) clean.

```python
from innards.sae import SAE, ActivationStore, SAETrainer, TrainerConfig

layer = model.gpt_neox.layers[3]
store = ActivationStore(model, layer.post_attention_layernorm, loader,
                        target_module=layer.mlp)            # yields aligned (input, target) pairs
tc = SAE(d_in=512, d_sae=8192, activation="topk", k=32, transcoder=True,
         skip_connection=True)                              # optional linear skip x W_skip
trainer = SAETrainer(tc, TrainerConfig(total_tokens=50_000_000))
trainer.train(store)
trainer.evaluate_downstream(model, layer.post_attention_layernorm, loader, kl_cfg,
                            target_module=layer.mlp)
```

Differences from an SAE:

- `forward(x, target)`; the encoder has no pre-encoder bias, and input and
  target get separate scales (both folded into the weights on save).
- `d_out` may differ from `d_in`.
- `skip_connection=True` adds a linear term $x W_{skip}$, initialized at zero,
  which takes over the linear part of the MLP so the latents can focus on the
  rest (Paulo, Shabalin & Belrose, 2025).
- Everything that splices a dictionary into the model (KL fine-tuning,
  downstream evaluation, attribution, interventions) takes a `target_module`:
  latents are read from `hook_module`, `target_module`'s output is replaced.

## Cross-layer transcoders

A cross-layer transcoder (CLT) replaces all MLPs at once. Latents at layer
$l$ read that layer's MLP input and write to the MLP outputs of layer $l$
*and every later layer* (Lindsey et al., 2024; Ameisen et al., 2025):

$$\hat y_m = \sum_{l \le m} a_l W_{dec}^{l \to m} + b_{dec}^m$$

so a feature computed once can explain the same output across several
layers, which keeps attribution graphs small.

```python
from innards.edits import run
from innards.sae import CrossLayerTranscoder, clt_edits, clt_hooks

layers = model.gpt_neox.layers
ins = [l.post_attention_layernorm for l in layers]
outs = [l.mlp for l in layers]
store = ActivationStore(model, clt_hooks(ins, outs), loader)   # aligned inputs/outputs of every MLP

clt = CrossLayerTranscoder(n_layers=len(layers), d_in=512, d_sae=4096, activation="topk", k=32)
trainer = SAETrainer(clt, TrainerConfig(total_tokens=100_000_000))
trainer.train(store)
trainer.evaluate(val_store)        # eval/fve and eval/layer<i>/fve, eval/layer<i>/l0
trainer.evaluate_downstream(model, ins, val_loader, kl_cfg, target_module=outs, loss_fn=...)

logits, _ = run(model, inputs, clt_edits(clt, ins, outs, model=model))   # the replacement model
```

Any registered activation works (Ameisen et al. use JumpReLU). Each latent's
decoder vectors are unit-norm jointly over all layers it writes to; AuxK is
applied per layer; `clt_edits` can keep each layer's error term and edit
latents (`edit=lambda layer, acts: ...`). The CLT keeps its per-layer scales
as buffers (its latents write to layers with different scales), so its
activations are in normalized units.

```{note}
At initialization the encoder is rescaled to fit the targets. A transcoder's
or CLT's encoder is not tied to its decoder, so the least-squares scale is
close to zero: fine for TopK (selection is unchanged and small initial
activations trained fastest), but it would push every ReLU / JumpReLU
latent below its threshold. Those models therefore match the prediction's
magnitude to the target instead.
```

KL fine-tuning is not implemented for CLTs yet (see the [roadmap](../roadmap.md)).
