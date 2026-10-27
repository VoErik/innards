# TODO

Open work, roughly in priority order. Each item says what is missing, why it
matters, and where it would go.

## Attribution graphs

- **Freeze attention patterns and normalization scales (`sae/graph.py`).**
  Graphs are currently the local linearization of the model at the input:
  gradients between sites flow through softmax attention and layer norms.
  Ameisen et al. (2025) instead freeze attention patterns and LayerNorm
  denominators (computed on the clean forward pass), which makes the
  replacement model linear between sites, so edges become exact and
  input-independent decompositions. Needs model-specific hooks (attention
  probabilities, norm scales), e.g. adapters for GPT-NeoX / Llama / ViT blocks
  that recompute those modules with the frozen quantities detached.

- **Verify edges between non-adjacent sites directly (`tests/test_graph.py`).**
  Edges are direct effects with every intermediate latent and error held
  fixed. The tests check them by ablating the source and re-running the
  model, which recomputes intermediate sites, so that check is only valid for
  adjacent sites. Non-adjacent edges (e.g. a CLT latent writing two layers
  down) are covered only indirectly, by the completeness check (attributions
  sum to the metric difference). A direct check needs a "frozen replay": a
  splice mode that re-runs the model with intermediate latents and errors
  pinned to their clean values.

## Cross-layer transcoders

- **KL fine-tuning for CLTs (`sae/trainer.py`, `sae/clt.py`).**
  `SAETrainer.finetune_kl` supports single SAEs and transcoders only. For a
  CLT, the patched pass must splice all layers at once (as `splice_clt` does)
  while collecting every layer's inputs and outputs to compute the CLT loss
  next to the KL term.

## Evaluation (deferred)

- **Auto-interpretation.** LLM-written explanations of latents scored by
  detection / fuzzing (as in EleutherAI's `delphi`), built on the latent cache
  (`LatentCache.top_examples`, `quantile_examples`).
- **SAEBench runs.** Export with `innards.sae.interop.save_saelens` and run SAEBench's
  absorption, sparse probing, RAVEL, etc.; document the setup.
