"""
Analysis of trained SAEs: latents as units, and attributions to latents.

Everything that is not specific to SAEs lives in :mod:`innards.analysis` and
takes the dictionary as directions (``sae.decoder_weight()``) or the latents
as an encode function (:func:`latent_fn`):

* statistics and top examples over a dataset: ``units.unit_stats``,
  ``units.cache_units`` (+ ``cache_top``, ``quantile_examples``,
  ``cooccurrence``) and ``dashboard.save_dashboard``;
* dictionary geometry: ``geometry.nearest`` (redundant latents, neighbours),
  ``geometry.cosine_matrix``, ``geometry.embed``;
* output effects: ``interp.readout_effect`` and ``interp.top_outputs``;
* where latents fire and what drives them: ``interp.spatial_map``,
  ``interp.top_k``, ``interp.input_attribution`` with ``latent_fn``.

Everything here is modality-agnostic. Activations are read with nnsight
(:func:`innards.extract.capture_tokens`, layouts inferred), so the same
functions work for ViTs, CNNs and language models.

References:
    nostalgebraist (2020). interpreting GPT: the logit lens. LessWrong.
    Bricken, T. et al. (2023). Towards Monosemanticity: Decomposing Language
        Models With Dictionary Learning. Transformer Circuits Thread.
        (Feature dashboards; latent effects on logits via the unembedding.)
    Nanda, N. (2023). Attribution Patching: Activation Patching At Industrial
        Scale. Blog post, neelnanda.io.
    Syed, A., Rager, C. & Conmy, A. (2023). Attribution Patching Outperforms
        Automated Circuit Discovery. arXiv:2310.10348.
    Marks, S. et al. (2024). Sparse Feature Circuits: Discovering and Editing
        Interpretable Causal Graphs in Language Models. arXiv:2403.19647.
        (SAE error terms in attribution.)
    Simonyan, K., Vedaldi, A. & Zisserman, A. (2013). Deep Inside
        Convolutional Networks: Visualising Image Classification Models and
        Saliency Maps. arXiv:1312.6034.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Sequence
)
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from einops import einsum
from torch import (
    Tensor,
    nn
)

from ..edits import (
    cross,
    run
)
from ..extract import (
    default_input_fn,
    layer_tokens
)
from ..layouts import keep_mask
from ..models import (
    frozen,
    layer_name
)
from .model import SAE


@dataclass
class Attribution:
    """
    Result of :func:`latent_attribution`.

    Attributes:
        acts: Latent activations, ``(batch, tokens, d_sae)``; zero at
            tokens that were not patched.
        attribution: ``acts * d target / d acts``, same shape: the
            first-order estimate of how much the target drops if the latent
            is zero-ablated at that position. TODO: also add median ablation
        error_attribution: Attribution of the SAE error term, summed over
            ``d_in``, shape ``(batch, tokens)``.
        keep: Boolean ``(batch, tokens)`` mask of the patched tokens.
        target: The target values, shape ``(batch,)``.
    """

    acts: Tensor
    attribution: Tensor
    error_attribution: Tensor
    keep: Tensor
    target: Tensor

    def top_latents(self, example: int = 0, n: int = 10) -> tuple[Tensor, Tensor]:
        """
        Latents with the largest total attribution for one example.

        Args:
            example: Index within the batch.
            n: Number of latents.

        Returns:
            ``(values, latent_ids)``, sorted by absolute value.
        """
        total = self.attribution[example].sum(0)
        idx = total.abs().topk(min(n, total.numel())).indices
        return total[idx], idx


def latent_fn(sae: SAE, latents: int | Sequence[int] | Tensor | None = None,
              pre: bool = False) -> Callable[[Tensor], Tensor]:
    """
    SAE latents as a function of a layer's tokens, to use wherever innards scores directions.

    E.g. ``interp.spatial_map(tokens, latent_fn(sae, 42), grid)``,
    ``interp.top_k(tokens, latent_fn(sae, 42))`` or
    ``interp.input_attribution(model, x, layer, latent_fn(sae, 42, pre=True))``.

    Args:
        sae: The SAE (put in eval mode).
        latents: One latent (the result has no latent axis), several, or
            ``None`` for all.
        pre: Encoder pre-activations instead of the (sparse) activations;
            informative even where a latent is not selected, e.g. for gradients.

    Returns:
        ``fn(tokens (..., d)) -> (...)`` for one latent, ``(..., k)`` otherwise.
    """
    sae.eval()
    idx = None if latents is None else torch.as_tensor(latents, device=sae.device)

    def fn(tokens: Tensor) -> Tensor:
        x = tokens.to(sae.w_enc.device, sae.w_enc.dtype)
        acts = sae.pre_activations(x) if pre else sae.encode(x)
        return acts if idx is None else acts[..., idx]

    return fn


@torch.no_grad()
def encode_inputs(
    sae: SAE,
    model: nn.Module,
    layer: str | nn.Module,
    batch: Any,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
) -> Tensor:
    """
    Latent activations for every token of a batch of model inputs.

    Args:
        sae: The SAE (put in eval mode).
        model: The model the SAE was trained on.
        layer: Module (or layer name) whose output the SAE reads.
        batch: Model inputs, already on the right device.
        input_fn: Maps the batch to model ``(args, kwargs)``.

    Returns:
        Activations ``(batch, tokens, d_sae)`` for all positions, including
        ones excluded during training.
    """
    model.eval()
    return latent_fn(sae)(layer_tokens(model, batch, layer, input_fn))


@torch.no_grad()
def _unit_atoms(sae: SAE) -> Tensor:
    return F.normalize(sae.decoder_weight().detach().float(), dim=-1)


@torch.no_grad()
def encoder_decoder_cosine(sae: SAE) -> Tensor:
    """
    Cosine similarity between each latent's encoder and decoder direction.

    Low values are expected for latents that must suppress interference from
    related latents (feature absorption; Chanin et al., 2024).

    Args:
        sae: The SAE.

    Returns:
        Tensor of shape ``(d_sae,)``.
    """
    enc = F.normalize(sae.w_enc.detach().float().T, dim=-1)
    return einsum(enc, _unit_atoms(sae), "s d, s d -> s").cpu()


def latent_attribution(
    sae: SAE,
    model: nn.Module,
    layer: str | nn.Module,
    batch: Any,
    target_fn: Callable[[Any, Any], Tensor],
    exclude: Sequence[int] = (),
    mask_fn: Callable[[Any], Tensor] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    target: str | nn.Module | None = None,
) -> Attribution:
    """
    Attributes a scalar model output to SAE latents at every position.

    The SAE is spliced in as ``x = decode(encode(x)) + error`` with the error
    term kept (Marks et al., 2024), so the forward pass is exactly the clean
    model. The attribution ``a * dT/da`` is the attribution-patching estimate
    (Nanda, 2023; Syed et al., 2023) of the change in the target ``T`` when
    latent ``a`` is zero-ablated.

    Args:
        sae: The SAE.
        model: The model the SAE was trained on.
        layer: Module (or layer name) whose output the SAE reads.
        batch: Model inputs, already on the right device.
        target_fn: ``fn(model_output, batch) -> (batch,)`` scalar per
            example, e.g. the logit of the predicted class, or of the correct
            next token at the last position.
        exclude: Positions left unpatched (e.g. CLS / BOS).
        mask_fn: Optional ``fn(batch) -> bool (batch, tokens)`` mask of
            tokens to patch, e.g. non-padding tokens.
        input_fn: Maps the batch to model ``(args, kwargs)``.
        target: For transcoders: the module whose output the
            transcoder replaces (``layer`` is its input). A skip
            connection's contribution is kept out of both the latent and the
            error attributions.

    Returns:
        An :class:`Attribution`.
    """
    sae.eval()
    model.eval()
    name = layer_name(model, layer)
    target = name if target is None else layer_name(model, target)
    mask = mask_fn(batch) if mask_fn is not None else None
    store: dict[str, Tensor] = {}

    def splice_fn(x: Tensor, y: Tensor) -> Tensor:
        x, y = x.detach().to(sae.w_enc.dtype), y.detach().to(sae.w_enc.dtype)
        with torch.no_grad():
            acts = sae.encode(x)
            skip = sae.skip(x)
        acts.requires_grad_(True)
        recon = sae.decode(acts)
        error = (y - recon.detach() - skip).detach().requires_grad_(True)
        store.update(acts=acts, error=error)
        return recon + skip + error

    edits = cross(name, target, splice_fn, exclude, mask)
    write = edits[target]

    def record_keep(tokens: Tensor) -> Tensor:
        store["keep"] = keep_mask(tokens, exclude, mask)
        return write(tokens)

    edits[target] = record_keep
    with frozen(model, sae):
        out, _ = run(model, batch, edits, input_fn=input_fn)
        target_value = target_fn(out, batch)
        target_value.sum().backward()
    acts, error, keep = store["acts"], store["error"], store["keep"]

    def scatter(values: Tensor) -> Tensor:
        full = values.new_zeros(*keep.shape, *values.shape[1:])
        full[keep] = values
        return full

    return Attribution(
        acts=scatter(acts.detach()),
        attribution=scatter((acts * acts.grad).detach()),
        error_attribution=scatter((error * error.grad).sum(-1).detach()),
        keep=keep,
        target=target_value.detach(),
    )
