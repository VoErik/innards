"""
Conversion to and from the SAELens on-disk format.

SAELens (github.com/decoderesearch/SAELens) stores an SAE as a directory with
``cfg.json`` and ``sae_weights.safetensors``. Exporting lets SAEs trained
here be used with its tooling, Neuronpedia and SAEBench; importing lets the
analysis code here run on published SAEs (Gemma Scope, Llama Scope, ...).
The SAELens package itself is not needed for either direction.

Architecture mapping (SAE / transcoder):

=====================  ==================================================
here                   SAELens
=====================  ==================================================
``topk``               ``topk`` / not supported for transcoders
``topk`` + abs_topk    ``abstopk`` / not supported
``batchtopk``          ``jumprelu`` with a constant threshold (SAELens'
                       own convention), ``jumprelu_transcoder``
``jumprelu``           ``jumprelu``, ``jumprelu_[skip_]transcoder``
``relu``               ``standard``, ``[skip_]transcoder``
=====================  ==================================================

Weights are exported with the input scale folded in, so the SAELens model
works on raw activations. Any decoder is exported as its (fused) dictionary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import (
    load_file,
    save_file
)

from .model import (
    SAE,
    SAEConfig
)

CFG_FILENAME = "cfg.json"
WEIGHTS_FILENAME = "sae_weights.safetensors"
# Marks exported configs as the SAELens >= 6 format so no legacy defaulting is applied on load.
FORMAT_VERSION = "6.0.0"


def _architecture(cfg: SAEConfig) -> str:
    act = cfg.activation
    if act in ("batchtopk", "jumprelu"):
        base = "jumprelu"
    elif act == "relu":
        base = "standard"
    elif act == "topk":
        base = "abstopk" if cfg.abs_topk else "topk"
    else:
        raise ValueError(f"activation {act!r} has no SAELens equivalent")
    if act == "batchtopk" and cfg.abs_topk:
        raise ValueError("BatchTopK with abs_topk has no SAELens equivalent")
    if not cfg.transcoder:
        return base
    if base not in ("jumprelu", "standard"):
        raise ValueError(f"SAELens has no {base} transcoder; use activation 'jumprelu', 'batchtopk' or 'relu'")
    prefix = "jumprelu_" if base == "jumprelu" else ""
    return f"{prefix}{'skip_' if cfg.skip_connection else ''}transcoder"


@torch.no_grad()
def save_saelens(sae: SAE, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    """
    Writes the SAE in the SAELens format.

    Args:
        sae: The SAE or transcoder (BatchTopK needs its inference threshold,
            i.e. some training steps).
        path: Output directory; created if needed.
        metadata: Entries of SAELens' ``metadata``, e.g. ``model_name``,
            ``hook_name``, ``context_size``, ``prepend_bos``.

    Returns:
        The output directory.

    Raises:
        ValueError: If the architecture has no SAELens equivalent, or a
            BatchTopK threshold was never estimated.
    """
    cfg = sae.cfg
    architecture = _architecture(cfg)
    sd = sae.folded_state_dict()
    weights = {
        "W_enc": sd["w_enc"],
        "b_enc": sd["b_enc"],
        "W_dec": sd["decoder.w_dec"],  # the (fused) dictionary, for every decoder type
        "b_dec": sd["b_dec"],
    }
    if cfg.activation == "jumprelu":
        weights["threshold"] = sd["activation.log_threshold"].exp()
    elif cfg.activation == "batchtopk":
        threshold = sd["activation.threshold"]
        if not bool(torch.isfinite(threshold)):
            raise ValueError("BatchTopK threshold not estimated yet; train (or run training-mode batches) first")
        weights["threshold"] = torch.full((cfg.d_sae,), threshold.clamp_min(0).item())
    if cfg.skip_connection:
        weights["W_skip"] = sd["w_skip"].T  # SAELens: (d_out, d_in)
    weights = {k: v.float().contiguous().cpu() for k, v in weights.items()}

    out = {
        "architecture": architecture,
        "d_in": cfg.d_in,
        "d_sae": cfg.d_sae,
        "dtype": "float32",
        "device": "cpu",
        "apply_b_dec_to_input": not cfg.transcoder,
        "normalize_activations": "none",
        "reshape_activations": "none",
        "metadata": {"sae_lens_version": FORMAT_VERSION, "sae_lens_training_version": FORMAT_VERSION,
                     **(metadata or {})},
    }
    if architecture in ("topk", "abstopk"):
        out.update(k=cfg.k, rescale_acts_by_decoder_norm=False)
    if cfg.transcoder:
        out["d_out"] = cfg.d_out
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(path / WEIGHTS_FILENAME))
    (path / CFG_FILENAME).write_text(json.dumps(out, indent=2))
    return path


@torch.no_grad()
def load_saelens(path: str | Path, device: str | torch.device = "cpu") -> tuple[SAE, dict[str, Any]]:
    """
    Loads an SAE or transcoder stored in the SAELens format.

    Supported architectures: ``standard`` (ReLU), ``topk``, ``abstopk``,
    ``jumprelu``, ``transcoder``, ``skip_transcoder``, ``jumprelu_transcoder``
    and ``jumprelu_skip_transcoder``; configs older than SAELens 6 that
    describe one of these (``activation_fn_str`` ``"relu"`` / ``"topk"``) are
    understood too.

    The loaded model computes the same function as SAELens' in eval mode:

    * ``apply_b_dec_to_input=False`` is absorbed into ``b_enc``.
    * ReLU and JumpReLU decoders are rescaled to unit-norm atoms (with the
      encoder, ``b_enc`` and thresholds rescaled to compensate), as SAELens'
      ``fold_W_dec_norm`` does. TopK decoders are kept as stored, because
      rescaling would change which latents are selected.
    * JumpReLU thresholds below 0 behave like 0, as SAELens applies a ReLU first.

    Args:
        path: Directory with ``cfg.json`` and ``sae_weights.safetensors``.
        device: Device to load onto.

    Returns:
        ``(sae, cfg)``: the SAE in eval mode, and the SAELens config
        (including its ``metadata``, e.g. ``hook_name``).

    Raises:
        ValueError: For unsupported architectures, input normalization or
            decoder-norm rescaling.
    """
    path = Path(path)
    raw = json.loads((path / CFG_FILENAME).read_text())
    weights = {k: v.float() for k, v in load_file(str(path / WEIGHTS_FILENAME)).items()}

    architecture = raw.get("architecture")
    if architecture is None:  # SAELens < 6
        act_fn = raw.get("activation_fn_str", raw.get("activation_fn", "relu"))
        kwargs = raw.get("activation_fn_kwargs") or {}
        architecture = "topk" if act_fn == "topk" else "standard"
        if act_fn not in ("relu", "topk"):
            raise ValueError(f"unsupported legacy activation_fn {act_fn!r}")
        raw.setdefault("k", kwargs.get("k"))
        raw.setdefault("apply_b_dec_to_input", True)
    if raw.get("normalize_activations", "none") != "none":
        raise ValueError(f"normalize_activations={raw['normalize_activations']!r} is not supported; "
                         "fold the normalization into the weights first")
    if raw.get("rescale_acts_by_decoder_norm"):
        raise ValueError("rescale_acts_by_decoder_norm=True is not supported")

    transcoder = "transcoder" in architecture
    skip = "skip" in architecture
    base = "jumprelu" if architecture.startswith("jumprelu") else architecture.split("_")[0]
    activation = {"standard": "relu", "transcoder": "relu", "skip": "relu", "topk": "topk", "abstopk": "topk",
                  "jumprelu": "jumprelu"}.get(base)
    if activation is None:
        raise ValueError(f"unsupported SAELens architecture {architecture!r}")

    w_enc, b_enc, w_dec, b_dec = weights["W_enc"], weights["b_enc"], weights["W_dec"], weights["b_dec"]
    if not transcoder and not raw.get("apply_b_dec_to_input", True):
        b_enc = b_enc + b_dec @ w_enc  # (x - b_dec) W + b_enc' == x W + b_enc
    threshold = weights.get("threshold")
    if activation in ("relu", "jumprelu"):
        norms = w_dec.norm(dim=-1).clamp_min(1e-8)
        w_dec, w_enc, b_enc = w_dec / norms[:, None], w_enc * norms, b_enc * norms
        if threshold is not None:
            threshold = threshold * norms

    cfg = SAEConfig(
        d_in=raw["d_in"],
        d_sae=raw["d_sae"],
        activation=activation,
        k=raw.get("k") if activation == "topk" else None,
        abs_topk=architecture == "abstopk",
        transcoder=transcoder,
        d_out=raw.get("d_out") if transcoder else None,
        skip_connection=skip,
    )
    sae = SAE(cfg)
    sd = sae.state_dict()
    sd.update({"w_enc": w_enc, "b_enc": b_enc, "decoder.w_dec": w_dec, "b_dec": b_dec})
    if activation == "jumprelu":
        sd["activation.log_threshold"] = threshold.clamp_min(0).log()
    if skip:
        sd["w_skip"] = weights["W_skip"].T
    sae.load_state_dict(sd)
    return sae.to(device).eval(), raw
