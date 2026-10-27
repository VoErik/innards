"""
Cross-layer transcoders (CLTs).

A CLT replaces all MLPs of a model at once (Ameisen et al., 2025; Lindsey et
al., 2024). Latents at layer ``l`` read that layer's MLP input ``x_l`` and
write to the MLP outputs of layer ``l`` *and every later layer*:

    a_l = sigma(W_enc[l]^T x_l + b_enc[l])
    y_hat_m = sum_{l <= m} a_l W_dec[l -> m] + b_dec[m]   (+ x_m W_skip[m])

so a feature computed once can explain the same output across several
layers, which keeps attribution graphs small. The activation is any
:mod:`innards.sae.activations` entry (Ameisen et al. use JumpReLU; TopK and BatchTopK
work as well). Each latent's decoder vectors are unit-norm jointly over all
layers it writes to. The loss is the mean normalised MSE over layers plus the
activation's sparsity penalty (summed over layers) and AuxK.

Training data are aligned MLP inputs and outputs of every layer, harvested
in one pass::

    store = ActivationStore(model, clt_hooks(mlp_inputs, mlp_outputs), loader)
    clt = CrossLayerTranscoder(n_layers=6, d_in=512, d_sae=4096, activation="topk", k=32)
    SAETrainer(clt, TrainerConfig(total_tokens=...)).train(store)

The model keeps its input and target scales as buffers (activations are in
these normalised units) rather than folding them into the weights, since a
latent writes to layers with different scales.

References:
    Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J. & Olah, C.
        (2024). Sparse Crosscoders for Cross-Layer Features and Model Diffing.
        Transformer Circuits Thread.
    Ameisen, E. et al. (2025). Circuit Tracing: Revealing Computational Graphs
        in Language Models. Transformer Circuits Thread.
"""

from __future__ import annotations

import json
import math
from collections.abc import (
    Callable,
    Iterator,
    Sequence
)
from contextlib import (
    ExitStack,
    contextmanager
)
from dataclasses import (
    asdict,
    dataclass,
    fields
)
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import (
    einsum,
    pack,
    rearrange,
    reduce,
    unpack
)
from safetensors import safe_open
from safetensors.torch import (
    load_file,
    save_file
)
from torch import (
    Tensor,
    nn
)

from .activations import (
    ACTIVATIONS,
    SparseCode,
    decode_code,
    rowwise_topk
)
from .data import (
    patch_kept_tokens,
    replace_activation
)
from .model import (
    SAEConfig,
    SAEOutput,
    calibration_scale,
    geometric_median,
    normalized_mse,
    register_model_type
)

FORMAT_NAME = "sae-clt"


def clt_hooks(inputs: Sequence[nn.Module], outputs: Sequence[nn.Module]) -> dict[str, nn.Module]:
    """
    Hook dict for a multi-hook :class:`~innards.sae.data.ActivationStore` feeding a CLT.

    Args:
        inputs: Module whose output is each layer's MLP input (e.g. the norm
            before the MLP), in layer order.
        outputs: Each layer's MLP, in layer order.

    Returns:
        ``{"in.0": ..., "out.0": ..., "in.1": ...}``.

    Raises:
        ValueError: If the lists differ in length.
    """
    if len(inputs) != len(outputs):
        raise ValueError("need one input and one output module per layer")
    hooks: dict[str, nn.Module] = {}
    for layer, (i, o) in enumerate(zip(inputs, outputs)):
        hooks[f"in.{layer}"] = i
        hooks[f"out.{layer}"] = o
    return hooks


@dataclass
class CLTConfig:
    """
    Configuration of a :class:`CrossLayerTranscoder`.

    Activation-related fields mean the same as in :class:`~innards.sae.model.SAEConfig`
    and apply to every layer.

    Attributes:
        n_layers: Number of layers (MLPs) replaced.
        d_in: Width of the MLP inputs.
        d_sae: Latents per layer.
        d_out: Width of the MLP outputs; ``None`` means ``d_in``.
        activation: Activation name (see :data:`innards.sae.activations.ACTIVATIONS`).
        k: Active latents per token and layer for TopK-style activations.
        abs_topk: Signed TopK selection.
        sparsity_coeff: Penalty weight for JumpReLU / ReLU (summed over layers).
        jumprelu_bandwidth: JumpReLU straight-through kernel width.
        jumprelu_init_threshold: Initial JumpReLU threshold.
        k_aux: Dead latents per layer used by AuxK; ``None`` means ``d_in // 2``.
        aux_coeff: AuxK weight; ``None`` uses the activation's default.
        dead_tokens_threshold: Tokens without firing after which a latent is dead.
        threshold_ema_decay: EMA decay of BatchTopK thresholds.
        skip_connection: Add a per-layer linear skip ``x_m W_skip[m]``.
    """

    n_layers: int
    d_in: int
    d_sae: int
    d_out: int | None = None
    activation: str = "jumprelu"
    k: int | None = None
    abs_topk: bool = False
    sparsity_coeff: float = 1e-3
    jumprelu_bandwidth: float = 0.05
    jumprelu_init_threshold: float = 0.1
    k_aux: int | None = None
    aux_coeff: float | None = None
    dead_tokens_threshold: int = 10_000_000
    threshold_ema_decay: float = 0.99
    skip_connection: bool = False

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")
        if self.d_out is None:
            self.d_out = self.d_in
        layer = self.layer_config()  # validates the activation settings
        self.k_aux, self.aux_coeff = layer.k_aux, layer.aux_coeff

    def layer_config(self) -> SAEConfig:
        """The per-layer settings as an :class:`~innards.sae.model.SAEConfig` (used to build activations)."""
        return SAEConfig(
            d_in=self.d_in, d_sae=self.d_sae, activation=self.activation, k=self.k, abs_topk=self.abs_topk,
            sparsity_coeff=self.sparsity_coeff, jumprelu_bandwidth=self.jumprelu_bandwidth,
            jumprelu_init_threshold=self.jumprelu_init_threshold, k_aux=self.k_aux, aux_coeff=self.aux_coeff,
            dead_tokens_threshold=self.dead_tokens_threshold, threshold_ema_decay=self.threshold_ema_decay,
            transcoder=True, d_out=self.d_out, skip_connection=self.skip_connection,
        )

    def to_json(self) -> str:
        """Serializes the config to JSON."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "CLTConfig":
        """Builds a config from :meth:`to_json` output, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in json.loads(s).items() if k in known})


@register_model_type
class CrossLayerTranscoder(nn.Module):
    """
    Cross-layer transcoder over ``n_layers`` MLPs.

    Inputs are stacked per layer: ``x`` of shape ``(..., n_layers, d_in)`` (MLP
    inputs) and ``target`` of shape ``(..., n_layers, d_out)`` (MLP outputs).
    Implements the trainer protocol of :class:`~innards.sae.model.SAE`, so it trains
    with :class:`~innards.sae.trainer.SAETrainer` on batches from :func:`clt_hooks`.

    Args:
        cfg: Configuration; if ``None``, ``kwargs`` build a :class:`CLTConfig`.
        **kwargs: Fields of :class:`CLTConfig`.

    Attributes:
        cfg: The configuration.
        k_current: The k used by TopK-style activations (k annealing).
        sparsity_scale: Multiplier of the sparsity penalty, for warmup.
    """

    def __init__(self, cfg: CLTConfig | None = None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else CLTConfig(**kwargs)
        n, d_in, d_sae, d_out = self.cfg.n_layers, self.cfg.d_in, self.cfg.d_sae, self.cfg.d_out
        layer_cfg = self.cfg.layer_config()
        self.k_current = self.cfg.k
        self.sparsity_scale = 1.0

        self.w_enc = nn.Parameter(torch.empty(n, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(n, d_sae))
        # w_dec[l]: (d_sae, n_layers - l, d_out), from layer l to layers l..n-1
        self.w_dec = nn.ParameterList([nn.Parameter(torch.empty(d_sae, n - layer, d_out)) for layer in range(n)])
        self.b_dec = nn.Parameter(torch.zeros(n, d_out))
        if self.cfg.skip_connection:
            self.w_skip = nn.Parameter(torch.zeros(n, d_in, d_out))
        self.activations = nn.ModuleList([ACTIVATIONS[self.cfg.activation](layer_cfg) for _ in range(n)])

        self.register_buffer("input_scale", torch.ones(n))
        self.register_buffer("target_scale", torch.ones(n))
        self.register_buffer("tokens_since_active", torch.zeros(n, d_sae, dtype=torch.long))
        self.reset_parameters()

    # ---- parameters ----

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Random unit-norm decoders and encoder directions, zero biases and skip."""
        n, d_in, d_sae = self.cfg.n_layers, self.cfg.d_in, self.cfg.d_sae
        for w in self.w_dec:
            nn.init.normal_(w)
        self._normalize_decoders()
        self.w_enc.copy_(rearrange(F.normalize(torch.randn(n, d_sae, d_in), dim=-1), "n s d -> n d s"))
        self.b_enc.zero_()
        self.b_dec.zero_()
        if self.cfg.skip_connection:
            self.w_skip.zero_()

    @torch.no_grad()
    def _normalize_decoders(self) -> None:
        for w in self.w_dec:
            w.div_(w.flatten(1).norm(dim=-1).clamp_min(1e-8)[:, None, None])

    def decoder_weight(self, layer: int) -> Tensor:
        """Decoder of ``layer``'s latents, flattened to ``(d_sae, (n_layers - layer) * d_out)``."""
        return self.w_dec[layer].flatten(1)

    @torch.no_grad()
    def pre_optimizer_step(self) -> None:
        """Removes each decoder gradient's component parallel to the (joint) decoder vector."""
        for w in self.w_dec:
            if w.grad is not None:
                proj = (w.grad * w).sum((1, 2))
                w.grad.sub_(proj[:, None, None] * w)

    def post_optimizer_step(self) -> None:
        """Projects decoder vectors back to unit norm."""
        self._normalize_decoders()

    # ---- forward ----

    @property
    def device(self) -> torch.device:
        return self.w_enc.device

    @property
    def dead_mask(self) -> Tensor:
        """Boolean ``(n_layers, d_sae)`` mask of dead latents."""
        return self.tokens_since_active >= self.cfg.dead_tokens_threshold

    def _stack(self, t: Tensor, name: str) -> tuple[Tensor, list]:
        flat, ps = pack([t], f"* n {name}")
        if flat.shape[1] != self.cfg.n_layers:
            raise ValueError(f"expected {self.cfg.n_layers} layers in dim -2, got {flat.shape[1]}")
        return flat, ps

    def _pre(self, x_s: Tensor) -> Tensor:
        return einsum(x_s, self.w_enc, "b n d, n d s -> b n s") + self.b_enc

    def _skip(self, x_s: Tensor) -> Tensor | float:
        if not self.cfg.skip_connection:
            return 0.0
        return einsum(x_s.to(self.w_skip.dtype), self.w_skip, "b n d, n d o -> b n o")

    def _codes(self, pre: Tensor) -> list[SparseCode]:
        return [act(pre[:, layer], self.k_current) for layer, act in enumerate(self.activations)]

    def _predict(self, codes: Sequence[SparseCode], x_s: Tensor) -> Tensor:
        """Prediction in scaled units, ``(b, n_layers, d_out)``."""
        b = x_s.shape[0]
        n = self.cfg.n_layers
        parts = [self.b_dec[m].expand(b, -1) for m in range(n)]
        for layer, code in enumerate(codes):
            contrib = decode_code(code, self.decoder_weight(layer)).view(b, n - layer, self.cfg.d_out)
            for j in range(n - layer):
                parts[layer + j] = parts[layer + j] + contrib[:, j]
        return torch.stack(parts, 1) + self._skip(x_s)

    @staticmethod
    def _dense(code: SparseCode, d_sae: int) -> Tensor:
        if code.dense is not None:
            return code.dense
        dense = torch.zeros(code.n, d_sae, device=code.vals.device, dtype=code.vals.dtype)
        dense[code.rows, code.cols] = code.vals
        return dense

    def _counts(self, code: SparseCode) -> Tensor:
        if code.dense is not None:
            return (code.dense != 0).sum(0)
        return torch.bincount(code.cols[code.vals != 0], minlength=self.cfg.d_sae)

    def encode(self, x: Tensor) -> Tensor:
        """
        Latent activations, ``(..., n_layers, d_sae)``, in the CLT's normalised units.

        Args:
            x: MLP inputs, ``(..., n_layers, d_in)``.
        """
        flat, ps = self._stack(x, "d")
        x_s = flat.to(self.w_enc.dtype) * self.input_scale[:, None]
        acts = torch.stack([self._dense(c, self.cfg.d_sae) for c in self._codes(self._pre(x_s))], 1)
        [acts] = unpack(acts, ps, "* n s")
        return acts

    def reconstruct(self, x: Tensor) -> Tensor:
        """
        Predicted MLP outputs, ``(..., n_layers, d_out)``, in input units.

        Args:
            x: MLP inputs, ``(..., n_layers, d_in)``.
        """
        flat, ps = self._stack(x, "d")
        x_s = flat.to(self.w_enc.dtype) * self.input_scale[:, None]
        recon = self._predict(self._codes(self._pre(x_s)), x_s) / self.target_scale[:, None]
        [recon] = unpack(recon.to(x.dtype), ps, "* n o")
        return recon

    def forward(self, x: Tensor, target: Tensor | None = None, return_acts: bool = False) -> SAEOutput:
        """
        Predicts every layer's MLP output and computes the training objective.

        Args:
            x: MLP inputs, ``(..., n_layers, d_in)``.
            target: MLP outputs, ``(..., n_layers, d_out)``.
            return_acts: Return the ``(..., n_layers, d_sae)`` activations.

        Returns:
            An :class:`~innards.sae.model.SAEOutput`; metrics include the mean and
            per-layer (``layer<i>/nmse``) normalised MSE, and ``latent_counts``
            has shape ``(n_layers, d_sae)``.

        Raises:
            ValueError: If ``target`` is missing.
        """
        if target is None:
            raise ValueError("a cross-layer transcoder needs targets (the MLP outputs)")
        cfg = self.cfg
        xf, ps = self._stack(x, "d")
        yf, _ = self._stack(target, "o")
        x_s = xf.to(self.w_enc.dtype) * self.input_scale[:, None]
        y_s = yf.to(self.w_enc.dtype) * self.target_scale[:, None]
        pre = self._pre(x_s)
        codes = self._codes(pre)
        recon_s = self._predict(codes, x_s)
        counts = torch.stack([self._counts(c) for c in codes])
        b = xf.shape[0]

        if self.training:
            with torch.no_grad():
                self.tokens_since_active += b
                self.tokens_since_active[counts > 0] = 0
                for act, code in zip(self.activations, codes):
                    act.update_stats(code)

        y_f, recon_f = y_s.float(), recon_s.float()
        layer_nmse = torch.stack([normalized_mse(recon_f[:, m], y_f[:, m]) for m in range(cfg.n_layers)])
        loss = layer_nmse.mean()
        metrics: dict[str, Tensor] = {"nmse": loss.detach(), "l0": (counts.sum() / (b * cfg.n_layers)).float()}
        for m in range(cfg.n_layers):
            metrics[f"layer{m}/nmse"] = layer_nmse[m].detach()
            metrics[f"layer{m}/l0"] = (counts[m].sum() / b).float()

        if self.training:
            penalties = [act.penalty(pre[:, layer], code, self.decoder_weight(layer))
                         for layer, (act, code) in enumerate(zip(self.activations, codes))]
            penalties = [p for p in penalties if p is not None]
            if penalties:
                penalty = torch.stack(penalties).sum()
                loss = loss + self.sparsity_scale * penalty
                metrics["sparsity_loss"] = penalty.detach()
            if cfg.aux_coeff > 0:
                aux = self._aux_loss(pre, (y_f - recon_f).detach())
                loss = loss + cfg.aux_coeff * aux
                metrics["aux_loss"] = aux.detach()
        metrics["loss"] = loss.detach()

        [recon] = unpack((recon_s / self.target_scale[:, None]).to(x.dtype), ps, "* n o")
        acts = None
        if return_acts:
            [acts] = unpack(torch.stack([self._dense(c, cfg.d_sae) for c in codes], 1), ps, "* n s")
        return SAEOutput(acts=acts, recon=recon, loss=loss, metrics=metrics, latent_counts=counts)

    def _aux_loss(self, pre: Tensor, residual: Tensor) -> Tensor:
        """AuxK per layer: dead latents of layer ``l`` reconstruct the residual of layers ``l..``."""
        dead = self.dead_mask
        losses = []
        for layer, act in enumerate(self.activations):
            num_dead = int(dead[layer].sum())
            if num_dead == 0:
                continue
            p = pre[:, layer]
            code = rowwise_topk(p, act.score(p), min(self.cfg.k_aux, num_dead), signed=self.cfg.abs_topk,
                                mask=dead[layer])
            target = residual[:, layer:].reshape(residual.shape[0], -1)
            recon = decode_code(code, self.decoder_weight(layer)).float()
            losses.append(torch.nan_to_num(normalized_mse(recon, target), nan=0.0))
        if not losses:
            return pre.new_zeros((), dtype=torch.float32)
        return torch.stack(losses).mean()

    # ---- initialization ----

    @torch.no_grad()
    def initialize_from_data(
        self,
        x: Tensor,
        target: Tensor,
        normalize_input: bool = True,
        b_dec_init: str = "geometric_median",
        calibrate_encoder: bool = True,
    ) -> None:
        """
        Sets per-layer input / target scales (``E||.|| = sqrt(d)``), the output
        biases, and a global encoder scale chosen as for transcoders (see
        :meth:`innards.sae.model.SAE.calibrate_encoder_scale_`).

        Args:
            x: MLP inputs, ``(..., n_layers, d_in)``.
            target: MLP outputs, ``(..., n_layers, d_out)``.
            normalize_input: Estimate the scales.
            b_dec_init: ``"geometric_median"``, ``"mean"`` or ``"zeros"``.
            calibrate_encoder: Rescale the encoder as for transcoders (see
                :meth:`innards.sae.model.SAE.calibrate_encoder_scale_`).

        Raises:
            ValueError: If ``b_dec_init`` is unknown.
        """
        xf = self._stack(x.to(self.device, torch.float32), "d")[0]
        yf = self._stack(target.to(self.device, torch.float32), "o")[0]
        if normalize_input:
            self.input_scale.copy_(math.sqrt(self.cfg.d_in) / xf.norm(dim=-1).mean(0).clamp_min(1e-8))
            self.target_scale.copy_(math.sqrt(self.cfg.d_out) / yf.norm(dim=-1).mean(0).clamp_min(1e-8))
        x_s, y_s = xf * self.input_scale[:, None], yf * self.target_scale[:, None]
        for m in range(self.cfg.n_layers):
            if b_dec_init == "geometric_median":
                self.b_dec[m].copy_(geometric_median(y_s[:, m]))
            elif b_dec_init == "mean":
                self.b_dec[m].copy_(y_s[:, m].mean(0))
            elif b_dec_init == "zeros":
                self.b_dec[m].zero_()
            else:
                raise ValueError(f"unknown b_dec_init {b_dec_init!r}")
        if calibrate_encoder and self.b_enc.abs().max() == 0:
            was_training = self.training
            self.train()
            base = self.b_dec + self._skip(x_s)
            recon = self._predict(self._codes(self._pre(x_s)), x_s) - base
            target_c = y_s - base
            alpha = calibration_scale(recon, target_c, match_magnitude=not self.uses_k)
            if alpha > 0:
                self.w_enc.mul_(alpha)
            self.train(was_training)

    # ---- trainer protocol ----

    def batch_tensors(self, batch: Any) -> tuple[Tensor, Tensor]:
        """
        Stacks a :func:`clt_hooks` dict batch into ``(x, y)`` of shape ``(n, n_layers, d)``.

        Args:
            batch: Dict with ``in.<l>`` / ``out.<l>`` tensors, or an already
                stacked ``(x, y)`` pair.
        """
        if isinstance(batch, dict):
            n = self.cfg.n_layers
            x = torch.stack([pack([batch[f"in.{m}"]], "* d")[0] for m in range(n)], 1)
            y = torch.stack([pack([batch[f"out.{m}"]], "* d")[0] for m in range(n)], 1)
            return x, y
        x, y = batch
        return self._stack(x, "d")[0], self._stack(y, "o")[0]

    @property
    def min_init_tokens(self) -> int:
        return 0

    @property
    def uses_k(self) -> bool:
        return self.activations[0].uses_k

    @property
    def lr_d_sae(self) -> int:
        return self.cfg.d_sae

    def set_k(self, k: int) -> None:
        self.k_current = int(max(1, min(k, self.cfg.d_sae)))

    def anneal_k(self, progress: float, k_initial: int | None = None) -> None:
        if self.uses_k:
            k0 = k_initial or min(self.cfg.d_sae, 10 * self.cfg.d_in)
            self.set_k(round(k0 * (1 - progress) + self.cfg.k * progress))

    @torch.no_grad()
    def eval_sums(self, out: SAEOutput, x: Tensor, y: Tensor) -> dict[str, tuple[float, float, float]]:
        """Evaluation sums over all layers (``""``) and per layer (``layer<i>``); L0 is per layer."""
        y = y.float()
        err = ((out.recon.float() - y) ** 2).sum((0, 2))
        tot = ((y - reduce(y, "b n d -> 1 n d", "mean")) ** 2).sum((0, 2))
        counts = out.latent_counts.sum(-1).float()
        sums = {f"layer{m}": (err[m].item(), tot[m].item(), counts[m].item()) for m in range(self.cfg.n_layers)}
        sums[""] = (err.sum().item(), tot.sum().item(), counts.mean().item())
        return sums

    def config_json(self) -> str:
        return self.cfg.to_json()

    @classmethod
    def from_config_json(cls, s: str) -> "CrossLayerTranscoder":
        return cls(CLTConfig.from_json(s))

    def save(self, path: str | Path, extra_metadata: dict[str, str] | None = None) -> Path:
        """
        Saves weights, scales and thresholds to a safetensors file, config in its metadata.

        Args:
            path: Destination file.
            extra_metadata: Additional string metadata.

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sd = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        metadata = {"format": FORMAT_NAME, "config": self.cfg.to_json(),
                    **{str(k): str(v) for k, v in (extra_metadata or {}).items()}}
        save_file(sd, str(path), metadata=metadata)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "CrossLayerTranscoder":
        """
        Loads a CLT saved with :meth:`save`.

        Raises:
            ValueError: If the file is not a CLT file.
        """
        with safe_open(str(path), framework="pt") as f:
            meta = dict(f.metadata() or {})
        if meta.get("format") != FORMAT_NAME:
            raise ValueError(f"{path} is not a cross-layer transcoder file")
        model = cls(CLTConfig.from_json(meta["config"]))
        model.load_state_dict(load_file(str(path)))
        return model.to(device).eval()


@contextmanager
def splice_clt(
    clt: CrossLayerTranscoder,
    inputs: Sequence[nn.Module],
    outputs: Sequence[nn.Module],
    keep_error: bool = False,
    edit: Callable[[int, Tensor], Tensor] | None = None,
    layout: str = "btd",
    exclude_positions: Sequence[int] = (),
    token_mask_fn: Callable[[Any], Tensor] | None = None,
    batch: Any = None,
) -> Iterator[None]:
    """
    Replaces every MLP output by the CLT's prediction during forward passes.

    Latents of layer ``l`` are computed from the (possibly already patched)
    MLP input of layer ``l``; MLP ``m``'s output is replaced at kept tokens by
    the sum of the contributions of layers ``<= m``. With ``keep_error`` the
    CLT error of each layer is added back, so only ``edit`` changes the model.

    Args:
        clt: The CLT (should be in eval mode).
        inputs: Modules whose outputs are the MLP inputs, in layer order.
        outputs: The MLPs, in layer order.
        keep_error: Add each layer's error ``y_m - y_hat_m`` back.
        edit: Optional ``fn(layer, acts) -> acts`` on a layer's
            ``(n, d_sae)`` activations (normalised units).
        layout: Activation layout at the hooks.
        exclude_positions: Positions left unchanged.
        token_mask_fn: Optional ``fn(batch) -> bool (batch, tokens)`` mask.
        batch: The model input batch passed to ``token_mask_fn``.

    Yields:
        None.
    """
    n, d_out = clt.cfg.n_layers, clt.cfg.d_out
    state: dict[int, Any] = {}

    def read(layer: int):
        def fn(x: Tensor, y: Tensor) -> Tensor:
            x_s = x.to(clt.w_enc.dtype) * clt.input_scale[layer]
            pre = einsum(x_s, clt.w_enc[layer], "b d, d s -> b s") + clt.b_enc[layer]
            code = clt.activations[layer](pre, clt.k_current)
            acts = CrossLayerTranscoder._dense(code, clt.cfg.d_sae)
            clean = (acts @ clt.decoder_weight(layer)).view(-1, n - layer, d_out)
            edited = clean if edit is None else (edit(layer, acts) @ clt.decoder_weight(layer)).view(-1, n - layer, d_out)
            skip = x_s @ clt.w_skip[layer] if clt.cfg.skip_connection else 0.0
            state[layer] = (clean, edited, skip)
            return y
        return fn

    def write(m: int):
        def fn(x: Tensor, y: Tensor) -> Tensor:
            clean = clt.b_dec[m] + sum(state[layer][0][:, m - layer] for layer in range(m + 1)) + state[m][2]
            edited = clt.b_dec[m] + sum(state[layer][1][:, m - layer] for layer in range(m + 1)) + state[m][2]
            scale = clt.target_scale[m]
            out = edited / scale
            return out + (y.to(out.dtype) - clean / scale) if keep_error else out
        return fn

    hooks = []
    for layer in range(n):
        hooks.append(replace_activation(inputs[layer], lambda a, f=read(layer): patch_kept_tokens(
            a, f, layout, exclude_positions, token_mask_fn, batch)))
        hooks.append(replace_activation(outputs[layer], lambda a, f=write(layer): patch_kept_tokens(
            a, f, layout, exclude_positions, token_mask_fn, batch)))
    with ExitStack() as stack:
        for h in hooks:
            stack.enter_context(h)
        yield
