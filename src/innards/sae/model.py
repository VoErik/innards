"""
Sparse autoencoders.

:class:`SAE` is one model class assembled from three independent parts:

* an **activation** (:mod:`innards.sae.activations`): TopK, BatchTopK, JumpReLU or
  ReLU + L1, i.e. what usually distinguishes SAE "architectures";
* a **decoder** (:mod:`innards.sae.decoders`): unit-norm linear or Archetypal;
* **loss terms** shared by all of them: normalised MSE, AuxK, Matryoshka and
  Multi-TopK.

With ``transcoder=True`` the same model reconstructs a *different*
activation than it reads, e.g. an MLP's output from its input (Dunefsky,
Chlenski & Nanda, 2024), optionally with a linear skip connection (Paulo,
Shabalin & Belrose, 2025).

Optimization-side tricks (learning-rate schedule, optimizers, k annealing,
sparsity warmup, KL fine-tuning, logging) live in :mod:`innards.sae.trainer`.

Typical usage::

    sae = SAE(d_in=768, d_sae=32768, activation="batchtopk", k=64)
    ...  # train with innards.sae.trainer.SAETrainer
    sae.save("sae.safetensors")
    sae = SAE.load("sae.safetensors", device="cuda")
    acts = sae.encode(x)  # (..., d_sae), x in raw activation units

References:
    Bricken, T. et al. (2023). Towards Monosemanticity: Decomposing Language
        Models With Dictionary Learning. Transformer Circuits Thread.
    Conerly, T., Templeton, A., Bricken, T., Marcus, J. & Henighan, T. (2024).
        Update on how we train SAEs. Circuits Updates, April 2024.
    Gao, L. et al. (2024). Scaling and evaluating sparse autoencoders.
        arXiv:2406.04093.
    Lieberum, T. et al. (2024). Gemma Scope: Open Sparse Autoencoders
        Everywhere All At Once on Gemma 2. arXiv:2408.05147.
    Chanin, D. et al. (2024). A is for Absorption: Studying Feature Splitting
        and Absorption in Sparse Autoencoders. arXiv:2409.14507.
    Bussmann, B., Nabeshima, N., Karvonen, A. & Nanda, N. (2025). Learning
        Multi-Level Features with Matryoshka Sparse Autoencoders.
        arXiv:2503.17547.
    Dunefsky, J., Chlenski, P. & Nanda, N. (2024). Transcoders Find
        Interpretable LLM Feature Circuits. arXiv:2406.11944.
    Paulo, G., Shabalin, S. & Belrose, N. (2025). Transcoders Beat Sparse
        Autoencoders for Interpretability. arXiv:2501.18823.
"""

from __future__ import annotations

import json
import math
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
from .decoders import DECODERS

FORMAT_NAME = "sae"
FORMAT_VERSION = "2"

# Model classes the trainer can rebuild from a checkpoint, by class name.
MODEL_TYPES: dict[str, type[nn.Module]] = {}


def register_model_type(cls: type[nn.Module]) -> type[nn.Module]:
    """Class decorator registering a trainable model for :meth:`innards.sae.trainer.SAETrainer.from_checkpoint`."""
    MODEL_TYPES[cls.__name__] = cls
    return cls


@dataclass
class SAEConfig:
    """
    Architecture and loss configuration of an :class:`SAE`.

    Attributes:
        d_in: Width of the input activations.
        d_sae: Number of latents (dictionary size).
        activation: Name in :data:`innards.sae.activations.ACTIVATIONS`: ``"topk"``,
            ``"batchtopk"``, ``"jumprelu"`` or ``"relu"``.
        decoder: Name in :data:`innards.sae.decoders.DECODERS`: ``"linear"`` or
            ``"archetypal"``.
        k: Active latents per token (``topk``; batch average for
            ``batchtopk``). Unused by penalty-based activations.
        abs_topk: For ``topk`` / ``batchtopk``: select latents by absolute
            pre-activation and keep their sign, without a ReLU, so one latent
            can encode both directions of a feature (Zhu, Khalili & Zhu, 2025).
        sparsity_coeff: Weight of the sparsity penalty of ``jumprelu`` (L0)
            and ``relu`` (L1), relative to the normalised MSE. It sets the
            L0 / reconstruction trade-off, so tune it for the L0 you want;
            the L1 penalty typically needs a larger value than the L0 one.
        jumprelu_bandwidth: Width of the straight-through kernel of JumpReLU
            (Rajamanoharan et al., 2024), in scaled units.
        jumprelu_init_threshold: Initial JumpReLU threshold, in scaled units.
            Rajamanoharan et al. (2024) use 0.001 for both this and the
            bandwidth; with pre-activations of order 1 in our scaled units,
            thresholds then only leave the dense regime after a long run, so
            the defaults here are larger.
        k_aux: Number of dead latents used by the AuxK loss. ``None`` resolves
            to ``d_in // 2`` (Gao et al., 2024, App. A.2), capped at ``d_sae``.
        aux_coeff: Weight of the AuxK loss. ``None`` uses the activation's
            default: 1/32 for TopK variants (Gao et al., 2024), 0 otherwise.
        dead_tokens_threshold: A latent is dead after this many tokens without
            firing; 10M as in Gao et al. (2024).
        threshold_ema_decay: EMA decay for the BatchTopK inference threshold.
        multi_topk: Add the Multi-TopK loss L(k) + L(4k)/8 (Gao et al., 2024,
            Sec. 5.3.1); TopK variants only.
        multi_topk_factor: Multiplier of ``k`` for the Multi-TopK term.
        multi_topk_coeff: Weight of the Multi-TopK term.
        matryoshka: Train nested dictionary prefixes to reconstruct the input on
            their own (Bussmann, Nabeshima, Karvonen & Nanda, 2025), which
            reduces feature absorption (Chanin et al., 2024).
        matryoshka_prefixes: Prefix sizes. ``None`` with ``matryoshka=True``
            uses ``[d_sae/64, d_sae/16, d_sae/4, d_sae]``. ``d_sae`` is always appended.
        archetypal_n_candidates: Number of candidate points of the Archetypal
            decoder. ``None`` resolves to ``min(d_sae, 8192)``; its weight has
            ``d_sae * n_candidates`` entries, so memory grows quickly.
        archetypal_delta: Radius of the Archetypal relaxation ball; 1.0 is the
            default of the reference implementation.
        archetypal_learn_scale: Learn the Archetypal global multiplier ``exp(s)``.
        loss_normalization: ``"batch"`` divides the MSE by the variance of each
            batch; ``"fixed"`` divides by a constant computed once from the
            initialization sample, as in Gao et al. (2024, App. A.6).
        transcoder: Reconstruct a separate target activation (``forward(x,
            target)``) instead of the input; the encoder then has no
            pre-encoder bias and input and target get separate scales.
        d_out: Width of the target; ``None`` means ``d_in``. Only for
            transcoders.
        skip_connection: Add a linear skip term ``x W_skip`` to a transcoder's
            output (Paulo et al., 2025); ``W_skip`` starts at zero.
    """

    d_in: int
    d_sae: int
    activation: str = "topk"
    decoder: str = "linear"

    k: int | None = None
    abs_topk: bool = False

    sparsity_coeff: float = 1e-3
    jumprelu_bandwidth: float = 0.05
    jumprelu_init_threshold: float = 0.1

    k_aux: int | None = None
    aux_coeff: float | None = None
    dead_tokens_threshold: int = 10_000_000

    threshold_ema_decay: float = 0.99

    multi_topk: bool = False
    multi_topk_factor: int = 4
    multi_topk_coeff: float = 1.0 / 8.0

    matryoshka: bool = False
    matryoshka_prefixes: tuple[int, ...] | None = None

    archetypal_n_candidates: int | None = None
    archetypal_delta: float = 1.0
    archetypal_learn_scale: bool = True

    loss_normalization: str = "batch"

    transcoder: bool = False
    d_out: int | None = None
    skip_connection: bool = False

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}, expected one of {sorted(ACTIVATIONS)}")
        if self.decoder not in DECODERS:
            raise ValueError(f"unknown decoder {self.decoder!r}, expected one of {sorted(DECODERS)}")
        act_cls = ACTIVATIONS[self.activation]
        act_cls.validate(self)
        if self.multi_topk and not act_cls.uses_k:
            raise ValueError(f"multi_topk needs a TopK-style activation, got {self.activation!r}")
        if self.loss_normalization not in ("batch", "fixed"):
            raise ValueError(f"loss_normalization must be 'batch' or 'fixed', got {self.loss_normalization!r}")
        if self.d_out is None:
            self.d_out = self.d_in
        if not self.transcoder and (self.d_out != self.d_in or self.skip_connection):
            raise ValueError("d_out != d_in and skip_connection need transcoder=True")

        if self.aux_coeff is None:
            self.aux_coeff = act_cls.default_aux_coeff
        if self.k_aux is None:
            self.k_aux = self.d_in // 2
        self.k_aux = max(1, min(self.k_aux, self.d_sae))

        if self.matryoshka_prefixes is not None:
            self.matryoshka = True
        if self.matryoshka:
            prefixes = self.matryoshka_prefixes
            if prefixes is None:
                prefixes = [self.d_sae // f for f in (64, 16, 4)]
            prefixes = sorted({int(p) for p in prefixes if 0 < int(p) < self.d_sae})
            self.matryoshka_prefixes = (*prefixes, self.d_sae)
        else:
            self.matryoshka_prefixes = None

        if self.decoder == "archetypal" and self.archetypal_n_candidates is None:
            self.archetypal_n_candidates = min(self.d_sae, 8192)

    def to_json(self) -> str:
        """Serializes the config to a JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "SAEConfig":
        """
        Builds a config from :meth:`to_json` output, ignoring unknown keys.

        Args:
            s: JSON string.

        Returns:
            The parsed config.
        """
        raw = json.loads(s)
        known = {f.name for f in fields(cls)}
        raw = {k: v for k, v in raw.items() if k in known}
        if raw.get("matryoshka_prefixes") is not None:
            raw["matryoshka_prefixes"] = tuple(raw["matryoshka_prefixes"])
        return cls(**raw)


@dataclass
class SAEOutput:
    """
    Result of :meth:`SAE.forward`.

    Attributes:
        acts: Sparse activations ``(..., d_sae)`` in input units, or ``None``
            if ``return_acts=False``.
        recon: Reconstruction ``(..., d_in)`` in input units.
        loss: Scalar training objective.
        metrics: Detached scalar metrics (``nmse``, ``l0``, ``loss`` and,
            when active, ``sparsity_loss``, ``aux_loss``, ``matryoshka_loss``,
            ``multi_topk_nmse``).
        latent_counts: Number of tokens each latent fired on, shape ``(d_sae,)``
            (for models with several dictionaries, the counts of all of them).
        parts: Per-member outputs of models made of several dictionaries.
    """

    acts: Tensor | None
    recon: Tensor | None
    loss: Tensor
    metrics: dict[str, Tensor]
    latent_counts: Tensor
    parts: dict[str, "SAEOutput"] | None = None


def normalized_mse(recon: Tensor, target: Tensor, denom: Tensor | None = None) -> Tensor:
    """
    Normalised MSE, i.e. 1 - fraction of variance explained.

    Gao et al. (2024) define it as the MSE divided by the MSE of always
    predicting the mean activation.

    Args:
        recon: Reconstruction, shape ``(n, d)``.
        target: Target, shape ``(n, d)``.
        denom: Optional fixed denominator (mean per-dimension variance). If
            ``None``, the variance of ``target`` in this batch is used.

    Returns:
        Scalar tensor.
    """
    mse = reduce((recon - target) ** 2, "n d -> ", "mean")
    if denom is None:
        centred = target - reduce(target, "n d -> 1 d", "mean")
        denom = reduce(centred**2, "n d -> ", "mean")
    return mse / denom.clamp_min(1e-8)


def calibration_scale(recon: Tensor, target: Tensor, match_magnitude: bool) -> Tensor:
    """
    Encoder scale for :meth:`SAE.calibrate_encoder_scale_`.

    Args:
        recon: Initial prediction (without biases), ``(n, d)``.
        target: What it should predict, ``(n, d)``.
        match_magnitude: Match norms instead of the least-squares fit.

    Returns:
        The scalar (callers ignore non-positive values).
    """
    if match_magnitude:
        return (target.pow(2).sum() / recon.pow(2).sum().clamp_min(1e-8)).sqrt()
    return (recon * target).sum() / (recon * recon).sum().clamp_min(1e-8)


@torch.no_grad()
def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-6) -> Tensor:
    """
    Geometric median via Weiszfeld's algorithm.

    Used to initialize the decoder bias (Gao et al., 2024, App. A.1).

    Args:
        points: Data, shape ``(n, d)``.
        max_iter: Maximum number of iterations.
        tol: Relative tolerance on the update size.

    Returns:
        The geometric median, shape ``(d,)``.
    """
    points = points.float()
    median = reduce(points, "n d -> d", "mean")
    for _ in range(max_iter):
        dist = (points - median).norm(dim=-1).clamp_min(1e-8)
        w = 1.0 / dist
        new = einsum(w, points, "n, n d -> d") / w.sum()
        shift = (new - median).norm()
        median = new
        if shift <= tol * median.norm().clamp_min(1.0):
            break
    return median


@register_model_type
class SAE(nn.Module):
    """
    Sparse autoencoder with a pluggable activation and decoder.

    The encoder is ``f(x) = sigma(W_enc^T (x - b_dec) + b_enc)`` with the
    pre-encoder bias of Bricken et al. (2023) and ``sigma`` the configured
    :class:`~innards.sae.activations.Activation`. The decoder is
    ``x_hat = f(x) D + b_dec`` with ``D`` from the configured
    :class:`~innards.sae.decoders.Decoder`.

    Internally the model works on ``x * input_scale``, a single scalar chosen
    so that ``E||x|| = sqrt(d_in)`` (Conerly et al., 2024). Everything the
    public API returns is in input units, and :meth:`save` folds the scale into
    the weights (as done for Gemma Scope; Lieberum et al., 2024).

    A transcoder (``cfg.transcoder``) encodes ``f(x) = sigma(W_enc^T x +
    b_enc)`` and predicts a target ``y_hat = f(x) D + b_dec [+ x W_skip]``.
    Its target has its own scale (``target_scale``); latent activations are
    in target units.

    Args:
        cfg: Model configuration. If ``None``, ``kwargs`` are passed to
            :class:`SAEConfig`.
        **kwargs: Fields of :class:`SAEConfig`, used when ``cfg`` is ``None``.

    Attributes:
        cfg: The model configuration.
        activation: The activation module.
        decoder: The decoder module.
        k_current: The k used for selection; differs from ``cfg.k`` only during
            k annealing (see :meth:`set_k`).
        sparsity_scale: Multiplier of the sparsity penalty, for warmup.
    """

    def __init__(self, cfg: SAEConfig | None = None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else SAEConfig(**kwargs)
        self.k_current = self.cfg.k
        self.sparsity_scale = 1.0
        d_in, d_sae = self.cfg.d_in, self.cfg.d_sae

        d_out = self.cfg.d_out
        self.w_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_out))
        if self.cfg.skip_connection:
            self.w_skip = nn.Parameter(torch.zeros(d_in, d_out))
        self.activation = ACTIVATIONS[self.cfg.activation](self.cfg)
        self.decoder = DECODERS[self.cfg.decoder](self.cfg)

        self.register_buffer("input_scale", torch.ones(()))
        if self.cfg.transcoder:
            self.register_buffer("target_scale", torch.ones(()))
        self.register_buffer("loss_norm", torch.full((), float("nan")))
        self.register_buffer("tokens_since_active", torch.zeros(d_sae, dtype=torch.long))
        if self.cfg.matryoshka_prefixes is not None:
            bounds = torch.tensor(self.cfg.matryoshka_prefixes[:-1], dtype=torch.long)
            self.register_buffer("_latent_group", torch.bucketize(torch.arange(d_sae), bounds, right=True),
                                 persistent=False)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """
        Data-free initialization.

        The encoder is the transpose of the decoder's initial unit-norm
        directions (Conerly et al., 2024; Gao et al., 2024); a transcoder's
        encoder gets independent random unit directions. Biases and the skip
        connection are zero.
        """
        self.b_enc.zero_()
        self.b_dec.zero_()
        atoms = self.decoder.reset_parameters()
        if self.cfg.transcoder:
            atoms = F.normalize(torch.randn(self.cfg.d_sae, self.cfg.d_in), dim=-1)
        self.w_enc.copy_(rearrange(atoms, "d_sae d_in -> d_in d_sae"))
        if self.cfg.skip_connection:
            self.w_skip.zero_()

    @torch.no_grad()
    def initialize_from_data(
        self,
        sample: Tensor,
        target: Tensor | None = None,
        normalize_input: bool = True,
        b_dec_init: str = "geometric_median",
        calibrate_encoder: bool = True,
    ) -> None:
        """
        Data-dependent initialization from a sample of activations.

        Sets, in order: the input scale (``E||x|| = sqrt(d_in)``; Conerly et
        al., 2024), the decoder bias (geometric median; Gao et al., 2024, App.
        A.1), the fixed loss normalizer, the decoder's data-dependent state
        (e.g. Archetypal candidates), and the encoder scale.

        Args:
            sample: Activations, shape ``(..., d_in)``, with at least
                ``decoder.min_init_tokens`` rows.
            target: Matching targets ``(..., d_out)``; required for
                transcoders, which initialize decoder-side state from them.
            normalize_input: Whether to estimate ``input_scale`` (and
                ``target_scale``).
            b_dec_init: ``"geometric_median"``, ``"mean"`` or ``"zeros"``.
            calibrate_encoder: Rescale the encoder so the initial
                reconstruction matches the input magnitude (Gao et al., 2024,
                App. A.1). See :meth:`calibrate_encoder_scale_`.

        Raises:
            ValueError: If ``b_dec_init`` is unknown, or a transcoder gets no
                ``target``.
        """
        if self.cfg.transcoder and target is None:
            raise ValueError("a transcoder needs target activations to initialize")

        def scale_of(flat: Tensor) -> float:
            if not normalize_input:
                return 1.0
            return math.sqrt(flat.shape[-1]) / flat.norm(dim=-1).mean().clamp_min(1e-8).item()

        x, _ = pack([sample.to(self.device, torch.float32)], "* d_in")
        self.input_scale.fill_(scale_of(x))
        x = x * self.input_scale
        y = x
        if self.cfg.transcoder:
            y, _ = pack([target.to(self.device, torch.float32)], "* d_out")
            self.target_scale.fill_(scale_of(y))
            y = y * self.target_scale

        if b_dec_init == "geometric_median":
            self.b_dec.copy_(geometric_median(y))
        elif b_dec_init == "mean":
            self.b_dec.copy_(reduce(y, "n d -> d", "mean"))
        elif b_dec_init == "zeros":
            self.b_dec.zero_()
        else:
            raise ValueError(f"unknown b_dec_init {b_dec_init!r}")

        centred = y - reduce(y, "n d -> 1 d", "mean")
        self.loss_norm.fill_(reduce(centred**2, "n d -> ", "mean").item())

        atoms = self.decoder.initialize_from_data_(y - self.b_dec)
        if atoms is not None and not self.cfg.transcoder:
            self.w_enc.copy_(rearrange(atoms, "d_sae d_in -> d_in d_sae"))

        if calibrate_encoder:
            self.calibrate_encoder_scale_(x, y)

    @torch.no_grad()
    def calibrate_encoder_scale_(self, x_s: Tensor, y_s: Tensor | None = None) -> None:
        """
        Rescales ``w_enc`` so the initial reconstruction matches the input.

        Uses the least-squares optimal scalar. With ``b_enc = 0``, TopK-style
        selection and ReLU are equivariant to a positive rescaling of the
        encoder, so only activation magnitudes change. Gao et al. (2024, App.
        A.1) use the same initialization; it removes the large initial
        overshoot of the tied initialization.

        A transcoder's encoder is not tied to its decoder, so its random
        initial prediction is uncorrelated with the target and the
        least-squares scalar is close to 0. That still suits TopK-style
        activations (selection is unchanged, and small initial activations
        trained fastest in our tests), but would push every pre-activation of
        a ReLU / JumpReLU transcoder below its threshold, so those match the
        prediction's magnitude to the target's instead.

        Args:
            x_s: Activations in scaled units, shape ``(n, d_in)``.
            y_s: Scaled targets for a transcoder; ``None`` means ``x_s``.
        """
        if self.b_enc.abs().max() > 0:
            return
        was_training = self.training
        self.train()  # training-mode selection
        target = (x_s if y_s is None else y_s) - self.b_dec - self._skip(x_s)
        recon = self._decode(self.activation(self._pre_acts(x_s), self.k_current), self.decoder_weight())
        alpha = calibration_scale(recon, target, match_magnitude=self.cfg.transcoder and not self.uses_k)
        if alpha > 0:
            self.w_enc.mul_(alpha)
        self.train(was_training)

    def decoder_weight(self) -> Tensor:
        """Returns the decoder dictionary, shape ``(d_sae, d_in)``."""
        return self.decoder.weight()

    def pre_optimizer_step(self) -> None:
        """Gradient surgery to run after ``backward()`` and before the step."""
        self.decoder.pre_step()

    def post_optimizer_step(self) -> None:
        """Constraint projections to run after every optimizer step."""
        self.decoder.post_step()

    # ---- trainer protocol (see innards.sae.trainer.SAETrainer) ----

    def batch_tensors(self, batch: Any) -> tuple[Tensor, ...]:
        """
        Turns a data batch into the tensors :meth:`forward` takes, flattened to tokens.

        Args:
            batch: A tensor, an ``(input, target)`` pair, a ``(tensor,)``
                tuple (``TensorDataset``) or a dict with ``"activations"``
                (and ``"targets"``).

        Returns:
            ``(x,)`` for an SAE, ``(x, y)`` for a transcoder.

        Raises:
            ValueError: If a transcoder gets no targets.
            TypeError: For unsupported batch types.
        """
        if isinstance(batch, Tensor):
            x, y = batch, None
        elif isinstance(batch, (tuple, list)):
            x, y = batch[0], (batch[1] if len(batch) > 1 else None)
        elif isinstance(batch, dict):
            x, y = batch["activations"], batch.get("targets")
        else:
            raise TypeError(f"unsupported batch type {type(batch)}")
        x = pack([x], "* d_in")[0]
        if not self.cfg.transcoder:
            return (x,)
        if y is None:
            raise ValueError("a transcoder needs (input, target) batches, e.g. RowStream(..., target=...)")
        return x, pack([y], "* d_out")[0]

    @property
    def min_init_tokens(self) -> int:
        """Minimum number of tokens :meth:`initialize_from_data` needs."""
        return self.decoder.min_init_tokens

    @property
    def uses_k(self) -> bool:
        """Whether the activation selects a fixed number ``k`` of latents."""
        return self.activation.uses_k

    @property
    def lr_d_sae(self) -> int:
        """Dictionary size that sets the default learning rate."""
        return self.cfg.d_sae

    def anneal_k(self, progress: float, k_initial: int | None = None) -> None:
        """
        Sets k for linear k annealing.

        Args:
            progress: Fraction of the annealing done, in ``[0, 1]``.
            k_initial: Starting k; ``None`` uses ``min(d_sae, 10 * d_in)`` as
                in ``sparsify``.
        """
        if self.uses_k:
            k0 = k_initial or min(self.cfg.d_sae, 10 * self.cfg.d_in)
            self.set_k(round(k0 * (1 - progress) + self.cfg.k * progress))

    def config_json(self) -> str:
        """The configuration as JSON (see :meth:`from_config_json`)."""
        return self.cfg.to_json()

    @classmethod
    def from_config_json(cls, s: str) -> "SAE":
        """Builds an untrained model from :meth:`config_json` output."""
        return cls(SAEConfig.from_json(s))

    @torch.no_grad()
    def eval_sums(self, out: SAEOutput, *tensors: Tensor) -> dict[str, tuple[float, float, float]]:
        """
        Sums for dataset-level evaluation of one batch.

        Args:
            out: Output of ``forward(*tensors)``.
            *tensors: The batch tensors.

        Returns:
            ``{"": (squared error, total sum of squares, active latents)}``;
            models with several dictionaries return one entry per member.
        """
        y = tensors[-1].float()
        sse = ((out.recon.float() - y) ** 2).sum().item()
        sst = ((y - reduce(y, "n d -> 1 d", "mean")) ** 2).sum().item()
        return {"": (sse, sst, out.latent_counts.sum().item())}

    def set_k(self, k: int) -> None:
        """
        Sets the k used for selection, e.g. during k annealing.

        Args:
            k: New k, clipped to ``[1, d_sae]``.
        """
        self.k_current = int(max(1, min(k, self.cfg.d_sae)))

    @property
    def device(self) -> torch.device:
        """Device of the parameters."""
        return self.w_enc.device

    @property
    def dead_mask(self) -> Tensor:
        """Boolean mask of latents dead for ``dead_tokens_threshold`` tokens."""
        return self.tokens_since_active >= self.cfg.dead_tokens_threshold

    @property
    def output_scale(self) -> Tensor:
        """Scale of the reconstructed activation: ``target_scale`` for transcoders, else ``input_scale``."""
        return self.target_scale if self.cfg.transcoder else self.input_scale

    def _scale_in(self, flat_x: Tensor) -> Tensor:
        return flat_x.to(self.w_enc.dtype) * self.input_scale

    def _pre_acts(self, x_s: Tensor) -> Tensor:
        centred = x_s if self.cfg.transcoder else x_s - self.b_dec
        return einsum(centred, self.w_enc, "n d_in, d_in d_sae -> n d_sae") + self.b_enc

    def _skip(self, x_s: Tensor) -> Tensor | float:
        if not self.cfg.skip_connection:
            return 0.0
        return einsum(x_s.to(self.w_skip.dtype), self.w_skip, "n d_in, d_in d_out -> n d_out")

    def _decode(self, code: SparseCode, weight: Tensor, latent_mask: Tensor | None = None) -> Tensor:
        """Decodes without bias, returning ``(n, d_out)`` (see :func:`~innards.sae.activations.decode_code`)."""
        return decode_code(code, weight, latent_mask)

    def _to_dense(self, code: SparseCode) -> Tensor:
        if code.dense is not None:
            return code.dense
        dense = torch.zeros(code.n, self.cfg.d_sae, device=code.vals.device, dtype=code.vals.dtype)
        dense[code.rows, code.cols] = code.vals
        return dense

    @torch.no_grad()
    def _update_stats(self, code: SparseCode) -> Tensor:
        if code.dense is not None:
            counts = (code.dense != 0).sum(0)
        else:
            counts = torch.bincount(code.cols[code.vals != 0], minlength=self.cfg.d_sae)
        if self.training:
            self.tokens_since_active += code.n
            self.tokens_since_active[counts > 0] = 0
            self.activation.update_stats(code)
        return counts

    def _nmse(self, recon: Tensor, target: Tensor) -> Tensor:
        fixed = self.cfg.loss_normalization == "fixed" and bool(torch.isfinite(self.loss_norm))
        return normalized_mse(recon, target, self.loss_norm if fixed else None)

    def _aux_loss(self, pre: Tensor, residual: Tensor, weight: Tensor) -> Tensor:
        """AuxK loss (Gao et al., 2024, Sec. 2.4 and App. A.2): the top
        ``k_aux`` dead latents reconstruct the main residual; normalized by
        the residual's variance and set to 0 if NaN."""
        dead = self.dead_mask
        num_dead = int(dead.sum())
        if num_dead == 0:
            return pre.new_zeros((), dtype=torch.float32)
        code = rowwise_topk(pre, self.activation.score(pre), min(self.cfg.k_aux, num_dead),
                            signed=self.cfg.abs_topk, mask=dead)
        recon_aux = self._decode(code, weight)
        return torch.nan_to_num(normalized_mse(recon_aux.float(), residual), nan=0.0)

    def _matryoshka_loss(self, code: SparseCode, base: Tensor, y_s: Tensor, weight: Tensor) -> Tensor:
        """Mean normalised MSE over nested prefixes (Bussmann et al., 2025);
        ``base`` is the latent-free part of the prediction (bias, skip)."""
        partial = base.float().expand_as(y_s)
        losses = []
        for g in range(len(self.cfg.matryoshka_prefixes)):
            mask = (self._latent_group == g).to(weight.dtype)
            partial = partial + self._decode(code, weight, mask).float()
            losses.append(self._nmse(partial, y_s))
        return torch.stack(losses).mean()

    def _encode_code(self, flat_x: Tensor) -> SparseCode:
        return self.activation(self._pre_acts(self._scale_in(flat_x)), self.k_current)

    def encode(self, x: Tensor) -> Tensor:
        """
        Encodes activations into sparse latents.

        Args:
            x: Activations, shape ``(..., d_in)``.

        Returns:
            Sparse activations, shape ``(..., d_sae)``, in input units (target
            units for a transcoder). In eval mode a BatchTopK SAE uses its
            learned threshold, so the result does not depend on the rest of
            the batch.
        """
        flat_x, ps = pack([x], "* d_in")
        acts = self._to_dense(self._encode_code(flat_x)) / self.output_scale
        [acts] = unpack(acts, ps, "* d_sae")
        return acts

    def pre_activations(self, x: Tensor) -> Tensor:
        """
        Encoder outputs before the activation function, in input units.

        Args:
            x: Activations, shape ``(..., d_in)``.

        Returns:
            Pre-activations, shape ``(..., d_sae)``.
        """
        flat_x, ps = pack([x], "* d_in")
        pre = self._pre_acts(self._scale_in(flat_x)) / self.output_scale
        [pre] = unpack(pre, ps, "* d_sae")
        return pre

    def decode(self, acts: Tensor) -> Tensor:
        """
        Decodes latents back to activation space.

        For a transcoder with a skip connection this omits the skip term,
        which depends on the input; see :meth:`reconstruct`.

        Args:
            acts: Latent activations, shape ``(..., d_sae)``, in input units.

        Returns:
            Reconstruction, shape ``(..., d_out)``.
        """
        flat, ps = pack([acts], "* d_sae")
        weight = self.decoder_weight()
        recon = einsum(flat.to(weight.dtype), weight, "n d_sae, d_sae d_out -> n d_out")
        recon = recon + self.b_dec / self.output_scale
        [recon] = unpack(recon, ps, "* d_out")
        return recon

    def skip(self, x: Tensor) -> Tensor:
        """
        A transcoder's skip-connection term ``x W_skip``, in output units.

        Args:
            x: Activations, shape ``(..., d_in)``.

        Returns:
            Shape ``(..., d_out)``; zeros without a skip connection.
        """
        flat_x, ps = pack([x], "* d_in")
        if not self.cfg.skip_connection:
            out = flat_x.new_zeros(flat_x.shape[0], self.cfg.d_out)
        else:
            out = self._skip(self._scale_in(flat_x)) / self.output_scale
        [out] = unpack(out.to(x.dtype), ps, "* d_out")
        return out

    def reconstruct(self, x: Tensor) -> Tensor:
        """
        Encodes and decodes without computing the loss.

        Args:
            x: Activations, shape ``(..., d_in)``.

        Returns:
            Reconstruction (a transcoder's prediction), shape ``(..., d_out)``,
            in the units and dtype of the input.
        """
        flat_x, ps = pack([x], "* d_in")
        x_s = self._scale_in(flat_x)
        recon_s = self._decode(self.activation(self._pre_acts(x_s), self.k_current), self.decoder_weight())
        recon_s = recon_s + self.b_dec + self._skip(x_s)
        [recon] = unpack((recon_s / self.output_scale).to(x.dtype), ps, "* d_out")
        return recon

    def forward(self, x: Tensor, target: Tensor | None = None, return_acts: bool = True) -> SAEOutput:
        """
        Encodes, decodes and computes the training objective.

        The objective is the normalised MSE (Gao et al., 2024), or the mean
        Matryoshka prefix loss (Bussmann et al., 2025), plus in training mode
        the activation's sparsity penalty (times ``sparsity_scale``),
        ``aux_coeff * AuxK`` and ``multi_topk_coeff * L(factor * k)`` (both
        Gao et al., 2024).

        Args:
            x: Activations, shape ``(..., d_in)``.
            target: For transcoders, the activation to predict,
                ``(..., d_out)`` with the same leading shape as ``x``.
            return_acts: Build the dense ``(..., d_sae)`` activation tensor.

        Returns:
            An :class:`SAEOutput`.

        Raises:
            ValueError: If a transcoder gets no target or an SAE gets one.
            RuntimeError: If an Archetypal SAE is trained before
                :meth:`initialize_from_data`.
        """
        cfg = self.cfg
        if cfg.transcoder != (target is not None):
            raise ValueError("transcoders need a target; SAEs reconstruct their input and take none")
        flat_x, ps = pack([x], "* d_in")
        x_s = self._scale_in(flat_x)
        y_s = x_s if target is None else pack([target], "* d_out")[0].to(self.w_enc.dtype) * self.output_scale
        pre = self._pre_acts(x_s)
        weight = self.decoder_weight()

        code = self.activation(pre, self.k_current)
        base = self.b_dec + self._skip(x_s)
        recon_s = self._decode(code, weight) + base
        latent_counts = self._update_stats(code)

        y_f, recon_f = y_s.float(), recon_s.float()
        nmse = self._nmse(recon_f, y_f)
        metrics: dict[str, Tensor] = {
            "nmse": nmse.detach(),
            "l0": (latent_counts.sum() / code.n).float(),
        }

        loss = nmse
        if cfg.matryoshka_prefixes is not None:
            loss = self._matryoshka_loss(code, base, y_f, weight)
            metrics["matryoshka_loss"] = loss.detach()

        if self.training:
            penalty = self.activation.penalty(pre, code, weight)
            if penalty is not None:
                loss = loss + self.sparsity_scale * penalty
                metrics["sparsity_loss"] = penalty.detach()
            if cfg.aux_coeff > 0:
                aux = self._aux_loss(pre, (y_f - recon_f).detach(), weight)
                loss = loss + cfg.aux_coeff * aux
                metrics["aux_loss"] = aux.detach()
            if cfg.multi_topk:
                code_m = self.activation(pre, min(self.k_current * cfg.multi_topk_factor, cfg.d_sae))
                multi = self._nmse((self._decode(code_m, weight) + base).float(), y_f)
                loss = loss + cfg.multi_topk_coeff * multi
                metrics["multi_topk_nmse"] = multi.detach()

        metrics["loss"] = loss.detach()

        [recon] = unpack((recon_s / self.output_scale).to(x.dtype), ps, "* d_out")
        acts = None
        if return_acts:
            [acts] = unpack(self._to_dense(code) / self.output_scale, ps, "* d_sae")
        return SAEOutput(
            acts=acts,
            recon=recon,
            loss=loss,
            metrics=metrics,
            latent_counts=latent_counts
        )

    @torch.no_grad()
    def folded_state_dict(self) -> dict[str, Tensor]:
        """
        Returns a state dict with the input (and target) scales folded into the weights.

        With input scale ``s_in`` and output scale ``s_out`` (equal for an
        SAE), encoder pre-activations become ``pre / s_out`` (same selection,
        activations in output units), so ``W_enc`` is multiplied by
        ``s_in / s_out`` (unchanged for an SAE) and ``b_enc``, ``b_dec``, the
        fixed loss normalizer and the activation's thresholds are divided by
        ``s_out``. The decoder is unchanged, so unit-norm atoms stay unit
        norm; a skip connection is multiplied by ``s_in / s_out``.

        Returns:
            A state dict whose model works on raw activations.
        """
        self.decoder.refresh_()
        s_in, s_out = self.input_scale.clone(), self.output_scale.clone()
        sd = {k: v.detach().clone().contiguous() for k, v in self.state_dict().items()}
        if self.cfg.transcoder:
            sd["w_enc"] = sd["w_enc"] * (s_in / s_out)
            sd["target_scale"] = torch.ones_like(s_out)
        if self.cfg.skip_connection:
            sd["w_skip"] = sd["w_skip"] * (s_in / s_out)
        sd["b_enc"] = sd["b_enc"] / s_out
        sd["b_dec"] = sd["b_dec"] / s_out
        sd["loss_norm"] = sd["loss_norm"] / s_out**2
        sd["input_scale"] = torch.ones_like(s_in)
        self.activation.fold_(sd, "activation.", s_out)
        self.decoder.fold_(sd, "decoder.", s_out)
        return sd

    @torch.no_grad()
    def fold_input_scale_(self) -> None:
        """In-place version of :meth:`folded_state_dict`."""
        self.load_state_dict(self.folded_state_dict())

    def save(self, path: str | Path, extra_metadata: dict[str, str] | None = None) -> Path:
        """
        Saves the SAE to a safetensors file.

        Weights are stored with the input scale folded in; the config is
        stored as JSON in the file's metadata.

        Args:
            path: Destination ``.safetensors`` path. Parent directories are
                created.
            extra_metadata: Additional string metadata (e.g. step, model name,
                hook point).

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sd = {k: v.cpu() for k, v in self.folded_state_dict().items()}
        metadata = {"format": FORMAT_NAME, "format_version": FORMAT_VERSION, "config": self.cfg.to_json()}
        if extra_metadata:
            metadata.update({str(k): str(v) for k, v in extra_metadata.items()})
        save_file(sd, str(path), metadata=metadata)
        return path

    @staticmethod
    def read_metadata(path: str | Path) -> dict[str, str]:
        """
        Reads the string metadata of a saved SAE.

        Args:
            path: Path to a ``.safetensors`` file.

        Returns:
            The metadata dictionary.
        """
        with safe_open(str(path), framework="pt") as f:
            return dict(f.metadata() or {})

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "SAE":
        """
        Loads an SAE saved with :meth:`save`.

        Args:
            path: Path to the ``.safetensors`` file.
            device: Device to load onto.

        Returns:
            The SAE in eval mode.

        Raises:
            ValueError: If the file was not written by :meth:`save`.
        """
        meta = cls.read_metadata(path)
        if meta.get("format") != FORMAT_NAME or meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path} is not a {FORMAT_NAME} v{FORMAT_VERSION} file (metadata: {meta})")
        model = cls(SAEConfig.from_json(meta["config"]))
        model.load_state_dict(load_file(str(path), device="cpu"))
        return model.to(device).eval()
