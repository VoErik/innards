"""TopK / BatchTopK SAEs.

The model is a TopK SAE (Makhzani & Frey, 2013; Gao et al., 2024) with optional
BatchTopK selection, AbsTopK (signed) activations, Matryoshka losses, Multi-TopK,
and an Archetypal (convex-hull-constrained) decoder. Optimization-side tricks
(learning-rate schedule, optimizers, k annealing, KL fine-tuning, logging) live
in :mod:`sae.trainer`.

Note (Erik): I tried to include all tricks and improvements for the TopK variants.
If any are missing feel free to reach out.

Note (Erik): I chose to focus on TopK variants as I could not make out any noteworthy
qualitative difference between SAE variants when I conducted research during my Masters.
Might add other architectures in the future.

Typical usage::

    sae = TopKSAE(d_in=768, d_sae=32768, k=64, as_batchtopk=True)
    ...  # train with sae.trainer.SAETrainer
    sae.save("sae.safetensors")
    sae = TopKSAE.load("sae.safetensors", device="cuda")
    acts = sae.encode(x)  # (..., d_sae), x in raw activation units

References:
    Makhzani, A. & Frey, B. (2013). k-Sparse Autoencoders.
        arXiv:1312.5663.
    Bricken, T. et al. (2023). Towards Monosemanticity: Decomposing Language
        Models With Dictionary Learning. Transformer Circuits Thread.
    Conerly, T., Templeton, A., Bricken, T., Marcus, J. & Henighan, T. (2024).
        Update on how we train SAEs. Circuits Updates, April 2024.
    Gao, L. et al. (2024). Scaling and evaluating sparse autoencoders.
        arXiv:2406.04093.
    Lieberum, T. et al. (2024). Gemma Scope: Open Sparse Autoencoders
        Everywhere All At Once on Gemma 2. arXiv:2408.05147.
    Bussmann, B., Leask, P. & Nanda, N. (2024). BatchTopK Sparse Autoencoders.
        arXiv:2412.06410.
    Chanin, D. et al. (2024). A is for Absorption: Studying Feature Splitting
        and Absorption in Sparse Autoencoders. arXiv:2409.14507.
    Bussmann, B., Nabeshima, N., Karvonen, A. & Nanda, N. (2025). Learning
        Multi-Level Features with Matryoshka Sparse Autoencoders.
        arXiv:2503.17547.
    Fel, T. et al. (2025). Archetypal SAE: Adaptive and Stable Dictionary
        Learning for Concept Extraction in Large Vision Models. ICML 2025,
        arXiv:2502.12892.
    Zhu, X., Khalili, M. M. & Zhu, Z. (2025). AbsTopK: Rethinking Sparse
        Autoencoders For Bidirectional Features. ICLR 2026, arXiv:2510.00404.
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
from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import (
    einsum,
    pack,
    rearrange,
    reduce,
    repeat,
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

FORMAT_NAME = "sae"
FORMAT_VERSION = "1"


@dataclass
class SAEConfig:
    """
    Architecture and loss configuration of a :class:`TopKSAE`.

    Attributes:
        d_in: Width of the input activations.
        d_sae: Number of latents (dictionary size).
        k: Target number of active latents per token. For BatchTopK this is the
            average over the batch.
        as_batchtopk: Select the top ``k * batch_size`` activations over the
            whole batch instead of the top ``k`` per token, and use a learned
            global threshold at inference (Bussmann, Leask & Nanda, 2024).
        abs_topk: Select latents by absolute pre-activation and keep their
            sign, without a ReLU, so one latent can encode both directions of a
            feature (Zhu, Khalili & Zhu, 2025).
        k_aux: Number of dead latents used by the AuxK loss. ``None`` resolves
            to ``d_in // 2`` (Gao et al., 2024, App. A.2), capped at ``d_sae``.
        aux_coeff: Weight of the AuxK loss; 1/32 as in Gao et al. (2024).
        dead_tokens_threshold: A latent is dead after this many tokens without
            firing; 10M as in Gao et al. (2024).
        threshold_ema_decay: EMA decay for the BatchTopK inference threshold.
            Bussmann et al. (2024) use a plain average of the per-batch minimum
            selected activation; an EMA of the same quantity follows the
            threshold when activations drift during training.
        multi_topk: Add the Multi-TopK loss L(k) + L(4k)/8 (Gao et al., 2024,
            Sec. 5.3.1). Off by default: it costs a little MSE at ``k``.
        multi_topk_factor: Multiplier of ``k`` for the Multi-TopK term.
        multi_topk_coeff: Weight of the Multi-TopK term.
        matryoshka: Train nested dictionary prefixes to reconstruct the input on
            their own (Bussmann, Nabeshima, Karvonen & Nanda, 2025), which
            reduces feature absorption (Chanin et al., 2024).
        matryoshka_prefixes: Prefix sizes. ``None`` with ``matryoshka=True``
            uses ``[d_sae/64, d_sae/16, d_sae/4, d_sae]``. ``d_sae`` is always appended.
        archetypal: Constrain decoder atoms to (a relaxation of) the convex
            hull of data candidates: ``D = (W C + Lambda) * exp(s)`` with
            ``W`` row-stochastic and ``||Lambda_i|| <= delta`` (RA-SAE; Fel et
            al., 2025). Improves dictionary stability across seeds.
        archetypal_n_candidates: Number of candidate points ``C`` (k-means
            centroids of the centred data). ``None`` resolves to
            ``min(d_sae, 8192)``; ``W`` has ``d_sae * n_candidates`` entries,
            so memory grows quickly.
        archetypal_delta: Radius of the relaxation ball; 1.0 is the default of
            the reference implementation (KempnerInstitute/overcomplete).
        archetypal_learn_scale: Learn the global multiplier ``exp(s)``.
        loss_normalization: ``"batch"`` divides the MSE by the variance of each
            batch; ``"fixed"`` divides by a constant computed once from the
            initialization sample, as in Gao et al. (2024, App. A.6).
    """

    d_in: int
    d_sae: int
    k: int

    as_batchtopk: bool = False
    abs_topk: bool = False

    k_aux: int | None = None
    aux_coeff: float = 1.0 / 32.0
    dead_tokens_threshold: int = 10_000_000

    threshold_ema_decay: float = 0.99

    multi_topk: bool = False
    multi_topk_factor: int = 4
    multi_topk_coeff: float = 1.0 / 8.0

    matryoshka: bool = False
    matryoshka_prefixes: tuple[int, ...] | None = None

    archetypal: bool = False
    archetypal_n_candidates: int | None = None
    archetypal_delta: float = 1.0
    archetypal_learn_scale: bool = True

    loss_normalization: str = "batch"

    def __post_init__(self) -> None:
        if not (0 < self.k <= self.d_sae):
            raise ValueError(f"k must be in (0, d_sae], got k={self.k}")
        if self.loss_normalization not in ("batch", "fixed"):
            raise ValueError(f"loss_normalization must be 'batch' or 'fixed', got {self.loss_normalization!r}")
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

        if self.archetypal and self.archetypal_n_candidates is None:
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


class SparseCode(NamedTuple):
    """
    Row-sorted COO representation of an ``(n, d_sae)`` activation matrix.

    Attributes:
        rows: Row (token) index of each non-zero, shape ``(nnz,)``.
        cols: Latent index of each non-zero, shape ``(nnz,)``.
        vals: Activation values, shape ``(nnz,)``.
        offsets: Start of each row's segment in ``cols``/``vals``, shape ``(n,)``.
        n: Number of rows.
    """

    rows: Tensor
    cols: Tensor
    vals: Tensor
    offsets: Tensor
    n: int


@dataclass
class SAEOutput:
    """
    Result of :meth:`TopKSAE.forward`.

    Attributes:
        acts: Sparse activations ``(..., d_sae)`` in input units, or ``None``
            if ``return_acts=False``.
        recon: Reconstruction ``(..., d_in)`` in input units.
        loss: Scalar training objective.
        metrics: Detached scalar metrics (``nmse``, ``l0``, ``loss`` and,
            when active, ``aux_loss``, ``matryoshka_loss``, ``multi_topk_nmse``).
        latent_counts: Number of tokens each latent fired on, shape ``(d_sae,)``.
    """

    acts: Tensor | None
    recon: Tensor
    loss: Tensor
    metrics: dict[str, Tensor]
    latent_counts: Tensor

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


@torch.no_grad()
def kmeans(points: Tensor, n_clusters: int, n_iter: int = 25, seed: int = 0, chunk: int = 8192) -> Tensor:
    """
    Lloyd's k-means with k-means++-style random initialization from the data.

    Produces the candidate points of the Archetypal decoder.

    Args:
        points: Data, shape ``(n, d)`` with ``n >= n_clusters``.
        n_clusters: Number of centroids.
        n_iter: Number of Lloyd iterations.
        seed: Seed for the initial centroid draw.
        chunk: Rows per chunk when computing distances, to bound memory.

    Returns:
        Centroids, shape ``(n_clusters, d)``.

    Raises:
        ValueError: If there are fewer points than clusters.
    """
    points = points.float()
    n, d = points.shape
    if n < n_clusters:
        raise ValueError(f"need at least {n_clusters} points for k-means, got {n}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    centroids = points[torch.randperm(n, generator=g)[:n_clusters].to(points.device)].clone()
    for _ in range(n_iter):
        assign = torch.cat([torch.cdist(c, centroids).argmin(-1) for c in points.split(chunk)])
        sums = torch.zeros_like(centroids).index_add_(0, assign, points)
        counts = torch.bincount(assign, minlength=n_clusters).unsqueeze(-1)
        empty = counts.squeeze(-1) == 0
        centroids = torch.where(empty.unsqueeze(-1), centroids, sums / counts.clamp_min(1))
        if empty.any(): # re-seed empty clusters with random points
            idx = torch.randint(0, n, (int(empty.sum()),), generator=g).to(points.device)
            centroids[empty] = points[idx]
    return centroids


def _offsets_from_rows(rows: Tensor, n: int) -> Tensor:
    counts = torch.bincount(rows, minlength=n)
    return torch.cumsum(counts, 0) - counts


class TopKSAE(nn.Module):
    """
    TopK sparse autoencoder with optional BatchTopK, AbsTopK, Matryoshka,
    Multi-TopK and Archetypal decoder.

    The encoder is ``f(x) = sigma(W_enc^T (x - b_dec) + b_enc)`` with the
    pre-encoder bias of Bricken et al. (2023) and ``sigma`` one of TopK
    (Makhzani & Frey, 2013; Gao et al., 2024), BatchTopK (Bussmann et al.,
    2024) or AbsTopK (Zhu et al., 2025). The decoder is
    ``x_hat = f(x) W_dec + b_dec`` with unit-norm atoms (Gao et al., 2024), or
    an Archetypal dictionary (Fel et al., 2025).

    Internally the model works on ``x * input_scale``, a single scalar chosen
    so that ``E||x|| = sqrt(d_in)`` (Conerly et al., 2024). Everything the
    public API returns is in input units, and :meth:`save` folds the scale into
    the weights (as done for Gemma Scope; Lieberum et al., 2024).

    Args:
        cfg: Model configuration. If ``None``, ``kwargs`` are passed to
            :class:`SAEConfig`.
        **kwargs: Fields of :class:`SAEConfig`, used when ``cfg`` is ``None``.

    Attributes:
        cfg: The model configuration.
        k_current: The k used for selection; differs from ``cfg.k`` only during
            k annealing (see :meth:`set_k`).
    """

    def __init__(self, cfg: SAEConfig | None = None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else SAEConfig(**kwargs)
        self.k_current = self.cfg.k
        d_in, d_sae = self.cfg.d_in, self.cfg.d_sae

        self.w_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        if self.cfg.archetypal:
            n_cand = self.cfg.archetypal_n_candidates
            self.arch_weight = nn.Parameter(torch.empty(d_sae, n_cand))
            self.arch_relax = nn.Parameter(torch.zeros(d_sae, d_in))
            log_scale = torch.zeros(())
            if self.cfg.archetypal_learn_scale:
                self.arch_log_scale = nn.Parameter(log_scale)
            else:
                self.register_buffer("arch_log_scale", log_scale)
            self.register_buffer("arch_points", torch.zeros(n_cand, d_in))
            self.register_buffer("arch_initialized", torch.zeros((), dtype=torch.bool))
            # Fused dictionary, used at inference and stored in saved files.
            self.register_buffer("w_dec", torch.zeros(d_sae, d_in))
        else:
            self.w_dec = nn.Parameter(torch.empty(d_sae, d_in))

        self.register_buffer("input_scale", torch.ones(()))
        self.register_buffer("loss_norm", torch.full((), float("nan")))
        self.register_buffer("tokens_since_active", torch.zeros(d_sae, dtype=torch.long))
        # BatchTopK inference threshold (scaled units); NaN until estimated.
        self.register_buffer("threshold", torch.full((), float("nan")))
        if self.cfg.matryoshka_prefixes is not None:
            self.register_buffer(
                "_group_bounds",
                torch.tensor(self.cfg.matryoshka_prefixes[:-1], dtype=torch.long),
                persistent=False,
            )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Data-free initialization.

        Decoder atoms are random unit vectors and the encoder is their
        transpose (Conerly et al., 2024; Gao et al., 2024). Biases are zero.
        For the Archetypal decoder, atom ``i`` starts on candidate
        ``i mod n_candidates`` plus small noise; the candidates themselves are set by
        :meth:`initialize_from_data`.
        """
        self.b_enc.zero_()
        self.b_dec.zero_()
        if self.cfg.archetypal:
            d_sae, n_cand = self.arch_weight.shape
            one_hot = F.one_hot(torch.arange(d_sae) % n_cand, n_cand).float()
            self.arch_weight.copy_(one_hot + 0.01 * torch.rand(d_sae, n_cand))
            self.arch_relax.zero_()
            self.arch_log_scale.zero_()
            self.project_archetypal_()
            atoms = F.normalize(torch.randn(d_sae, self.cfg.d_in), dim=-1)
        else:
            nn.init.normal_(self.w_dec)
            self.normalize_decoder()
            atoms = self.w_dec
        self.w_enc.copy_(rearrange(atoms, "d_sae d_in -> d_in d_sae"))

    @torch.no_grad()
    def initialize_from_data(
        self,
        sample: Tensor,
        normalize_input: bool = True,
        b_dec_init: str = "geometric_median",
        calibrate_encoder: bool = True,
    ) -> None:
        """
        Data-dependent initialization from a sample of activations.

        Sets, in order: the input scale (``E||x|| = sqrt(d_in)``; Conerly et
        al., 2024), the decoder bias (geometric median; Gao et al., 2024, App.
        A.1), the fixed loss normalizer, the Archetypal candidates (k-means
        centroids of the centred data; Fel et al., 2025), and the encoder
        scale.

        Args:
            sample: Activations, shape ``(..., d_in)``. For the Archetypal
                decoder it needs at least ``archetypal_n_candidates`` rows;
                several times more gives better centroids.
            normalize_input: Whether to estimate ``input_scale``.
            b_dec_init: ``"geometric_median"``, ``"mean"`` or ``"zeros"``.
            calibrate_encoder: Rescale the encoder so the initial
                reconstruction matches the input magnitude (Gao et al., 2024,
                App. A.1). See :meth:`calibrate_encoder_scale_`.

        Raises:
            ValueError: If ``b_dec_init`` is unknown.
        """
        flat, _ = pack([sample.to(self.device, torch.float32)], "* d_in")
        if normalize_input:
            mean_norm = flat.norm(dim=-1).mean()
            self.input_scale.fill_(math.sqrt(self.cfg.d_in) / mean_norm.clamp_min(1e-8).item())
        else:
            self.input_scale.fill_(1.0)
        flat = flat * self.input_scale

        if b_dec_init == "geometric_median":
            self.b_dec.copy_(geometric_median(flat))
        elif b_dec_init == "mean":
            self.b_dec.copy_(reduce(flat, "n d -> d", "mean"))
        elif b_dec_init == "zeros":
            self.b_dec.zero_()
        else:
            raise ValueError(f"unknown b_dec_init {b_dec_init!r}")

        centred = flat - reduce(flat, "n d -> 1 d", "mean")
        self.loss_norm.fill_(reduce(centred**2, "n d -> ", "mean").item())

        if self.cfg.archetypal:
            self.arch_points.copy_(kmeans(flat - self.b_dec, self.cfg.archetypal_n_candidates))
            self.arch_initialized.fill_(True)
            self.refresh_decoder_cache()
            self.w_enc.copy_(rearrange(F.normalize(self.w_dec, dim=-1), "d_sae d_in -> d_in d_sae"))

        if calibrate_encoder:
            self.calibrate_encoder_scale_(flat)

    @torch.no_grad()
    def calibrate_encoder_scale_(self, x_s: Tensor) -> None:
        """
        Rescales ``w_enc`` so the initial reconstruction matches the input.

        Uses the least-squares optimal scalar. With ``b_enc = 0``, TopK,
        BatchTopK and AbsTopK selection are invariant to a positive rescaling
        of the encoder, so only the activation magnitudes change. Gao et al.
        (2024, App. A.1) use the same initialization; they report no effect
        or a weakly negative one at convergence, but it removes the large
        initial overshoot of the tied initialization.

        Args:
            x_s: Activations in scaled units, shape ``(n, d_in)``.
        """
        if self.b_enc.abs().max() > 0:
            return
        was_training = self.training
        self.train() # selection by top-k
        target = x_s - self.b_dec
        code = self._select(self._pre_acts(x_s))
        recon = self._decode_sparse(code.cols, code.vals, code.offsets)
        alpha = (recon * target).sum() / (recon * recon).sum().clamp_min(1e-8)
        if alpha > 0:
            self.w_enc.mul_(alpha)
        self.train(was_training)

    def decoder_weight(self) -> Tensor:
        """
        Returns the decoder dictionary, shape ``(d_sae, d_in)``.

        For the Archetypal decoder this is ``(W C + Lambda) * exp(s)`` in
        training mode and the cached fused dictionary in eval mode.
        """
        if not self.cfg.archetypal or not self.training:
            return self.w_dec
        d = einsum(self.arch_weight, self.arch_points, "d_sae n_cand, n_cand d_in -> d_sae d_in")
        return (d + self.arch_relax) * self.arch_log_scale.exp()

    @torch.no_grad()
    def refresh_decoder_cache(self) -> None:
        """Recomputes the fused Archetypal dictionary stored in ``w_dec``."""
        if self.cfg.archetypal:
            was_training = self.training
            self.training = True
            self.w_dec.copy_(self.decoder_weight())
            self.training = was_training

    def train(self, mode: bool = True) -> "TopKSAE":
        """
        Sets training mode and refreshes the fused Archetypal dictionary on eval.

        Args:
            mode: Training mode flag.

        Returns:
            self
        """
        super().train(mode)
        if not mode:
            self.refresh_decoder_cache()
        return self

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """
        Projects decoder atoms to unit norm in place (Gao et al., 2024).

        No-op for the Archetypal decoder.
        """
        if self.cfg.archetypal:
            return
        norm = self.w_dec.norm(dim=-1, keepdim=True)
        self.w_dec.div_(norm.clamp_min(1e-8))

    @torch.no_grad()
    def project_archetypal_(self) -> None:
        """
        Projects the Archetypal parameters onto their constraint sets.

        ``W`` is made row-stochastic by a ReLU followed by row normalization,
        and each row of ``Lambda`` is projected onto the L2 ball of radius
        ``delta``, following the reference implementation of Fel et al. (2025).

        No-op for non-archetypal decoders.
        """
        if not self.cfg.archetypal:
            return
        w = F.relu(self.arch_weight)
        self.arch_weight.copy_(w / (w.sum(dim=-1, keepdim=True) + 1e-8))
        norm = self.arch_relax.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.arch_relax.mul_(torch.clamp(self.cfg.archetypal_delta / norm, max=1.0))

    @torch.no_grad()
    def remove_decoder_gradient_parallel_component(self) -> None:
        """
        Removes the component of each atom's gradient parallel to the atom.

        Keeps updates tangent to the unit sphere (Gao et al., 2024, App. A.3).
        No-op for the Archetypal decoder.
        """
        if self.cfg.archetypal or self.w_dec.grad is None:
            return
        proj = einsum(self.w_dec.grad, self.w_dec, "d_sae d_in, d_sae d_in -> d_sae")
        self.w_dec.grad.sub_(einsum(proj, self.w_dec, "d_sae, d_sae d_in -> d_sae d_in"))

    def pre_optimizer_step(self) -> None:
        """Gradient surgery to run after ``backward()`` and before the step."""
        self.remove_decoder_gradient_parallel_component()

    def post_optimizer_step(self) -> None:
        """Constraint projections to run after every optimizer step."""
        self.normalize_decoder()
        self.project_archetypal_()

    def set_k(self, k: int) -> None:
        """Sets the k used for selection, e.g. during k annealing.

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

    def _scale_in(self, flat_x: Tensor) -> Tensor:
        return flat_x.to(self.w_enc.dtype) * self.input_scale

    def _pre_acts(self, x_s: Tensor) -> Tensor:
        return einsum(x_s - self.b_dec, self.w_enc, "n d_in, d_in d_sae -> n d_sae") + self.b_enc

    def _score(self, pre: Tensor) -> Tensor:
        return pre.abs() if self.cfg.abs_topk else F.relu(pre)

    def _rowwise_topk(self, pre: Tensor, k: int, mask: Tensor | None = None) -> SparseCode:
        n = pre.shape[0]
        score = self._score(pre)
        if mask is not None:
            score = score.masked_fill(~mask, float("-inf"))
        top_score, cols = score.topk(k, dim=-1, sorted=False)
        vals = pre.gather(-1, cols) if self.cfg.abs_topk else top_score
        vals = torch.where(torch.isfinite(top_score), vals, torch.zeros_like(vals))
        rows = repeat(torch.arange(n, device=pre.device), "n -> (n k)", k=k)
        offsets = torch.arange(0, n * k, k, device=pre.device)
        return SparseCode(
            rows=rows,
            cols=rearrange(cols, "n k -> (n k)"),
            vals=rearrange(vals, "n k -> (n k)"),
            offsets=offsets,
            n=n
        )

    def _batch_topk(self, pre: Tensor, k: int) -> SparseCode:
        n, d_sae = pre.shape
        flat_pre = rearrange(pre, "n d_sae -> (n d_sae)")
        score = self._score(flat_pre)
        _, idx = score.topk(min(k * n, score.numel()), sorted=False)
        idx, _ = idx.sort()
        vals = flat_pre[idx] if self.cfg.abs_topk else score[idx]
        rows, cols = idx // d_sae, idx % d_sae
        return SparseCode(
            rows=rows,
            cols=cols,
            vals=vals,
            offsets=_offsets_from_rows(rows, n),
            n=n
        )


    def _threshold_select(self, pre: Tensor) -> SparseCode:
        n = pre.shape[0]
        thr = self.threshold.clamp_min(0.0).to(pre.dtype)
        score = pre.abs() if self.cfg.abs_topk else pre
        rows, cols = torch.nonzero(score > thr, as_tuple=True)  # row-major order
        vals = pre[rows, cols]
        return SparseCode(
            rows=rows,
            cols=cols,
            vals=vals,
            offsets=_offsets_from_rows(rows, n),
            n=n
        )

    def _select(self, pre: Tensor, k: int | None = None) -> SparseCode:
        k = self.k_current if k is None else min(k, self.cfg.d_sae)
        if not self.cfg.as_batchtopk:
            return self._rowwise_topk(pre, k)
        use_threshold = (not self.training) and k == self.cfg.k and bool(torch.isfinite(self.threshold))
        return self._threshold_select(pre) if use_threshold else self._batch_topk(pre, k)

    def _decode_sparse(self, cols: Tensor, vals: Tensor, offsets: Tensor, weight: Tensor | None = None) -> Tensor:
        """Sparse decode without bias, touching only active rows (Gao et al.,
        2024, App. D.2 motivate sparse kernels; ``embedding_bag`` is our
        portable substitute). Returns ``(n, d_in)``."""
        weight = self.decoder_weight() if weight is None else weight
        return F.embedding_bag(cols, weight, offsets, mode="sum", per_sample_weights=vals.to(weight.dtype))

    def _to_dense(self, code: SparseCode) -> Tensor:
        dense = torch.zeros(code.n, self.cfg.d_sae, device=code.vals.device, dtype=code.vals.dtype)
        dense[code.rows, code.cols] = code.vals
        return dense

    @torch.no_grad()
    def _update_stats(self, code: SparseCode) -> Tensor:
        active = code.vals != 0
        counts = torch.bincount(code.cols[active], minlength=self.cfg.d_sae)
        if self.training:
            self.tokens_since_active += code.n
            self.tokens_since_active[counts > 0] = 0
            if self.cfg.as_batchtopk:
                mags = code.vals[active].abs()
                if mags.numel() > 0:
                    batch_min = mags.min().float()
                    decay = self.cfg.threshold_ema_decay
                    new = torch.where(
                        torch.isfinite(self.threshold),
                        decay * self.threshold + (1 - decay) * batch_min,
                        batch_min,
                    )
                    self.threshold.copy_(new)
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
        code = self._rowwise_topk(pre, min(self.cfg.k_aux, num_dead), mask=dead)
        recon_aux = self._decode_sparse(code.cols, code.vals, code.offsets, weight)
        return torch.nan_to_num(normalized_mse(recon_aux.float(), residual), nan=0.0)

    def _matryoshka_loss(self, code: SparseCode, x_s: Tensor, weight: Tensor) -> Tensor:
        """Mean normalised MSE over nested prefixes (Bussmann et al., 2025)."""
        groups = torch.bucketize(code.cols, self._group_bounds, right=True)
        partial = self.b_dec.float().expand_as(x_s)
        losses = []
        for g in range(len(self.cfg.matryoshka_prefixes)):
            vals_g = code.vals * (groups == g).to(code.vals.dtype)
            partial = partial + self._decode_sparse(code.cols, vals_g, code.offsets, weight).float()
            losses.append(self._nmse(partial, x_s))
        return torch.stack(losses).mean()

    def encode(self, x: Tensor) -> Tensor:
        """
        Encodes activations into sparse latents.

        Args:
            x: Activations, shape ``(..., d_in)``.

        Returns:
            Sparse activations, shape ``(..., d_sae)``, in input units. In eval
            mode a BatchTopK SAE uses its learned threshold, so the result
            does not depend on the rest of the batch.
        """
        flat_x, ps = pack([x], "* d_in")
        code = self._select(self._pre_acts(self._scale_in(flat_x)))
        acts = self._to_dense(code) / self.input_scale
        [acts] = unpack(acts, ps, "* d_sae")
        return acts

    def pre_activations(self, x: Tensor) -> Tensor:
        """
        Encoder outputs before TopK selection, in input units.

        Args:
            x: Activations, shape ``(..., d_in)``.

        Returns:
            Pre-activations, shape ``(..., d_sae)``.
        """
        flat_x, ps = pack([x], "* d_in")
        pre = self._pre_acts(self._scale_in(flat_x)) / self.input_scale
        [pre] = unpack(pre, ps, "* d_sae")
        return pre

    def decode(self, acts: Tensor) -> Tensor:
        """
        Decodes latents back to activation space.

        Args:
            acts: Latent activations, shape ``(..., d_sae)``, in input units.

        Returns:
            Reconstruction, shape ``(..., d_in)``.
        """
        flat, ps = pack([acts], "* d_sae")
        weight = self.decoder_weight()
        recon = einsum(flat.to(weight.dtype), weight, "n d_sae, d_sae d_in -> n d_in")
        recon = recon + self.b_dec / self.input_scale
        [recon] = unpack(recon, ps, "* d_in")
        return recon

    def reconstruct(self, x: Tensor) -> Tensor:
        """
        Encodes and decodes without computing the loss.

        Args:
            x: Activations, shape ``(..., d_in)``.

        Returns:
            Reconstruction, shape ``(..., d_in)``, in input units and dtype.
        """
        flat_x, ps = pack([x], "* d_in")
        code = self._select(self._pre_acts(self._scale_in(flat_x)))
        recon_s = self._decode_sparse(code.cols, code.vals, code.offsets) + self.b_dec
        [recon] = unpack((recon_s / self.input_scale).to(x.dtype), ps, "* d_in")
        return recon

    def forward(self, x: Tensor, return_acts: bool = True) -> SAEOutput:
        """
        Encodes, decodes and computes the training objective.

        The objective is the normalised MSE (Gao et al., 2024), or the mean
        Matryoshka prefix loss (Bussmann et al., 2025), plus in training mode
        ``aux_coeff * AuxK`` and ``multi_topk_coeff * L(factor * k)`` (both
        Gao et al., 2024).

        Args:
            x: Activations, shape ``(..., d_in)``.
            return_acts: Build the dense ``(..., d_sae)`` activation tensor.

        Returns:
            An :class:`SAEOutput`.

        Raises:
            RuntimeError: If an Archetypal SAE is trained before
                :meth:`initialize_from_data`.
        """
        cfg = self.cfg
        if cfg.archetypal and self.training and not bool(self.arch_initialized):
            raise RuntimeError("Archetypal SAE needs initialize_from_data() before training")
        flat_x, ps = pack([x], "* d_in")
        x_s = self._scale_in(flat_x)
        pre = self._pre_acts(x_s)
        weight = self.decoder_weight()

        code = self._select(pre)
        recon_s = self._decode_sparse(code.cols, code.vals, code.offsets, weight) + self.b_dec
        latent_counts = self._update_stats(code)

        x_f, recon_f = x_s.float(), recon_s.float()
        nmse = self._nmse(recon_f, x_f)
        metrics: dict[str, Tensor] = {
            "nmse": nmse.detach(),
            "l0": (latent_counts.sum() / code.n).float(),
        }

        main = nmse
        if cfg.matryoshka_prefixes is not None:
            main = self._matryoshka_loss(code, x_f, weight)
            metrics["matryoshka_loss"] = main.detach()
        loss = main

        if self.training:
            if cfg.aux_coeff > 0:
                aux = self._aux_loss(pre, (x_f - recon_f).detach(), weight)
                loss = loss + cfg.aux_coeff * aux
                metrics["aux_loss"] = aux.detach()
            if cfg.multi_topk:
                code_m = self._select(pre, k=self.k_current * cfg.multi_topk_factor)
                recon_m = self._decode_sparse(code_m.cols, code_m.vals, code_m.offsets, weight) + self.b_dec
                multi = self._nmse(recon_m.float(), x_f)
                loss = loss + cfg.multi_topk_coeff * multi
                metrics["multi_topk_nmse"] = multi.detach()

        metrics["loss"] = loss.detach()

        recon = recon_s / self.input_scale
        [recon] = unpack(recon.to(x.dtype), ps, "* d_in")
        acts = None
        if return_acts:
            [acts] = unpack(self._to_dense(code) / self.input_scale, ps, "* d_sae")
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
        Returns a state dict with ``input_scale`` folded into the weights.

        With ``s = input_scale``, encoder pre-activations become ``pre / s``
        (same selection, activations in input units) and the decoder is
        unchanged, so unit-norm atoms stay unit norm. ``b_dec``, the
        BatchTopK threshold and the fixed loss normalizer are rescaled
        accordingly. For the Archetypal decoder, ``w_dec`` holds the fused
        dictionary.

        Returns:
            A state dict whose model works on raw activations.
        """
        self.refresh_decoder_cache()
        s = self.input_scale.clone()
        sd = {k: v.detach().clone().contiguous() for k, v in self.state_dict().items()}
        sd["b_enc"] = sd["b_enc"] / s
        sd["b_dec"] = sd["b_dec"] / s
        sd["threshold"] = sd["threshold"] / s
        sd["loss_norm"] = sd["loss_norm"] / s**2
        sd["input_scale"] = torch.ones_like(s)
        if self.cfg.archetypal:
            # The decoder D = (W C + Lambda) e^s stays fixed; C lives in the
            # scaled space, so re-express it (and Lambda) in input units while
            # compensating with the learned scale.
            sd["arch_points"] = sd["arch_points"] / s
            sd["arch_relax"] = sd["arch_relax"] / s
            sd["arch_log_scale"] = sd["arch_log_scale"] + torch.log(s)
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
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "TopKSAE":
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
        if meta.get("format") != FORMAT_NAME:
            raise ValueError(f"{path} is not a {FORMAT_NAME} file (metadata: {meta})")
        model = cls(SAEConfig.from_json(meta["config"]))
        model.load_state_dict(load_file(str(path), device="cpu"))
        return model.to(device).eval()
