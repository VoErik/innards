"""
Trainer for :class:`sae.model.SAE`.

Two phases:

1. :meth:`SAETrainer.train` trains on reconstruction (normalised MSE plus the
   auxiliary terms configured on the model) from an iterable of activation
   batches, e.g. a :class:`sae.data.ActivationStore` or a ``DataLoader`` over
   cached activations.
2. :meth:`SAETrainer.finetune_kl` optionally fine-tunes the SAE spliced into
   the model on KL + MSE (Karvonen, 2025), which shrinks the downstream loss
   gap at a few percent of the training budget. It works for any model whose
   output can be turned into logits: classifier logits for vision, next-token
   logits for language models.

Default settings are derived from literature. I recommend running initial tests using those
and adapt only when explicitly needed.

References:
    Conerly, T., Templeton, A., Bricken, T., Marcus, J. & Henighan, T. (2024).
        Update on how we train SAEs. Circuits Updates, April 2024.
    Gao, L. et al. (2024). Scaling and evaluating sparse autoencoders.
        arXiv:2406.04093.
    Lieberum, T. et al. (2024). Gemma Scope: Open Sparse Autoencoders
        Everywhere All At Once on Gemma 2. arXiv:2408.05147.
    Braun, D., Taylor, J., Goldowsky-Dill, N. & Sharkey, L. (2024).
        Identifying Functionally Important Features with End-to-End Sparse
        Dictionary Learning. arXiv:2405.12241.
    Karvonen, A. (2025). Revisiting End-To-End Sparse Autoencoder Training:
        A Short Finetune Is All You Need. arXiv:2503.17272.
    EleutherAI (2024-2025). sparsify. github.com/EleutherAI/sparsify.
"""

from __future__ import annotations

import copy
import itertools
import math
import time
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Sequence
)
from dataclasses import (
    asdict,
    dataclass
)
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import (
    pack,
    rearrange,
    reduce
)
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .data import (
    _to_device,
    default_input_fn,
    splice,
)
from .optim import (
    build_optimizers,
    default_lr
)
from .model import (
    SAE,
    SAEConfig
)

LOGGING_BACKENDS: set = {"wandb", "mlflow", "tensorboard"}
CHECKPOINT_FORMAT = "sae-trainer-checkpoint"

@dataclass
class TrainerConfig:
    """
    Configuration of :meth:`SAETrainer.train`.

    Attributes:
        total_tokens: Train for this many tokens, cycling over the data. If
            ``None``, train for ``num_epochs`` passes.
        num_epochs: Passes over the data when ``total_tokens`` is ``None``.
        optimizer: ``"adam"`` (default), ``"muon"`` or ``"signum"``.
        lr: Learning rate; ``None`` uses the optimizer's default from
            :func:`~sae.optim.default_lr`.
        adam_betas: Adam betas (Gao et al., 2024).
        adam_eps: Adam epsilon (Gao et al., 2024).
        warmup_steps: Linear warmup steps, capped at 5% of training.
        decay_frac: Fraction of training at the end over which the learning
            rate decays linearly to 0 (Conerly et al., 2024).
        max_grad_norm: Gradient-norm clipping threshold (Conerly et al., 2024);
            ``None`` disables clipping.
        ema_decay: Weight-EMA decay, e.g. 0.999 (Gao et al., 2024); ``None``
            disables it.
        k_decay_steps: Steps of linear k annealing (TopK-style activations);
            0 disables it.
        k_initial: Starting k for annealing; ``None`` uses
            ``min(d_sae, 10 * d_in)`` as in ``sparsify``.
        sparsity_warmup_steps: Steps over which the sparsity penalty of
            penalty-based activations (JumpReLU, ReLU) ramps linearly from 0
            to its full weight (Conerly et al., 2024); 0 disables it.
        init_tokens: Tokens used for the data-dependent initialisation. The
            Archetypal decoder uses at least ``4 * n_candidates``.
        normalize_input: Estimate the input scale (Conerly et al., 2024).
        b_dec_init: ``"geometric_median"``, ``"mean"`` or ``"zeros"``.
        outlier_norm_factor: Drop tokens with norm above this multiple of the
            median norm; ``None`` disables the filter.
        device: ``"auto"``, ``"cuda"``, ``"cpu"``, ...
        autocast_dtype: ``"auto"`` (bf16 on CUDA if supported), ``"bfloat16"``
            or ``None``.
        compile: Wrap the SAE forward in ``torch.compile``. Data-dependent
            shapes cause graph breaks, so the speed-up is limited.
        seed: Random seed.
        loggers: Any of ``"wandb"``, ``"mlflow"``, ``"tensorboard"``.
        project: wandb project / MLflow experiment name.
        run_name: Run name; ``None`` uses a timestamp.
        log_dir: TensorBoard root directory.
        log_every: Steps between metric logs.
        hist_every: Steps between latent-density histograms.
        dense_threshold: Latents firing on more than this fraction of tokens
            are counted as dense.
        output_dir: Checkpoint directory.
        save_every: Steps between exported SAE files (``sae_step*.safetensors``);
            ``None`` saves only the final one.
        checkpoint_every: Steps between full training checkpoints
            (``checkpoint.pt``, overwritten each time, plus one at the end) for
            :meth:`SAETrainer.from_checkpoint`; ``None`` disables them.
    """

    total_tokens: int | None = None
    num_epochs: int = 1

    optimizer: str = "adam"
    lr: float | None = None
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 6.25e-10
    warmup_steps: int = 1000
    decay_frac: float = 0.2
    max_grad_norm: float | None = 1.0
    ema_decay: float | None = None

    k_decay_steps: int = 0
    k_initial: int | None = None
    sparsity_warmup_steps: int = 0

    init_tokens: int = 65_536
    normalize_input: bool = True
    b_dec_init: str = "geometric_median"
    outlier_norm_factor: float | None = 10.0

    device: str = "auto"
    autocast_dtype: str | None = "auto"
    compile: bool = False
    seed: int = 0

    loggers: tuple[str, ...] = ()
    project: str = "sae"
    run_name: str | None = None
    log_dir: str = "runs"
    log_every: int = 100
    hist_every: int = 2000
    dense_threshold: float = 0.1

    output_dir: str = "checkpoints"
    save_every: int | None = None
    checkpoint_every: int | None = None

# TODO: maybe merge this into the training config?
@dataclass
class KLFinetuneConfig:
    """
    Configuration of :meth:`SAETrainer.finetune_kl` (Karvonen, 2025).

    The loss is ``0.5 * (L_sae + alpha * KL)`` with ``alpha = L_sae / KL``
    recomputed every batch (detached), so both terms have equal weight and the
    total stays on the scale of the reconstruction loss. Karvonen (2025) uses
    the plain MSE for ``L_sae``; here it is the SAE's full objective
    (normalised MSE plus AuxK and any Matryoshka / Multi-TopK terms), which
    keeps the ratio between reconstruction and auxiliary losses unchanged.

    Attributes:
        steps: Number of fine-tuning steps. Karvonen (2025) uses about 5% of
            the training tokens.
        lr: Peak learning rate. ``None`` uses ``lr_scale`` times the training
            learning rate (our default; the paper uses 5e-5 for its setup).
        lr_scale: Multiplier of the training learning rate when ``lr`` is
            ``None``.
        layout: Activation layout at the hook (see ``sae.data.LAYOUTS``).
        exclude_positions: Token positions left unpatched (e.g. CLS / BOS),
            consistent with the positions excluded during training.
        token_mask_fn: Optional ``fn(batch) -> bool (batch, tokens)`` mask of
            tokens to patch, e.g. non-padding tokens; others pass through
            unchanged and do not enter the SAE loss.
        input_fn: Maps a loader batch to model ``(args, kwargs)``.
        output_fn: Maps the model output to logits ``(..., n_classes)``; e.g.
            ``lambda out: out.logits`` for Hugging Face models.
        kl_mask_fn: Optional ``fn(batch) -> bool (...)`` selecting which
            logit rows enter the KL, e.g. non-padding tokens.
        log_every: Steps between metric logs.
    """

    steps: int = 1000
    lr: float | None = None
    lr_scale: float = 0.25
    layout: str = "btd"
    exclude_positions: Sequence[int] = ()
    token_mask_fn: Callable[[Any], Tensor] | None = None
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn
    output_fn: Callable[[Any], Tensor] = lambda out: out
    kl_mask_fn: Callable[[Any], Tensor] | None = None
    log_every: int = 50


class MetricLogger:
    """
    Sends metrics to wandb, MLflow and TensorBoard, and prints to the console.

    Args:
        backends: Any of ``"wandb"``, ``"mlflow"``, ``"tensorboard"``.
        project: wandb project / MLflow experiment name.
        run_name: Run name.
        log_dir: TensorBoard root directory.
        config: Hyperparameters to record.

    Raises:
        ValueError: If a backend name is unknown.
    """

    def __init__(self, backends: Iterable[str], project: str, run_name: str, log_dir: str, config: dict):
        self.backends = [b.lower() for b in backends]
        unknown = set(self.backends) - LOGGING_BACKENDS
        if unknown:
            raise ValueError(f"unknown loggers: {unknown}")
        self._wandb = self._mlflow = self._tb = None

        if "wandb" in self.backends:
            import wandb

            self._wandb = wandb
            wandb.init(project=project, name=run_name, config=config)
        if "mlflow" in self.backends:
            import mlflow

            self._mlflow = mlflow
            mlflow.set_experiment(project)
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({k: str(v)[:500] for k, v in _flatten(config).items()})
        if "tensorboard" in self.backends:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(Path(log_dir) / run_name))
            self._tb.add_text("config", "\n".join(f"    {k}: {v}" for k, v in _flatten(config).items()))

    def log(self, metrics: dict[str, float], step: int) -> None:
        """
        Logs scalar metrics.

        Args:
            metrics: Name to value mapping.
            step: Global step.
        """
        if self._wandb:
            self._wandb.log(metrics, step=step)
        if self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)
        if self._tb:
            for k, v in metrics.items():
                self._tb.add_scalar(k, v, step)
        shown = ("loss", "fve", "l0", "sparsity_loss", "aux_loss", "kl", "dead_frac", "dense_frac", "k", "lr")
        parts = [f"{k.split('/')[-1]} {v:.4g}" for k, v in metrics.items() if k.split("/")[-1] in shown]
        step_str = f"step {step:>7d} | " + " | ".join(parts)
        print("-" * len(step_str), flush=True)
        print(step_str, flush=True)

    def log_histogram(self, name: str, values: Tensor, step: int) -> None:
        """
        Logs a histogram (wandb and TensorBoard only).

        Args:
            name: Metric name.
            values: 1-D tensor of values.
            step: Global step.
        """
        values = values.detach().float().cpu()
        if self._wandb:
            self._wandb.log({name: self._wandb.Histogram(values.numpy())}, step=step)
        if self._tb:
            self._tb.add_histogram(name, values, step)

    def close(self) -> None:
        """Finishes all runs."""
        if self._wandb:
            self._wandb.finish()
        if self._mlflow:
            self._mlflow.end_run()
        if self._tb:
            self._tb.close()


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        elif callable(v):
            out[key] = getattr(v, "__name__", repr(v))
        else:
            out[key] = v
    return out


def _unwrap_batch(batch: Any, paired: bool = False) -> tuple[Tensor, Tensor | None]:
    """Returns ``(x, y)`` with ``y`` the transcoder target (``None`` unless ``paired``)."""
    if isinstance(batch, Tensor):
        x, y = batch, None
    elif isinstance(batch, (tuple, list)):
        x, y = batch[0], (batch[1] if len(batch) > 1 else None)
    elif isinstance(batch, dict):
        x, y = batch["activations"], batch.get("targets")
    else:
        raise TypeError(f"unsupported batch type {type(batch)}")
    if paired and y is None:
        raise ValueError("a transcoder needs (input, target) batches, e.g. ActivationStore(target_module=...)")
    return x, (y if paired else None)


def _linear_schedule(total_steps: int, warmup: int, decay_frac: float) -> Callable[[int], float]:
    decay_start = int(total_steps * (1 - decay_frac))

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if step >= decay_start:
            return max(0.0, (total_steps - step) / max(1, total_steps - decay_start))
        return 1.0

    return fn


class SAETrainer:
    """
    Trains a :class:`sae.model.SAE`.

    Args:
        model: The SAE.
        cfg: Trainer configuration; defaults to :class:`TrainerConfig()`.

    Attributes:
        model: The SAE being trained.
        cfg: The trainer configuration.
        device: Training device.
        lr: Resolved learning rate.
        step: Optimiser steps taken in :meth:`train`.
        tokens_seen: Tokens consumed in :meth:`train`.
    """

    def __init__(self, model: SAE, cfg: TrainerConfig | None = None):
        self.cfg = cfg or TrainerConfig()
        self.model = model
        torch.manual_seed(self.cfg.seed)

        dev = self.cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.model.to(self.device)

        ac = self.cfg.autocast_dtype
        if ac == "auto":
            ac = "bfloat16" if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else None
        self.autocast_dtype = getattr(torch, ac) if ac else None

        self.lr = self.cfg.lr or default_lr(self.cfg.optimizer, model.cfg.d_sae)
        self.optimizers = build_optimizers(model, self.cfg.optimizer, self.lr, self.cfg.adam_betas, self.cfg.adam_eps)
        self.forward = torch.compile(model, dynamic=True) if self.cfg.compile else model

        self.ema: dict[str, Tensor] | None = None
        if self.cfg.ema_decay is not None:
            self.ema = {n: p.detach().clone() for n, p in model.named_parameters()}

        self.step = 0
        self.tokens_seen = 0
        self.median_norm: float | None = None
        self.median_norm_target: float | None = None

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.autocast_dtype or torch.float32,
                              enabled=self.autocast_dtype is not None)

    def _filter(self, x: Tensor, y: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Drops tokens whose input (or target) norm exceeds ``outlier_norm_factor`` times the median."""
        if self.cfg.outlier_norm_factor is None or self.median_norm is None:
            return x, y
        keep = x.float().norm(dim=-1) <= self.cfg.outlier_norm_factor * self.median_norm
        if y is not None and self.median_norm_target is not None:
            keep &= y.float().norm(dim=-1) <= self.cfg.outlier_norm_factor * self.median_norm_target
        return x[keep], (None if y is None else y[keep])

    def _prepare(self, batch: Any) -> tuple[Tensor, Tensor | None]:
        x, y = _unwrap_batch(batch, self.model.cfg.transcoder)
        x, _ = pack([x.to(self.device, non_blocking=True)], "* d_in")
        if y is not None:
            y, _ = pack([y.to(self.device, non_blocking=True)], "* d_out")
        return self._filter(x, y)

    def _optimizer_step(self, loss: Tensor, optimizers: list[tuple[Optimizer, bool]],
                        schedulers: list[LambdaLR]) -> Tensor:
        for opt, _ in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        self.model.pre_optimizer_step()
        params = [p for p in self.model.parameters() if p.grad is not None]
        if self.cfg.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
        else:
            grad_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params]))
        for opt, _ in optimizers:
            opt.step()
        for sched in schedulers:
            sched.step()
        self.model.post_optimizer_step()
        self._update_ema()
        return grad_norm

    @torch.no_grad()
    def initialize(self, dataloader: Iterable) -> None:
        """Data-dependent initialization of the SAE.

        Estimates the median token norm for the outlier filter, then calls
        :meth:`sae.model.SAE.initialize_from_data` on the filtered sample.

        Args:
            dataloader: Iterable of activation batches.
        """
        needed = max(self.cfg.init_tokens, self.model.decoder.min_init_tokens)
        xs, ys, n = [], [], 0
        for batch in dataloader:
            x, y = _unwrap_batch(batch, self.model.cfg.transcoder)
            xs.append(pack([x.float()], "* d_in")[0])
            if y is not None:
                ys.append(pack([y.float()], "* d_out")[0])
            n += xs[-1].shape[0]
            if n >= needed:
                break
        x = torch.cat(xs)[:needed].to(self.device)
        y = torch.cat(ys)[:needed].to(self.device) if ys else None
        self.median_norm = x.norm(dim=-1).median().item()
        if y is not None:
            self.median_norm_target = y.norm(dim=-1).median().item()
        x, y = self._filter(x, y)
        self.model.initialize_from_data(x, y, normalize_input=self.cfg.normalize_input,
                                        b_dec_init=self.cfg.b_dec_init)
        if self.ema is not None:
            for name, p in self.model.named_parameters():
                self.ema[name].copy_(p)

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self.ema is None:
            return
        for name, p in self.model.named_parameters():
            self.ema[name].lerp_(p.detach(), 1 - self.cfg.ema_decay)

    def export_model(self) -> SAE:
        """
        Returns the model to evaluate or save.

        Returns:
            A copy carrying the EMA weights (re-projected onto the decoder
            constraints) if EMA is enabled, else the live model.
        """
        if self.ema is None:
            return self.model
        m = copy.deepcopy(self.model)
        with torch.no_grad():
            for name, p in m.named_parameters():
                p.copy_(self.ema[name])
        m.post_optimizer_step()
        return m

    def save(self, path: str | Path, **metadata: str) -> Path:
        """
        Saves :meth:`export_model` to a safetensors file.

        Args:
            path: Destination path.
            **metadata: Extra string metadata.

        Returns:
            The path written.
        """
        meta = {"step": str(self.step), "tokens_seen": str(self.tokens_seen), **metadata}
        return self.export_model().save(path, extra_metadata=meta)

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """
        Saves everything needed to resume :meth:`train`.

        Unlike :meth:`save`, the weights are stored unfolded together with
        optimizer states, EMA weights, step counters, the outlier-filter
        median and the RNG state. The file is written atomically.

        Args:
            path: Destination; defaults to ``output_dir/checkpoint.pt``.

        Returns:
            The path written.
        """
        path = Path(path) if path is not None else Path(self.cfg.output_dir) / "checkpoint.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format": CHECKPOINT_FORMAT,
            "sae_config": self.model.cfg.to_json(),
            "trainer_config": asdict(self.cfg),
            "model": self.model.state_dict(),
            "optimizers": [opt.state_dict() for opt, _ in self.optimizers],
            "ema": self.ema,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "median_norm": self.median_norm,
            "median_norm_target": self.median_norm_target,
            "rng": torch.get_rng_state(),
        }
        tmp = path.with_name(path.name + ".tmp")
        torch.save(state, tmp)
        tmp.replace(path)
        return path

    @classmethod
    def from_checkpoint(cls, path: str | Path, cfg: TrainerConfig | None = None) -> "SAETrainer":
        """
        Restores a trainer written by :meth:`save_checkpoint`.

        Call :meth:`train` with the same data and ``total_tokens`` /
        ``num_epochs`` to continue: the learning-rate schedule, k annealing
        and sparsity warmup resume at the saved step. The data stream starts
        again from its beginning, so with shuffled sources the remaining steps
        see a fresh pass rather than the exact continuation.

        Args:
            path: Checkpoint file.
            cfg: Trainer configuration; ``None`` reuses the saved one. Keep the
                optimizer and schedule settings unchanged.

        Returns:
            The restored trainer.

        Raises:
            ValueError: If the file is not a trainer checkpoint.
        """
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a trainer checkpoint")
        model = SAE(SAEConfig.from_json(state["sae_config"]))
        model.load_state_dict(state["model"])
        trainer = cls(model, cfg if cfg is not None else TrainerConfig(**state["trainer_config"]))
        for (opt, _), opt_state in zip(trainer.optimizers, state["optimizers"], strict=True):
            opt.load_state_dict(opt_state)
        if trainer.ema is not None and state["ema"] is not None:
            for name, value in state["ema"].items():
                trainer.ema[name].copy_(value)
        trainer.step = state["step"]
        trainer.tokens_seen = state["tokens_seen"]
        trainer.median_norm = state["median_norm"]
        trainer.median_norm_target = state["median_norm_target"]
        torch.set_rng_state(state["rng"])
        return trainer

    def _k_at(self, step: int) -> int:
        """k at ``step`` under linear k annealing."""
        cfg, mcfg = self.cfg, self.model.cfg
        if cfg.k_decay_steps <= 0 or step >= cfg.k_decay_steps:
            return mcfg.k
        k0 = cfg.k_initial or min(mcfg.d_sae, 10 * mcfg.d_in)
        progress = step / cfg.k_decay_steps
        return round(k0 * (1 - progress) + mcfg.k * progress)

    def _batches(self, dataloader: Iterable, total_steps: int) -> Iterator[Tensor]:
        if self.cfg.total_tokens is not None:
            source = itertools.chain.from_iterable(itertools.repeat(dataloader))
        else:
            source = itertools.chain.from_iterable(dataloader for _ in range(self.cfg.num_epochs))
        for batch in source:
            if self.step >= total_steps:
                return
            x, y = self._prepare(batch)
            if x.shape[0] > 1:
                yield x, y

    def _total_steps(self, dataloader: Iterable) -> int:
        if self.cfg.total_tokens is not None:
            batch_tokens = getattr(dataloader, "batch_tokens", None)
            if batch_tokens is None:
                first, _ = pack([_unwrap_batch(next(iter(dataloader)))[0]], "* d_in")
                batch_tokens = first.shape[0]
            return math.ceil(self.cfg.total_tokens / batch_tokens)
        if not hasattr(dataloader, "__len__"):
            raise ValueError(
                f"{type(dataloader).__name__} has no len(), so the number of steps per epoch is unknown; "
                "set TrainerConfig.total_tokens when streaming (e.g. from an ActivationStore)"
            )
        return len(dataloader) * self.cfg.num_epochs

    def train(self, dataloader: Iterable) -> SAE:
        """
        Trains the SAE on reconstruction.

        Args:
            dataloader: Iterable of activation batches ``(..., d_in)``: a
                :class:`~sae.data.ActivationStore`, or a ``DataLoader`` yielding
                tensors, tuples (first element used) or dicts with an
                ``"activations"`` key. Needs ``len()`` unless
                ``total_tokens`` is set.

        Returns:
            The trained model (EMA weights if enabled). A final checkpoint is
            written to ``output_dir/sae.safetensors``. A trainer restored with
            :meth:`from_checkpoint` continues from its saved step.
        """
        cfg, model = self.cfg, self.model
        if self.median_norm is None:
            self.initialize(dataloader)
        total_steps = self._total_steps(dataloader)
        warmup = max(1, min(cfg.warmup_steps, total_steps // 20)) # 5% cap
        # last_epoch resumes the schedule after from_checkpoint (-1 on a fresh run).
        schedulers = [
            LambdaLR(opt, _linear_schedule(total_steps, warmup if use_warmup else 0, cfg.decay_frac),
                     last_epoch=self.step - 1)
            for opt, use_warmup in self.optimizers
        ]

        run_name = cfg.run_name or time.strftime("sae-%Y%m%d-%H%M%S")
        full_config = {"sae": asdict(model.cfg), "trainer": asdict(cfg), "lr_resolved": self.lr,
                       "total_steps": total_steps}
        logger = MetricLogger(cfg.loggers, cfg.project, run_name, cfg.log_dir, full_config)

        window = _Window(model.cfg.d_sae, self.device)
        uses_k = model.activation.uses_k
        model.train()
        try:
            for flat, target in self._batches(dataloader, total_steps):
                if uses_k:
                    model.set_k(self._k_at(self.step))
                if cfg.sparsity_warmup_steps > 0:
                    model.sparsity_scale = min(1.0, (self.step + 1) / cfg.sparsity_warmup_steps)
                with self._autocast():
                    out = self.forward(flat, target=target, return_acts=False)
                grad_norm = self._optimizer_step(out.loss, self.optimizers, schedulers)

                self.step += 1
                self.tokens_seen += flat.shape[0]
                window.add(out, grad_norm, flat.shape[0])

                if self.step % cfg.log_every == 0 or self.step == total_steps:
                    metrics = window.summary(model, cfg.dense_threshold)
                    metrics["lr"] = schedulers[0].get_last_lr()[0]
                    if uses_k:
                        metrics["train/k"] = float(model.k_current)
                    metrics["tokens_seen"] = float(self.tokens_seen)
                    logger.log(metrics, self.step)
                    if self.step % cfg.hist_every == 0:
                        logger.log_histogram("latents/log10_density", window.log_density(), self.step)
                    window.reset()

                if cfg.save_every and self.step % cfg.save_every == 0:
                    self.save(Path(cfg.output_dir) / f"sae_step{self.step}.safetensors")
                if cfg.checkpoint_every and self.step % cfg.checkpoint_every == 0:
                    self.save_checkpoint()
        finally:
            if uses_k:
                model.set_k(model.cfg.k)
            model.sparsity_scale = 1.0
            logger.close()

        if cfg.checkpoint_every:
            self.save_checkpoint()
        self.save(Path(cfg.output_dir) / "sae.safetensors")
        return self.export_model()

    def finetune_kl(
        self,
        model: nn.Module,
        hook_module: nn.Module,
        input_loader: Iterable,
        kl_cfg: KLFinetuneConfig | None = None,
        target_module: nn.Module | None = None,
    ) -> SAE:
        """
        Fine-tunes the SAE spliced into ``model`` on KL + reconstruction.

        Each step runs the frozen model twice: a clean pass recording the
        activation at ``hook_module`` and the clean logits, and a patched pass
        in which the activation is replaced by the SAE reconstruction. The
        loss is ``0.5 * (L_sae + alpha * KL(clean || patched))`` with
        ``alpha = L_sae / KL`` (Karvonen, 2025; end-to-end SAEs: Braun et al.,
        2024). The learning rate decays linearly to 0 without warmup. Use it
        after :meth:`train`, for a few percent of the training budget.

        Args:
            model: The model the activations come from (frozen here).
            hook_module: The module whose output the SAE reconstructs.
            input_loader: Iterable of model inputs (images, token dicts, ...);
                cycled until ``kl_cfg.steps`` steps are done.
            kl_cfg: Fine-tuning configuration.
            target_module: For transcoders: the module whose output the
                transcoder replaces (``hook_module`` is its input).

        Returns:
            The fine-tuned model (EMA weights if enabled). A checkpoint is
            written to ``output_dir/sae_kl.safetensors``.
        """
        kc = kl_cfg or KLFinetuneConfig()
        sae = self.sae_for_finetune()
        lr = kc.lr or self.lr * kc.lr_scale
        optimizers = build_optimizers(sae, self.cfg.optimizer, lr, self.cfg.adam_betas, self.cfg.adam_eps)
        schedulers = [LambdaLR(opt, _linear_schedule(kc.steps, 0, 1.0)) for opt, _ in optimizers]

        run_name = (self.cfg.run_name or time.strftime("sae-%Y%m%d-%H%M%S")) + "-kl"
        logger = MetricLogger(self.cfg.loggers, self.cfg.project, run_name, self.cfg.log_dir,
                              {"sae": asdict(sae.cfg), "kl_finetune": asdict(kc), "lr_resolved": lr})

        model.eval()
        grad_flags = [p.requires_grad for p in model.parameters()]
        model.requires_grad_(False)
        sae.train()
        sums: dict[str, float] = {}
        step = 0
        try:
            for batch in itertools.chain.from_iterable(itertools.repeat(input_loader)):
                if step >= kc.steps:
                    break
                batch = _to_device(batch, self.device)
                args, kwargs = kc.input_fn(batch)
                captured: dict[str, Any] = {}

                def reconstruct(x: Tensor, y: Tensor) -> Tensor:
                    captured["out"] = sae(x, target=y if sae.cfg.transcoder else None, return_acts=False)
                    return captured["out"].recon

                with self._autocast():
                    with torch.no_grad():
                        clean = kc.output_fn(model(*args, **kwargs)).float()
                    with splice(hook_module, reconstruct, target_module, kc.layout, kc.exclude_positions,
                                kc.token_mask_fn, batch):
                        patched_logits = kc.output_fn(model(*args, **kwargs)).float()

                out = captured["out"]
                kl = self._kl(clean, patched_logits, kc.kl_mask_fn(batch) if kc.kl_mask_fn else None)
                alpha = out.loss.detach() / (kl.detach() + 1e-8)
                loss = 0.5 * (out.loss + alpha * kl)
                grad_norm = self._optimizer_step(loss, optimizers, schedulers)
                step += 1

                for k, v in {**out.metrics, "kl": kl.detach(), "alpha": alpha, "total": loss.detach(),
                             "grad_norm": grad_norm}.items():
                    sums[k] = sums.get(k, 0.0) + float(v)
                if step % kc.log_every == 0 or step == kc.steps:
                    n = kc.log_every if step % kc.log_every == 0 else step % kc.log_every
                    metrics = {f"kl_finetune/{k}": v / n for k, v in sums.items()}
                    metrics["kl_finetune/fve"] = 1.0 - metrics["kl_finetune/nmse"]
                    metrics["lr"] = schedulers[0].get_last_lr()[0]
                    logger.log(metrics, step)
                    sums = {}
        finally:
            for p, flag in zip(model.parameters(), grad_flags):
                p.requires_grad_(flag)
            logger.close()

        self.save(Path(self.cfg.output_dir) / "sae_kl.safetensors", phase="kl_finetune",
                  kl_steps=str(step))
        return self.export_model()

    def sae_for_finetune(self) -> SAE:
        """Merges EMA weights into SAE."""
        if self.ema is not None:
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    p.copy_(self.ema[name])
            self.model.post_optimizer_step()
        return self.model

    @staticmethod
    def _kl(clean: Tensor, patched: Tensor, mask: Tensor | None) -> Tensor:
        """Mean KL(clean || patched) over all logit rows (or the masked ones)."""
        clean = rearrange(clean, "... c -> (...) c")
        patched = rearrange(patched, "... c -> (...) c")
        if mask is not None:
            keep = rearrange(mask.to(torch.bool), "... -> (...)")
            clean, patched = clean[keep], patched[keep]
        return F.kl_div(F.log_softmax(patched, -1), F.log_softmax(clean, -1), log_target=True,
                        reduction="batchmean")

    @torch.no_grad()
    def evaluate(self, dataloader: Iterable, max_batches: int | None = None) -> dict[str, float]:
        """
        Inference-mode reconstruction metrics.

        Args:
            dataloader: Iterable of activation batches.
            max_batches: Optional cap on the number of batches.

        Returns:
            ``eval/nmse``, ``eval/fve`` and ``eval/l0``.
        """
        model = self.export_model()
        was_training = model.training
        model.eval()
        sse = sst = l0 = n = 0.0
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            flat, target = self._prepare(batch)
            with self._autocast():
                out = model(flat, target=target, return_acts=False)
            y = (flat if target is None else target).float()
            sse += ((out.recon.float() - y) ** 2).sum().item()
            sst += ((y - reduce(y, "n d -> 1 d", "mean")) ** 2).sum().item()
            l0 += out.latent_counts.sum().item()
            n += flat.shape[0]
        model.train(was_training)
        nmse = sse / max(sst, 1e-8)
        return {"eval/nmse": nmse, "eval/fve": 1 - nmse, "eval/l0": l0 / max(n, 1)}

    @torch.no_grad()
    def evaluate_downstream(
        self,
        model: nn.Module,
        hook_module: nn.Module,
        input_loader: Iterable,
        kl_cfg: KLFinetuneConfig | None = None,
        max_batches: int | None = None,
        loss_fn: Callable[[Any, Any], Tensor] | None = None,
        ablation: str = "zero",
        target_module: nn.Module | None = None,
    ) -> dict[str, float]:
        """
        Downstream fidelity of the model with the SAE spliced in.

        Reports the KL between the clean and the spliced model and the top-1
        agreement of their predictions. With ``loss_fn`` it also reports the
        model's loss clean, spliced, and with the hooked activation ablated,
        and the fraction of loss recovered
        ``(L_ablated - L_sae) / (L_ablated - L_clean)`` (1 = no degradation,
        0 = as bad as removing the layer's output; e.g. Bricken et al., 2023;
        Gao et al., 2024).

        Args:
            model: The model the activations come from.
            hook_module: The module whose output the SAE reconstructs.
            input_loader: Iterable of model inputs.
            kl_cfg: Supplies ``layout``, ``exclude_positions``,
                ``token_mask_fn``, ``input_fn``, ``output_fn`` and ``kl_mask_fn``.
            max_batches: Optional cap on the number of batches.
            loss_fn: Optional ``fn(model_output, batch) -> scalar`` mean loss,
                e.g. next-token cross-entropy for a language model or
                classification cross-entropy for a vision model.
            ablation: Baseline replacing the kept tokens for ``L_ablated``:
                ``"zero"``, or ``"mean"`` (mean of the batch's kept tokens).
            target_module: For transcoders: the module whose output the
                transcoder replaces (``hook_module`` is its input).

        Returns:
            ``downstream/kl`` and ``downstream/top1_agreement``; with
            ``loss_fn`` also ``downstream/loss_clean``, ``loss_sae``,
            ``loss_ablated``, ``delta_loss`` and ``loss_recovered``.

        Raises:
            ValueError: If ``ablation`` is unknown.
        """
        if ablation not in ("zero", "mean"):
            raise ValueError(f"ablation must be 'zero' or 'mean', got {ablation!r}")
        kc = kl_cfg or KLFinetuneConfig()
        sae = self.export_model()
        was_training = sae.training
        sae.eval()
        model.eval()
        sums = dict(kl=0.0, loss_clean=0.0, loss_sae=0.0, loss_ablated=0.0)
        agree = rows = 0.0
        n_batches = 0

        def ablate(x: Tensor, y: Tensor) -> Tensor:
            return y.mean(0, keepdim=True).expand_as(y) if ablation == "mean" else torch.zeros_like(y)

        for i, batch in enumerate(input_loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = _to_device(batch, self.device)
            args, kwargs = kc.input_fn(batch)

            def spliced(fn: Callable[[Tensor, Tensor], Tensor]):
                return splice(hook_module, fn, target_module, kc.layout, kc.exclude_positions, kc.token_mask_fn,
                              batch)

            with self._autocast():
                clean_out = model(*args, **kwargs)
                with spliced(lambda x, y: sae.reconstruct(x)):
                    patched_out = model(*args, **kwargs)
                if loss_fn is not None:
                    with spliced(ablate):
                        ablated_out = model(*args, **kwargs)
            clean, patched = kc.output_fn(clean_out).float(), kc.output_fn(patched_out).float()
            mask = kc.kl_mask_fn(batch) if kc.kl_mask_fn else None
            sums["kl"] += self._kl(clean, patched, mask).item()
            c = rearrange(clean, "... c -> (...) c").argmax(-1)
            p = rearrange(patched, "... c -> (...) c").argmax(-1)
            if mask is not None:
                keep = rearrange(mask.to(torch.bool), "... -> (...)")
                c, p = c[keep], p[keep]
            agree += (c == p).sum().item()
            rows += c.numel()
            if loss_fn is not None:
                sums["loss_clean"] += float(loss_fn(clean_out, batch))
                sums["loss_sae"] += float(loss_fn(patched_out, batch))
                sums["loss_ablated"] += float(loss_fn(ablated_out, batch))
            n_batches += 1
        sae.train(was_training)

        n = max(n_batches, 1)
        out = {"downstream/kl": sums["kl"] / n, "downstream/top1_agreement": agree / max(rows, 1)}
        if loss_fn is not None:
            clean_l, sae_l, abl_l = (sums[k] / n for k in ("loss_clean", "loss_sae", "loss_ablated"))
            out.update({
                "downstream/loss_clean": clean_l,
                "downstream/loss_sae": sae_l,
                "downstream/loss_ablated": abl_l,
                "downstream/delta_loss": sae_l - clean_l,
                "downstream/loss_recovered": (abl_l - sae_l) / (abl_l - clean_l) if abl_l != clean_l else float("nan"),
            })
        return out


class _Window:
    """Accumulates metrics between log steps."""

    def __init__(self, d_sae: int, device: torch.device):
        self.counts = torch.zeros(d_sae, dtype=torch.long, device=device)
        self.reset()

    def reset(self) -> None:
        self.sums: dict[str, float] = {}
        self.steps = 0
        self.tokens = 0
        self.counts.zero_()

    def add(self, out, grad_norm: Tensor, n_tokens: int) -> None:
        for k, v in out.metrics.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.sums["grad_norm"] = self.sums.get("grad_norm", 0.0) + float(grad_norm)
        self.counts += out.latent_counts
        self.steps += 1
        self.tokens += n_tokens

    def log_density(self) -> Tensor:
        return torch.log10(self.counts.float() / max(self.tokens, 1) + 1e-10)

    def summary(self, model: SAE, dense_threshold: float) -> dict[str, float]:
        s = max(self.steps, 1)
        m = {f"train/{k}": v / s for k, v in self.sums.items()}
        m["train/fve"] = 1.0 - m["train/nmse"]
        density = self.counts.float() / max(self.tokens, 1)
        m["train/dense_frac"] = (density > dense_threshold).float().mean().item()
        m["train/dead_frac"] = model.dead_mask.float().mean().item()
        m["train/never_fired_in_window_frac"] = (self.counts == 0).float().mean().item()
        return m
