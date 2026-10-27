"""
Training one SAE per hook point from a single pass over the model.

Harvesting is usually the expensive part of SAE training, so it pays to
capture many layers at once (as EleutherAI's ``sparsify`` does): a
multi-layer :class:`~innards.extract.RowStream` yields dict batches, and an
:class:`SAEGroup` holding one SAE per key trains them all in one loop with
:class:`~innards.sae.trainer.SAETrainer`. Members never share parameters, so they
train independently; only gradient clipping and the learning-rate schedule
are shared.

Typical usage::

    hooks = {f"layer{i}": model.blocks[i] for i in range(12)}
    stream = RowStream(model, loader, hooks, exclude=(0,))
    group = SAEGroup({name: SAE(d_in=768, d_sae=16384, k=32) for name in hooks})
    SAETrainer(group, TrainerConfig(total_tokens=100_000_000)).train(stream)
    group["layer3"].save("layer3.safetensors")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from einops import pack
from torch import (
    Tensor,
    nn
)

from .model import (
    SAE,
    SAEConfig,
    SAEOutput,
    register_model_type
)


@register_model_type
class SAEGroup(nn.Module):
    """
    Several SAEs trained together on dict batches, one per key.

    Implements the trainer protocol of :class:`~innards.sae.model.SAE` (batch
    unpacking, initialization, k annealing, saving), so it trains with
    :class:`~innards.sae.trainer.SAETrainer` like a single SAE. The loss is the sum of
    the members' losses; metrics are prefixed with the member names.

    Args:
        saes: Mapping from batch key (hook name) to SAE. Transcoders are not
            supported.

    Raises:
        ValueError: If the group is empty or contains a transcoder.
    """

    def __init__(self, saes: dict[str, SAE]):
        super().__init__()
        if not saes:
            raise ValueError("SAEGroup needs at least one SAE")
        if any(s.cfg.transcoder for s in saes.values()):
            raise ValueError("SAEGroup holds SAEs; train transcoders individually")
        self.saes = nn.ModuleDict(saes)

    @property
    def names(self) -> list[str]:
        """Member names, in batch order."""
        return list(self.saes.keys())

    def __getitem__(self, name: str) -> SAE:
        return self.saes[name]

    def batch_tensors(self, batch: dict[str, Tensor]) -> tuple[Tensor, ...]:
        """
        One flattened tensor per member, in :attr:`names` order.

        Args:
            batch: Dict with one ``(..., d_in)`` tensor per member name.

        Returns:
            The tensors, each ``(n, d_in)`` with matching rows (tokens).
        """
        return tuple(pack([batch[name]], "* d")[0] for name in self.names)

    def forward(self, *tensors: Tensor, return_acts: bool = False) -> SAEOutput:
        """
        Runs every member on its tensor.

        Args:
            *tensors: One activation tensor per member, in :attr:`names` order.
            return_acts: Passed to the members.

        Returns:
            An :class:`~innards.sae.model.SAEOutput` with the summed loss, prefixed
            metrics, concatenated latent counts and the members' outputs in
            ``parts``.
        """
        parts = {name: sae(x, return_acts=return_acts) for (name, sae), x in zip(self.saes.items(), tensors)}
        metrics = {f"{name}/{k}": v for name, out in parts.items() for k, v in out.metrics.items()}
        return SAEOutput(
            acts=None,
            recon=None,
            loss=torch.stack([out.loss for out in parts.values()]).sum(),
            metrics=metrics,
            latent_counts=torch.cat([out.latent_counts for out in parts.values()]),
            parts=parts,
        )

    def initialize_from_data(self, *tensors: Tensor, **kwargs: Any) -> None:
        """Initializes every member from its sample (see :meth:`SAE.initialize_from_data`)."""
        for sae, x in zip(self.saes.values(), tensors):
            sae.initialize_from_data(x, **kwargs)

    @torch.no_grad()
    def eval_sums(self, out: SAEOutput, *tensors: Tensor) -> dict[str, tuple[float, float, float]]:
        """Evaluation sums per member (see :meth:`SAE.eval_sums`)."""
        return {name: sae.eval_sums(out.parts[name], x)[""]
                for (name, sae), x in zip(self.saes.items(), tensors)}

    def pre_optimizer_step(self) -> None:
        for sae in self.saes.values():
            sae.pre_optimizer_step()

    def post_optimizer_step(self) -> None:
        for sae in self.saes.values():
            sae.post_optimizer_step()

    def anneal_k(self, progress: float, k_initial: int | None = None) -> None:
        for sae in self.saes.values():
            sae.anneal_k(progress, k_initial)

    @property
    def uses_k(self) -> bool:
        return any(s.uses_k for s in self.saes.values())

    @property
    def k_current(self) -> int | None:
        """k of the first TopK-style member (members anneal in lockstep)."""
        return next((s.k_current for s in self.saes.values() if s.uses_k), None)

    @property
    def sparsity_scale(self) -> float:
        return next(iter(self.saes.values())).sparsity_scale

    @sparsity_scale.setter
    def sparsity_scale(self, value: float) -> None:
        for sae in self.saes.values():
            sae.sparsity_scale = value

    @property
    def min_init_tokens(self) -> int:
        return max(s.min_init_tokens for s in self.saes.values())

    @property
    def lr_d_sae(self) -> int:
        """Largest member size: the learning rate default is set for the largest dictionary."""
        return max(s.cfg.d_sae for s in self.saes.values())

    @property
    def dead_mask(self) -> Tensor:
        return torch.cat([s.dead_mask for s in self.saes.values()])

    @property
    def tokens_since_active(self) -> Tensor:
        return torch.cat([s.tokens_since_active for s in self.saes.values()])

    @property
    def device(self) -> torch.device:
        return next(iter(self.saes.values())).device

    def config_json(self) -> str:
        """Member configurations as JSON (see :meth:`from_config_json`)."""
        return json.dumps({name: json.loads(sae.cfg.to_json()) for name, sae in self.saes.items()})

    @classmethod
    def from_config_json(cls, s: str) -> "SAEGroup":
        """Builds an untrained group from :meth:`config_json` output."""
        return cls({name: SAE(SAEConfig.from_json(json.dumps(c))) for name, c in json.loads(s).items()})

    def save(self, path: str | Path, extra_metadata: dict[str, str] | None = None) -> Path:
        """
        Saves every member as ``<dir>/<name>.safetensors``.

        Args:
            path: Output directory; a ``.safetensors`` suffix is dropped, so
                the trainer's ``sae.safetensors`` becomes the directory ``sae``.
            extra_metadata: Metadata added to every file (plus ``"name"``).

        Returns:
            The directory.
        """
        path = Path(path)
        if path.suffix == ".safetensors":
            path = path.with_suffix("")
        for name, sae in self.saes.items():
            sae.save(path / f"{name}.safetensors", {**(extra_metadata or {}), "name": name})
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "SAEGroup":
        """
        Loads a group saved with :meth:`save`.

        Args:
            path: The directory (a ``.safetensors`` suffix is dropped).
            device: Device to load onto.

        Returns:
            The group in eval mode, members in file-name order.
        """
        path = Path(path)
        if path.suffix == ".safetensors":
            path = path.with_suffix("")
        files = sorted(path.glob("*.safetensors"))
        return cls({f.stem: SAE.load(f, device) for f in files}).eval()
