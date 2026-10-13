"""
Caching harvested activations to disk as safetensors shards.

Streaming from an :class:`sae.data.ActivationStore` re-runs the model
every epoch. Caching is worth it when the same activations are used more than
once: several epochs, hyperparameter sweeps, or several SAEs on one layer.
Storage is ``n_tokens * d_in * 2`` bytes in bf16, e.g. about 0.7 GB for 1.9M
ViT-Tiny tokens (d = 192) or 1 GB per million Pythia-70m tokens (d = 512).

Layout of a cache directory::

    meta.json               # d_in, n_tokens, shard list, dtype, user metadata
    shard_00000.safetensors # {"activations": (n, d_in)}
    shard_00001.safetensors
    ...

Typical usage::

    store = ActivationStore(model, model.blocks[6], image_loader, exclude_positions=(0,))
    cache_activations(store, "acts/vit_tiny_block6", metadata={"hook": "blocks.6"})
    batches = CachedActivations("acts/vit_tiny_block6", batch_tokens=4096)
    SAETrainer(sae, cfg).train(batches)
"""

from __future__ import annotations

import json
from collections.abc import (
    Iterable,
    Iterator
)
from pathlib import Path
from typing import Any

import torch
from einops import pack
from safetensors.torch import (
    load_file,
    save_file
)
from torch import Tensor

META_NAME = "meta.json"


def cache_activations(
    source: Iterable[Tensor],
    out_dir: str | Path,
    shard_tokens: int = 2**20,
    dtype: torch.dtype = torch.bfloat16,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Writes activation batches to safetensors shards.

    Args:
        source: Iterable of activation batches ``(..., d_in)``, e.g. an
            :class:`~sae.data.ActivationStore` (whose batches are already
            shuffled across inputs).
        out_dir: Output directory; created if needed. Existing shards with the
            same names are overwritten.
        shard_tokens: Tokens per shard file.
        dtype: Storage dtype. bf16 halves the size of fp32 and is what most
            SAE pipelines train on.
        max_tokens: Stop after this many tokens.
        metadata: JSON-serialisable extra information, e.g. model name, hook
            point, layout and excluded positions.

    Returns:
        The output directory.

    Raises:
        ValueError: If ``source`` yields no data.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pending: list[Tensor] = []
    n_pending = n_total = 0
    shards: list[dict[str, Any]] = []
    d_in = None

    def flush(final: bool) -> None:
        nonlocal pending, n_pending
        data = torch.cat(pending)
        while data.shape[0] >= shard_tokens or (final and data.shape[0] > 0):
            chunk, data = data[:shard_tokens], data[shard_tokens:]
            name = f"shard_{len(shards):05d}.safetensors"
            save_file({"activations": chunk.contiguous()}, str(out_dir / name))
            shards.append({"file": name, "n_tokens": int(chunk.shape[0])})
        pending, n_pending = [data], data.shape[0]

    for batch in source:
        flat, _ = pack([batch.detach()], "* d_in")
        if max_tokens is not None:
            flat = flat[: max_tokens - n_total]
        d_in = flat.shape[-1]
        pending.append(flat.to("cpu", dtype))
        n_pending += flat.shape[0]
        n_total += flat.shape[0]
        if n_pending >= shard_tokens:
            flush(final=False)
        if max_tokens is not None and n_total >= max_tokens:
            break
    if d_in is None:
        raise ValueError("source yielded no activations")
    flush(final=True)

    meta = {
        "d_in": d_in,
        "n_tokens": n_total,
        "dtype": str(dtype).removeprefix("torch."),
        "shards": shards,
        "metadata": metadata or {},
    }
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2))
    return out_dir


class CachedActivations:
    """
    Iterable of shuffled activation batches read from a cache directory.

    Each epoch visits the shards in a random order and shuffles tokens within
    each shard. Leftovers carry over to the next shard, so every batch has
    exactly ``batch_tokens`` tokens (last partial batch is dropped).
    Only one shard is in memory at a time.

    Args:
        cache_dir: Directory written by :func:`cache_activations`.
        batch_tokens: Tokens per batch.
        shuffle: Shuffle shard order and tokens within shards.
        seed: Seed of the first epoch; later epochs use ``seed + epoch``.
        device: Device the batches are moved to.
        dtype: Dtype of the served batches; ``None`` keeps the storage dtype.

    Attributes:
        batch_tokens: Tokens per batch (read by the trainer).
        meta: Contents of ``meta.json``.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        batch_tokens: int = 4096,
        shuffle: bool = True,
        seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.meta = json.loads((self.cache_dir / META_NAME).read_text())
        self.batch_tokens = batch_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        self._epoch = 0

    @property
    def d_in(self) -> int:
        """Width of the cached activations."""
        return int(self.meta["d_in"])

    @property
    def n_tokens(self) -> int:
        """Total number of cached tokens."""
        return int(self.meta["n_tokens"])

    def __len__(self) -> int:
        return self.n_tokens // self.batch_tokens

    def __iter__(self) -> Iterator[Tensor]:
        g = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        shards = self.meta["shards"]
        order = torch.randperm(len(shards), generator=g).tolist() if self.shuffle else range(len(shards))
        carry: Tensor | None = None
        for i in order:
            data = load_file(str(self.cache_dir / shards[i]["file"]))["activations"]
            if self.shuffle:
                data = data[torch.randperm(data.shape[0], generator=g)]
            if carry is not None:
                data = torch.cat([carry, data])
            n_full = data.shape[0] // self.batch_tokens * self.batch_tokens
            for j in range(0, n_full, self.batch_tokens):
                batch = data[j : j + self.batch_tokens].to(self.device, non_blocking=True)
                yield batch.to(self.dtype) if self.dtype is not None else batch
            carry = data[n_full:]
