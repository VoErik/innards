"""
Caching harvested activations to disk as safetensors shards.

Streaming from an :class:`innards.sae.data.ActivationStore` re-runs the model
every epoch. Caching is worth it when the same activations are used more than
once: several epochs, hyperparameter sweeps, or several SAEs on one layer.
Storage is ``n_tokens * d_in * 2`` bytes in bf16, e.g. about 0.7 GB for 1.9M
ViT-Tiny tokens (d = 192) or 1 GB per million Pythia-70m tokens (d = 512).

Layout of a cache directory::

    meta.json               # fields, n_tokens, shard list, dtype, user metadata
    shard_00000.safetensors # {"activations": (n, d_in)}, or one tensor per field
    shard_00001.safetensors
    ...

Batches may be tensors, ``(input, target)`` pairs (fields ``activations`` /
``targets``) or dicts of tensors (one field per key, e.g. several layers
from a multi-hook :class:`~innards.sae.data.ActivationStore`); the cache serves them
back in the same structure.

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from einops import pack
from safetensors.torch import (
    load_file,
    save_file
)
from torch import Tensor

from .data import split_fields

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
        source: Iterable of activation batches ``(..., d_in)``, of
            ``(input, target)`` pairs for transcoders, or of dicts of
            ``(..., d)`` tensors, e.g. an :class:`~innards.sae.data.ActivationStore`
            (whose batches are already shuffled across inputs).
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
    kind: str | None = None
    names: list[str] = []
    widths: list[int] = []

    def flush(final: bool) -> None:
        nonlocal pending, n_pending
        data = torch.cat(pending)
        while data.shape[0] >= shard_tokens or (final and data.shape[0] > 0):
            chunk, data = data[:shard_tokens], data[shard_tokens:]
            name = f"shard_{len(shards):05d}.safetensors"
            tensors = {f: t.contiguous() for f, t in zip(names, chunk.split(widths, dim=-1))}
            save_file(tensors, str(out_dir / name))
            shards.append({"file": name, "n_tokens": int(chunk.shape[0])})
        pending, n_pending = [data], data.shape[0]

    for batch in source:
        # fields are kept side by side (concatenated features) so rows stay aligned until written
        if isinstance(batch, dict):
            kind, fields = "dict", list(batch.items())
        elif isinstance(batch, (tuple, list)):
            kind, fields = "pair", [("activations", batch[0]), ("targets", batch[1])]
        else:
            kind, fields = "tensor", [("activations", batch)]
        flats = [pack([t.detach()], "* d")[0] for _, t in fields]
        names, widths = [f for f, _ in fields], [t.shape[-1] for t in flats]
        flat = torch.cat([t.to(flats[0].dtype) for t in flats], -1)
        if max_tokens is not None:
            flat = flat[: max_tokens - n_total]
        pending.append(flat.to("cpu", dtype))
        n_pending += flat.shape[0]
        n_total += flat.shape[0]
        if n_pending >= shard_tokens:
            flush(final=False)
        if max_tokens is not None and n_total >= max_tokens:
            break
    if kind is None:
        raise ValueError("source yielded no activations")
    flush(final=True)

    meta = {
        "kind": kind,
        "fields": [[f, w] for f, w in zip(names, widths)],
        "d_in": widths[0],
        "d_out": widths[1] if kind == "pair" else None,
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

    Caches written from ``(input, target)`` pairs yield such pairs, and caches
    written from dict batches yield dicts.

    Each epoch visits the shards in a random order and shuffles tokens within
    each shard. Leftovers carry over to the next shard, so every batch has
    exactly ``batch_tokens`` tokens (last partial batch is dropped).

    A background thread loads (and shuffles) the next shard while the current
    one is served, so at most two shards are in memory. For CUDA devices the
    shards are pinned, so the copies to the GPU are asynchronous.

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
        fields = self.meta.get("fields") or [["activations", self.meta["d_in"]]]
        self._kind = self.meta.get("kind", "tensor")
        self._names, self._widths = [f for f, _ in fields], [int(w) for _, w in fields]

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

    def _load_shard(self, name: str, g: torch.Generator) -> Tensor:
        tensors = load_file(str(self.cache_dir / name))
        # fields side by side, so rows stay aligned through shuffling and carry-over
        data = torch.cat([tensors[f] for f in self._names], -1)
        if self.shuffle:
            data = data[torch.randperm(data.shape[0], generator=g)]
        if self.device.type == "cuda":
            data = data.pin_memory()
        return data

    def _serve(self, batch: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        batch = batch.to(self.device, non_blocking=True)
        batch = batch.to(self.dtype) if self.dtype is not None else batch
        return split_fields(batch, self._kind, self._names, self._widths)

    def __iter__(self) -> Iterator[Tensor]:
        g = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        shards = self.meta["shards"]
        order = torch.randperm(len(shards), generator=g).tolist() if self.shuffle else list(range(len(shards)))
        bs = self.batch_tokens
        carry: Tensor | None = None
        # One worker, so shards are loaded and shuffled (drawing from ``g``) in order.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = [pool.submit(self._load_shard, shards[i]["file"], g) for i in order[:1]]
            for pos in range(len(order)):
                data = pending.pop().result()
                if pos + 1 < len(order):
                    pending.append(pool.submit(self._load_shard, shards[order[pos + 1]]["file"], g))
                start = 0
                if carry is not None and carry.shape[0] > 0:
                    need = bs - carry.shape[0]
                    if data.shape[0] < need:
                        carry = torch.cat([carry, data])
                        continue
                    yield self._serve(torch.cat([carry, data[:need]]))
                    start = need
                end = start + (data.shape[0] - start) // bs * bs
                for j in range(start, end, bs):
                    yield self._serve(data[j : j + bs])
                carry = data[end:]
