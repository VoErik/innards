"""
Caching SAE latents to disk, and analyses that run on the cache.

:func:`cache_latents` runs the model and the SAE over a dataset once and
stores the sparse codes. :class:`LatentCache` then answers the usual
interpretability questions without touching the model again: per-latent
statistics, top-activating examples, examples sampled across activation
quantiles (which show what a latent does at typical, not only extreme,
activations; Bricken et al., 2023) and co-occurrence counts.

Layout of a cache directory::

    meta.json               # d_sae, n_tokens, n_examples, shards, user metadata
    shard_00000.safetensors # example (n,), position (n,) of each cached token;
    ...                     # token, latent, value (nnz,) of each non-zero

``token`` indexes the shard's tokens; a token's global id is the shard's
``token_offset`` plus that index. ``example`` is the input's index in loader
order and ``position`` its token position (counting excluded positions), as in
:class:`innards.sae.analysis.TopExamples`.

References:
    Bricken, T. et al. (2023). Towards Monosemanticity: Decomposing Language
        Models With Dictionary Learning. Transformer Circuits Thread.
"""

from __future__ import annotations

import json
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Sequence
)
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from safetensors.torch import (
    load_file,
    save_file
)
from torch import (
    Tensor,
    nn
)

from .analysis import (
    LatentStats,
    TopExamples
)
from .data import (
    _to_device,
    default_input_fn,
    run_capture,
    to_tokens,
    token_keep_mask
)
from .model import SAE

META_NAME = "meta.json"


@torch.no_grad()
def cache_latents(
    sae: SAE,
    model: nn.Module,
    hook_module: nn.Module | str,
    input_loader: Iterable,
    out_dir: str | Path,
    layout: str = "btd",
    exclude_positions: Sequence[int] = (),
    token_mask_fn: Callable[[Any], Tensor] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
    shard_tokens: int = 2**20,
    chunk_tokens: int = 4096,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Encodes a dataset with the SAE and writes the sparse codes to disk.

    The loader must be deterministic (``shuffle=False``) so that example
    indices can be mapped back to inputs.

    Args:
        sae: The SAE.
        model: The model the SAE was trained on.
        hook_module: Module (or layer name) whose output the SAE reads.
        input_loader: Iterable of model inputs.
        out_dir: Output directory; created if needed.
        layout: Activation layout at the hook.
        exclude_positions: Positions not cached, as in training.
        token_mask_fn: Optional ``fn(batch) -> bool (batch, tokens)`` mask of
            tokens to cache, e.g. non-padding tokens.
        input_fn: Maps a batch to model ``(args, kwargs)``.
        device: Device to run on; defaults to the SAE's device.
        max_batches: Optional cap on the number of batches.
        shard_tokens: Tokens per shard (shards end at batch boundaries, so
            they may be slightly larger).
        chunk_tokens: Tokens encoded at once, bounding memory to
            ``chunk_tokens * d_sae``.
        metadata: JSON-serialisable extra information, e.g. model name and
            hook point.

    Returns:
        The output directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device) if device is not None else sae.device
    sae.eval()
    model.eval()
    shards: list[dict[str, Any]] = []
    pending: dict[str, list[Tensor]] = {k: [] for k in ("example", "position", "token", "latent", "value")}
    n_pending = n_total = n_examples = 0

    def flush() -> None:
        nonlocal n_pending, pending
        if n_pending == 0:
            return
        data = {k: torch.cat(v).contiguous() for k, v in pending.items()}
        name = f"shard_{len(shards):05d}.safetensors"
        save_file(data, str(out_dir / name))
        shards.append({"file": name, "n_tokens": n_pending, "nnz": int(data["value"].numel()),
                       "token_offset": n_total - n_pending})
        pending = {k: [] for k in pending}
        n_pending = 0

    for i, batch in enumerate(input_loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = _to_device(batch, device)
        tokens = to_tokens(run_capture(model, hook_module, batch, input_fn), layout)
        b, t, _ = tokens.shape
        keep = token_keep_mask(tokens, exclude_positions, token_mask_fn, batch)
        flat_idx = rearrange(keep, "b t -> (b t)").nonzero().squeeze(-1)
        kept = tokens[keep].to(sae.w_enc.dtype)
        pending["example"].append((n_examples + flat_idx // t).cpu())
        pending["position"].append((flat_idx % t).to(torch.int32).cpu())
        for start in range(0, kept.shape[0], chunk_tokens):
            acts = sae.encode(kept[start : start + chunk_tokens]).float()
            rows, cols = acts.nonzero(as_tuple=True)
            pending["token"].append((n_pending + start + rows).to(torch.int32).cpu())
            pending["latent"].append(cols.to(torch.int32).cpu())
            pending["value"].append(acts[rows, cols].cpu())
        n_pending += kept.shape[0]
        n_total += kept.shape[0]
        n_examples += b
        if n_pending >= shard_tokens:
            flush()
    flush()

    meta = {
        "d_sae": sae.cfg.d_sae,
        "n_tokens": n_total,
        "n_examples": n_examples,
        "shards": shards,
        "sae_config": json.loads(sae.cfg.to_json()),
        "metadata": metadata or {},
    }
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2))
    return out_dir


def _top_per_latent(latent: Tensor, value: Tensor, n_top: int, *payload: Tensor) -> tuple[Tensor, ...]:
    """Keeps the ``n_top`` entries of largest ``|value|`` per latent (COO lists)."""
    order = value.abs().argsort(descending=True, stable=True)
    order = order[latent[order].argsort(stable=True)]  # grouped by latent, best first
    latent = latent[order]
    counts = torch.bincount(latent)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(latent.numel()) - starts[latent]
    sel = order[rank < n_top]
    return (latent[rank < n_top], value[sel], *(p[sel] for p in payload))


class LatentCache:
    """
    Read access to a directory written by :func:`cache_latents`.

    Every method streams over the shards, so memory is bounded by one shard
    (plus the per-token ``example`` / ``position`` index once it is needed).

    Args:
        cache_dir: The cache directory.

    Attributes:
        meta: Contents of ``meta.json``.
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.meta = json.loads((self.cache_dir / META_NAME).read_text())
        self._index: tuple[Tensor, Tensor] | None = None

    @property
    def d_sae(self) -> int:
        """Number of latents."""
        return int(self.meta["d_sae"])

    @property
    def n_tokens(self) -> int:
        """Number of cached tokens."""
        return int(self.meta["n_tokens"])

    def shards(self) -> Iterator[dict[str, Tensor]]:
        """
        Yields each shard's tensors, with ``token`` converted to global ids.

        Yields:
            Dicts with ``example``, ``position``, ``token`` (int64, global),
            ``latent`` (int64) and ``value``.
        """
        for shard in self.meta["shards"]:
            data = load_file(str(self.cache_dir / shard["file"]))
            data["token"] = data["token"].long() + shard["token_offset"]
            data["latent"] = data["latent"].long()
            yield data

    def token_index(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """
        Maps global token ids to ``(example, position)``.

        Args:
            tokens: Global token ids.

        Returns:
            ``(example, position)`` tensors shaped like ``tokens``.
        """
        if self._index is None:
            parts = [(d["example"], d["position"].long()) for d in self.shards()]
            self._index = (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]))
        return self._index[0][tokens], self._index[1][tokens]

    def stats(self) -> LatentStats:
        """
        Per-latent statistics, as in :func:`innards.sae.analysis.analyze_dataset`.

        Returns:
            A :class:`~innards.sae.analysis.LatentStats`.
        """
        count = torch.zeros(self.d_sae)
        total = torch.zeros(self.d_sae)
        peak = torch.zeros(self.d_sae)
        for d in self.shards():
            mags = d["value"].abs()
            count += torch.bincount(d["latent"], minlength=self.d_sae)
            total += torch.bincount(d["latent"], weights=mags, minlength=self.d_sae)
            peak = peak.scatter_reduce(0, d["latent"], mags, reduce="amax")
        return LatentStats(
            frequency=count / max(self.n_tokens, 1),
            mean_activation=total / count.clamp_min(1),
            max_activation=peak,
            n_tokens=self.n_tokens,
        )

    def top_examples(self, n_top: int = 16) -> TopExamples:
        """
        Top-activating tokens per latent by absolute activation.

        Args:
            n_top: Tokens kept per latent.

        Returns:
            A :class:`~innards.sae.analysis.TopExamples`; empty slots have value 0
            and example / position -1.
        """
        lat = torch.zeros(0, dtype=torch.long)
        val = torch.zeros(0)
        tok = torch.zeros(0, dtype=torch.long)
        for d in self.shards():
            lat, val, tok = _top_per_latent(torch.cat([lat, d["latent"]]), torch.cat([val, d["value"]]), n_top,
                                            torch.cat([tok, d["token"]]))
        counts = torch.bincount(lat, minlength=self.d_sae)
        slot = torch.arange(lat.numel()) - (torch.cumsum(counts, 0) - counts)[lat]
        values = torch.zeros(self.d_sae, n_top)
        example = torch.full((self.d_sae, n_top), -1, dtype=torch.long)
        position = torch.full((self.d_sae, n_top), -1, dtype=torch.long)
        values[lat, slot] = val
        example[lat, slot], position[lat, slot] = self.token_index(tok)
        return TopExamples(values=values, example=example, position=position)

    def latent_activations(self, latent: int) -> tuple[Tensor, Tensor]:
        """
        Every non-zero activation of one latent.

        Args:
            latent: Latent index.

        Returns:
            ``(tokens, values)``: global token ids and activations, in token order.
        """
        toks, vals = [], []
        for d in self.shards():
            hit = d["latent"] == latent
            toks.append(d["token"][hit])
            vals.append(d["value"][hit])
        return torch.cat(toks), torch.cat(vals)

    def quantile_examples(
        self,
        latent: int,
        n_bins: int = 5,
        per_bin: int = 5,
        seed: int = 0,
    ) -> list[list[tuple[float, int, int]]]:
        """
        Examples sampled uniformly within activation-magnitude quantile bins.

        Top examples only show a latent's most extreme behaviour; sampling
        each quantile bin shows what it responds to at typical activations,
        which is where polysemanticity tends to show up (Bricken et al., 2023).

        Args:
            latent: Latent index.
            n_bins: Number of equal-count bins over the latent's non-zero
                activations, ordered from weakest to strongest.
            per_bin: Examples sampled per bin (fewer if the bin is smaller).
            seed: Sampling seed.

        Returns:
            One list per bin of ``(value, example, position)`` triples,
            sorted by descending magnitude within the bin.
        """
        tokens, values = self.latent_activations(latent)
        if values.numel() == 0:
            return [[] for _ in range(n_bins)]
        order = values.abs().argsort()
        g = torch.Generator().manual_seed(seed)
        bins = []
        for part in torch.tensor_split(order, n_bins):
            pick = part[torch.randperm(part.numel(), generator=g)[:per_bin]]
            pick = pick[values[pick].abs().argsort(descending=True)]
            example, position = self.token_index(tokens[pick])
            bins.append(list(zip(values[pick].tolist(), example.tolist(), position.tolist())))
        return bins

    def cooccurrence(self, latents: Sequence[int] | Tensor, chunk_tokens: int | None = None) -> Tensor:
        """
        Number of tokens on which each pair of latents is active together.

        Args:
            latents: Latent indices (a few thousand at most: the result is
                ``(n, n)``).
            chunk_tokens: Tokens densified at once to ``(chunk_tokens, n)``;
                ``None`` picks about 64M entries.

        Returns:
            Symmetric count matrix of shape ``(n, n)``; the diagonal holds each
            latent's firing count. Divide by the diagonal for conditional
            probabilities ``P(j | i)``, or compute Jaccard similarities as
            ``C / (diag_i + diag_j - C)``.
        """
        latents = torch.as_tensor(latents, dtype=torch.long)
        lookup = torch.full((self.d_sae,), -1, dtype=torch.long)
        lookup[latents] = torch.arange(latents.numel())
        n = latents.numel()
        chunk = chunk_tokens or max(1, 2**26 // n)
        counts = torch.zeros(n, n)
        for shard, d in zip(self.meta["shards"], self.shards()):
            col = lookup[d["latent"]]
            hit = col >= 0
            tok, col = d["token"][hit] - shard["token_offset"], col[hit]  # tokens are stored in order
            bounds = torch.searchsorted(tok, torch.arange(0, shard["n_tokens"] + chunk, chunk))
            for c, (lo, hi) in enumerate(zip(bounds[:-1].tolist(), bounds[1:].tolist())):
                if lo == hi:
                    continue
                active = torch.zeros(chunk, n)
                active[tok[lo:hi] - c * chunk, col[lo:hi]] = 1.0
                counts += active.T @ active
        return counts
