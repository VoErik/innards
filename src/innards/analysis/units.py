"""
Units over a dataset: what each unit responds to, in one pass or from a disk cache.

A *unit* is one output of an encode function applied to a layer's tokens:
the neurons or channels themselves (the default), projections on directions
(``lambda t: t @ V.T``), probe logits, or the latents of an SAE
(``innards.sae.latent_fn(sae)``). Everything here works the same for all of
them, for ViT tokens, CNN feature maps and language-model positions alike.

* :func:`unit_stats` makes one pass over a dataset and returns per-unit
  statistics and the top-activating tokens (or examples).
* :func:`cache_units` stores the (sparse) activations instead, so later
  questions need no model: :func:`cache_stats`, :func:`cache_top`,
  :func:`unit_activations`, :func:`quantile_examples` (examples across
  activation strengths, which show what a unit does at typical, not only
  extreme, activations; Bricken et al., 2023) and :func:`cooccurrence`.
* :func:`entries` turns the top-k tensors of one unit into a list.

A unit is *active* on a token when ``|activation| > threshold``. SAE latents
are sparse, so ``threshold=0`` is the natural choice; dense units (neurons,
directions) need a threshold for frequencies, and for :func:`cache_units`
to store a sparse cache.

Examples are indexed in loader order (use ``shuffle=False``) and positions
count every token of the layer, including excluded ones, so both map back to
inputs.

Typical usage::

    stats, top = unit_stats(model, loader, "blocks.6", latent_fn(sae), exclude=(0,))
    entries(top, 42)                                      # [(value, example, position), ...]
    path = cache_units(model, loader, "blocks.6", "units/b6", latent_fn(sae), exclude=(0,))
    cache_top(path, n_top=20, per_example=True), quantile_examples(path, 42)

Layout of a cache directory::

    meta.json               # n_units, n_tokens, n_examples, threshold, shards, user metadata
    shard_00000.safetensors # example (n,), position (n,) of each cached token;
    ...                     # token, unit, value (nnz,) of each active entry

References:
    Bricken, T. et al. (2023). Towards Monosemanticity: Decomposing Language
        Models With Dictionary Learning. Transformer Circuits Thread.
"""

from __future__ import annotations

import functools
import json
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Sequence
)
from pathlib import Path
from typing import (
    Any,
    NamedTuple
)

import torch
from safetensors.torch import (
    load_file,
    save_file
)
from torch import Tensor

from ..extract import (
    default_input_fn,
    layer_tokens,
    to_device
)
from ..layouts import keep_mask

Encode = Callable[[Tensor], Tensor]
"""``fn(tokens (n, d)) -> (n, units)``."""

META_NAME = "meta.json"


class UnitStats(NamedTuple):
    """
    Per-unit statistics over the kept tokens of a dataset.

    Attributes:
        frequency: Fraction of tokens on which each unit is active, ``(units,)``.
            Dead units: ``(frequency == 0).nonzero()``.
        mean: Mean ``|activation|`` where active, ``(units,)``.
        max: Largest ``|activation|``, ``(units,)``.
        n_tokens: Number of kept tokens.
    """

    frequency: Tensor
    mean: Tensor
    max: Tensor
    n_tokens: int


class TopTokens(NamedTuple):
    """
    Top activations per unit, by magnitude, best first.

    Attributes:
        values: Signed activations ``(units, n_top)``; 0 in empty slots.
        example: Example index in loader order ``(units, n_top)``; -1 in empty slots.
        position: Token position within the example ``(units, n_top)``; -1 in
            empty slots.
    """

    values: Tensor
    example: Tensor
    position: Tensor


def entries(top: TopTokens, unit: int) -> list[tuple[float, int, int]]:
    """
    The non-empty ``(value, example, position)`` entries of one unit, best first.

    Args:
        top: Top activations, from :func:`unit_stats` or :func:`cache_top`.
        unit: Unit index.

    Returns:
        A list of triples.
    """
    return [(v, e, p) for v, e, p in zip(top.values[unit].tolist(), top.example[unit].tolist(),
                                         top.position[unit].tolist()) if e >= 0]


def _identity(tokens: Tensor) -> Tensor:
    return tokens


def _encode_kept(model, batch, layer, encode, exclude, mask_fn, input_fn) -> tuple[Tensor, Tensor, int, int]:
    """Kept tokens ``(n, d)`` of a batch, their flat ``(b t)`` indices, and ``b, t``."""
    tokens = layer_tokens(model, batch, layer, input_fn)
    b, t, _ = tokens.shape
    keep = keep_mask(tokens, exclude, mask_fn(batch) if mask_fn else None)
    return tokens[keep], keep.reshape(-1).nonzero().squeeze(-1), b, t


def _merge(top: tuple[Tensor, Tensor, Tensor] | None, v: Tensor, e: Tensor, p: Tensor,
           n_top: int) -> tuple[Tensor, Tensor, Tensor]:
    """Keeps the ``n_top`` candidates of largest magnitude per unit; all ``(n, units)``."""
    if top is not None:
        v, e, p = torch.cat([top[0], v]), torch.cat([top[1], e]), torch.cat([top[2], p])
    sel = v.abs().topk(min(n_top, v.shape[0]), dim=0).indices
    return v.gather(0, sel), e.gather(0, sel), p.gather(0, sel)


def _finish_top(top: tuple[Tensor, Tensor, Tensor] | None, n_units: int, n_top: int) -> TopTokens:
    values = torch.zeros(n_units, n_top)
    example = torch.full((n_units, n_top), -1, dtype=torch.long)
    position = torch.full((n_units, n_top), -1, dtype=torch.long)
    if top is not None:
        v, e, p = (x.T.cpu() for x in top)
        k = v.shape[1]
        empty = v == 0
        values[:, :k] = v.float()
        example[:, :k] = e.masked_fill(empty, -1)
        position[:, :k] = p.masked_fill(empty, -1)
    return TopTokens(values, example, position)


@torch.no_grad()
def unit_stats(
    model: Any,
    batches: Iterable,
    layer: str | Any,
    encode: Encode | None = None,
    n_top: int = 16,
    per_example: bool = False,
    threshold: float = 0.0,
    exclude: Sequence[int] = (),
    mask_fn: Callable[[Any], Tensor] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
    chunk_tokens: int = 4096,
) -> tuple[UnitStats, TopTokens]:
    """
    One pass over a dataset: per-unit statistics and top activations.

    Only kept tokens are encoded, ``chunk_tokens`` at a time, so memory is
    bounded by ``chunk_tokens * units`` whatever the batch size.

    Args:
        model: The model (plain or nnsight-wrapped).
        batches: Iterable of model inputs, e.g. a ``DataLoader`` with
            ``shuffle=False``.
        layer: Layer name or module.
        encode: ``fn(tokens (n, d)) -> (n, units)``; ``None`` takes the
            layer's dimensions (neurons, channels) as units.
        n_top: Entries kept per unit.
        per_example: Rank examples by their peak token (one entry per
            example, e.g. top images) instead of ranking tokens (where the
            top tokens can all come from one example).
        threshold: A unit is active where ``|activation| > threshold``.
        exclude: Token positions left out, e.g. ``(0,)`` for CLS / BOS.
        mask_fn: Optional ``fn(batch) -> bool (batch, tokens)`` of tokens to
            keep, e.g. from the attention mask.
        input_fn: Maps a batch to model ``(args, kwargs)``.
        device: Device for the inputs; defaults to the model's.
        max_batches: Optional cap on the number of batches.
        chunk_tokens: Tokens encoded at once.

    Returns:
        ``(stats, top)``: a :class:`UnitStats` and a :class:`TopTokens`.
    """
    encode = encode or _identity
    if device is None:
        device = next(iter(model.parameters())).device
    count = total = peak = top = None
    n_tokens = offset = 0
    model.eval()
    for i, batch in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        kept, flat, b, t = _encode_kept(model, to_device(batch, device), layer, encode, exclude, mask_fn, input_fn)
        n_tokens += kept.shape[0]
        batch_best = None
        for start in range(0, kept.shape[0], chunk_tokens):
            acts = encode(kept[start : start + chunk_tokens]).float()
            mags = acts.abs()
            if count is None:
                count, total, peak = (torch.zeros(acts.shape[1], device=acts.device) for _ in range(3))
            active = mags > threshold
            count += active.sum(0)
            total += (mags * active).sum(0)
            peak = torch.maximum(peak, mags.max(0).values)
            where = flat[start : start + acts.shape[0]].to(acts.device)
            if per_example:  # each example's peak token in this chunk, merged over chunks
                ex = (where // t)[:, None].expand_as(mags)
                best = torch.full((b, acts.shape[1]), -1.0, device=acts.device).scatter_reduce(0, ex, mags, "amax")
                rows, units = (mags == best.gather(0, ex)).nonzero(as_tuple=True)
                val = torch.zeros_like(best)
                pos = torch.zeros_like(best, dtype=torch.long)
                val[ex[rows, units], units] = acts[rows, units]
                pos[ex[rows, units], units] = (where % t)[rows]
                if batch_best is None:
                    batch_best = (best, val, pos)
                else:
                    better = best > batch_best[0]
                    batch_best = tuple(torch.where(better, new, old) for new, old in zip((best, val, pos), batch_best))
            else:
                idx = mags.topk(min(n_top, mags.shape[0]), dim=0).indices
                top = _merge(top, acts.gather(0, idx), offset + where[idx] // t, where[idx] % t, n_top)
        if per_example and batch_best is not None:
            _, val, pos = batch_best
            ex = torch.arange(offset, offset + b, device=val.device)[:, None].expand_as(val)
            top = _merge(top, val, ex, pos, n_top)
        offset += b

    if count is None:
        raise ValueError("no tokens were kept")
    stats = UnitStats(
        frequency=(count / max(n_tokens, 1)).cpu(),
        mean=(total / count.clamp_min(1)).cpu(),
        max=peak.cpu(),
        n_tokens=n_tokens,
    )
    return stats, _finish_top(top, len(count), n_top)


# --------------------------------------------------------------------------- disk cache


@torch.no_grad()
def cache_units(
    model: Any,
    batches: Iterable,
    layer: str | Any,
    out_dir: str | Path,
    encode: Encode | None = None,
    threshold: float = 0.0,
    exclude: Sequence[int] = (),
    mask_fn: Callable[[Any], Tensor] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
    shard_tokens: int = 2**20,
    chunk_tokens: int = 4096,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Encodes a dataset and writes the active unit activations to disk (sparse).

    Args:
        model: The model (plain or nnsight-wrapped).
        batches: Iterable of model inputs (``shuffle=False``).
        layer: Layer name or module.
        out_dir: Output directory; created if needed.
        encode: ``fn(tokens (n, d)) -> (n, units)``; ``None``: the layer's dimensions.
        threshold: Only ``|activation| > threshold`` is stored; for dense
            units pick it so the cache stays sparse.
        exclude: Token positions left out.
        mask_fn: Optional ``fn(batch) -> bool (batch, tokens)`` of tokens to keep.
        input_fn: Maps a batch to model ``(args, kwargs)``.
        device: Device for the inputs; defaults to the model's.
        max_batches: Optional cap on the number of batches.
        shard_tokens: Tokens per shard (shards end at batch boundaries, so
            an example never spans two shards).
        chunk_tokens: Tokens encoded at once.
        metadata: JSON-serialisable extra information (model, layer, SAE config).

    Returns:
        The output directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encode = encode or _identity
    if device is None:
        device = next(iter(model.parameters())).device
    model.eval()
    keys = ("example", "position", "token", "unit", "value")
    shards: list[dict[str, Any]] = []
    pending: dict[str, list[Tensor]] = {k: [] for k in keys}
    n_pending = n_total = n_examples = 0
    n_units = None

    def flush() -> None:
        nonlocal n_pending, pending
        if n_pending == 0:
            return
        data = {k: torch.cat(v).contiguous() for k, v in pending.items()}
        name = f"shard_{len(shards):05d}.safetensors"
        save_file(data, str(out_dir / name))
        shards.append({"file": name, "n_tokens": n_pending, "nnz": int(data["value"].numel()),
                       "token_offset": n_total - n_pending})
        pending = {k: [] for k in keys}
        n_pending = 0

    for i, batch in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        kept, flat, b, t = _encode_kept(model, to_device(batch, device), layer, encode, exclude, mask_fn, input_fn)
        pending["example"].append((n_examples + flat // t).cpu())
        pending["position"].append((flat % t).to(torch.int32).cpu())
        for start in range(0, kept.shape[0], chunk_tokens):
            acts = encode(kept[start : start + chunk_tokens]).float()
            n_units = acts.shape[1]
            rows, cols = (acts.abs() > threshold).nonzero(as_tuple=True)
            pending["token"].append((n_pending + start + rows).to(torch.int32).cpu())
            pending["unit"].append(cols.to(torch.int32).cpu())
            pending["value"].append(acts[rows, cols].cpu())
        n_pending += kept.shape[0]
        n_total += kept.shape[0]
        n_examples += b
        if n_pending >= shard_tokens:
            flush()
    flush()
    meta = {"n_units": n_units, "n_tokens": n_total, "n_examples": n_examples, "threshold": threshold,
            "shards": shards, "metadata": metadata or {}}
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2))
    return out_dir


def cache_meta(path: str | Path) -> dict[str, Any]:
    """
    Contents of a cache's ``meta.json``.

    Args:
        path: Cache directory from :func:`cache_units`.

    Returns:
        ``n_units``, ``n_tokens``, ``n_examples``, ``threshold``, ``shards``, ``metadata``.
    """
    return json.loads((Path(path) / META_NAME).read_text())


def shards(path: str | Path) -> Iterator[dict[str, Tensor]]:
    """
    Yields each shard of a cache, with ``token`` as global ids.

    Args:
        path: Cache directory.

    Yields:
        Dicts with ``example``, ``position`` (per cached token) and ``token``
        (int64, global), ``unit`` (int64), ``value`` (per active entry).
    """
    for shard in cache_meta(path)["shards"]:
        data = load_file(str(Path(path) / shard["file"]))
        data["token"] = data["token"].long() + shard["token_offset"]
        data["unit"] = data["unit"].long()
        yield data


@functools.lru_cache(maxsize=4)
def _token_index(path: str, stamp: float) -> tuple[Tensor, Tensor]:
    parts = [(d["example"], d["position"].long()) for d in shards(path)]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def token_index(path: str | Path, tokens: Tensor) -> tuple[Tensor, Tensor]:
    """
    Maps global token ids of a cache to ``(example, position)``.

    Args:
        path: Cache directory.
        tokens: Global token ids.

    Returns:
        ``(example, position)``, shaped like ``tokens``.
    """
    path = Path(path).resolve()
    example, position = _token_index(str(path), (path / META_NAME).stat().st_mtime)
    return example[tokens], position[tokens]


def cache_stats(path: str | Path) -> UnitStats:
    """
    Per-unit statistics of a cache, as :func:`unit_stats` (with the cache's threshold).

    Args:
        path: Cache directory.

    Returns:
        A :class:`UnitStats`.
    """
    meta = cache_meta(path)
    n = meta["n_units"] or 0
    count, total, peak = torch.zeros(n), torch.zeros(n), torch.zeros(n)
    for d in shards(path):
        mags = d["value"].abs()
        count += torch.bincount(d["unit"], minlength=n)
        total += torch.bincount(d["unit"], weights=mags, minlength=n)
        peak = peak.scatter_reduce(0, d["unit"], mags, reduce="amax")
    return UnitStats(frequency=count / max(meta["n_tokens"], 1), mean=total / count.clamp_min(1), max=peak,
                     n_tokens=meta["n_tokens"])


def _top_per_unit(unit: Tensor, value: Tensor, n_top: int, *payload: Tensor) -> tuple[Tensor, ...]:
    """Keeps the ``n_top`` entries of largest ``|value|`` per unit (COO lists)."""
    order = value.abs().argsort(descending=True, stable=True)
    order = order[unit[order].argsort(stable=True)]  # grouped by unit, best first
    unit = unit[order]
    counts = torch.bincount(unit)
    rank = torch.arange(unit.numel()) - (torch.cumsum(counts, 0) - counts)[unit]
    sel = order[rank < n_top]
    return (unit[rank < n_top], value[sel], *(p[sel] for p in payload))


def _best_per_example(unit: Tensor, value: Tensor, token: Tensor, example: Tensor) -> tuple[Tensor, ...]:
    """Keeps each (unit, example) pair's entry of largest ``|value|``."""
    order = value.abs().argsort(descending=True, stable=True)
    key = unit[order] * (int(example.max()) + 1 if example.numel() else 1) + example[order]
    first = torch.ones_like(key, dtype=torch.bool)
    sorted_key, perm = key.sort(stable=True)
    first[perm[1:]] = sorted_key[1:] != sorted_key[:-1]
    keep = order[first]
    return unit[keep], value[keep], token[keep]


def cache_top(path: str | Path, n_top: int = 16, per_example: bool = False) -> TopTokens:
    """
    Top activations per unit of a cache, as :func:`unit_stats`.

    Args:
        path: Cache directory.
        n_top: Entries kept per unit.
        per_example: One entry per example (its peak token), e.g. top images.

    Returns:
        A :class:`TopTokens`.
    """
    meta = cache_meta(path)
    unit = torch.zeros(0, dtype=torch.long)
    val = torch.zeros(0)
    tok = torch.zeros(0, dtype=torch.long)
    for shard, d in zip(meta["shards"], shards(path)):
        u, v, t = d["unit"], d["value"], d["token"]
        if per_example:  # examples never span shards
            u, v, t = _best_per_example(u, v, t, d["example"][t - shard["token_offset"]])
        unit, val, tok = _top_per_unit(torch.cat([unit, u]), torch.cat([val, v]), n_top, torch.cat([tok, t]))
    n = meta["n_units"] or 0
    counts = torch.bincount(unit, minlength=n)
    slot = torch.arange(unit.numel()) - (torch.cumsum(counts, 0) - counts)[unit]
    values = torch.zeros(n, n_top)
    example = torch.full((n, n_top), -1, dtype=torch.long)
    position = torch.full((n, n_top), -1, dtype=torch.long)
    values[unit, slot] = val
    example[unit, slot], position[unit, slot] = token_index(path, tok)
    return TopTokens(values, example, position)


def _collect(path: str | Path, units: Tensor) -> dict[int, tuple[Tensor, Tensor]]:
    """All activations of the given units in one pass: unit -> (global tokens, values), in token order."""
    us, toks, vals = [], [], []
    for d in shards(path):
        hit = torch.isin(d["unit"], units)
        us.append(d["unit"][hit])
        toks.append(d["token"][hit])
        vals.append(d["value"][hit])
    u, tok, val = torch.cat(us), torch.cat(toks), torch.cat(vals)
    order = u.argsort(stable=True)
    u, tok, val = u[order], tok[order], val[order]
    counts = torch.bincount(u, minlength=int(units.max()) + 1)
    out, start = {}, 0
    for unit in units.sort().values.tolist():
        n = int(counts[unit])
        out[unit] = (tok[start : start + n], val[start : start + n])
        start += n
    return out


def unit_activations(path: str | Path, unit: int) -> tuple[Tensor, Tensor, Tensor]:
    """
    Every stored activation of one unit.

    Args:
        path: Cache directory.
        unit: Unit index.

    Returns:
        ``(example, position, value)``, in token order.
    """
    tokens, values = _collect(path, torch.tensor([unit]))[unit]
    return (*token_index(path, tokens), values)


def quantile_examples(
    path: str | Path,
    unit: int,
    n_bins: int = 5,
    per_bin: int = 5,
    seed: int = 0,
) -> list[list[tuple[float, int, int]]]:
    """
    Examples sampled uniformly within activation-magnitude quantile bins.

    Top examples show a unit's most extreme behaviour; sampling each quantile
    bin shows what it responds to at typical activations, which is where
    polysemanticity tends to show up (Bricken et al., 2023).

    Args:
        path: Cache directory.
        unit: Unit index.
        n_bins: Equal-count bins over the unit's stored activations, weakest first.
        per_bin: Examples sampled per bin (fewer if the bin is smaller).
        seed: Sampling seed.

    Returns:
        One list per bin of ``(value, example, position)``, strongest first within the bin.
    """
    example, position, values = unit_activations(path, unit)
    if values.numel() == 0:
        return [[] for _ in range(n_bins)]
    g = torch.Generator().manual_seed(seed)
    bins = []
    for part in torch.tensor_split(values.abs().argsort(), n_bins):
        pick = part[torch.randperm(part.numel(), generator=g)[:per_bin]]
        pick = pick[values[pick].abs().argsort(descending=True)]
        bins.append(list(zip(values[pick].tolist(), example[pick].tolist(), position[pick].tolist())))
    return bins


def cooccurrence(path: str | Path, units: Sequence[int] | Tensor, chunk_tokens: int | None = None) -> Tensor:
    """
    Number of tokens on which each pair of units is active together.

    Args:
        path: Cache directory.
        units: Unit indices (a few thousand at most: the result is ``(n, n)``).
        chunk_tokens: Tokens densified at once to ``(chunk_tokens, n)``;
            ``None`` picks about 64M entries.

    Returns:
        Symmetric counts ``(n, n)``; the diagonal holds each unit's firing
        count. Divide by the diagonal for ``P(j | i)``, or take
        ``C / (diag_i + diag_j - C)`` for Jaccard similarities.
    """
    meta = cache_meta(path)
    units = torch.as_tensor(units, dtype=torch.long)
    lookup = torch.full((meta["n_units"],), -1, dtype=torch.long)
    lookup[units] = torch.arange(units.numel())
    n = units.numel()
    chunk = chunk_tokens or max(1, 2**26 // n)
    counts = torch.zeros(n, n)
    for shard, d in zip(meta["shards"], shards(path)):
        col = lookup[d["unit"]]
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
