"""
Saving and loading extracted activations.

A run directory holds one subdirectory per layer::

    run.json                    # layers, example shapes, layouts, n_examples, metadata
    targets.safetensors         # optional, one target per example
    blocks.6/
        meta.json               # same format as innards.sae.cache
        shard_00000.safetensors # {"activations": (n_rows, d)}
    layer3/
        ...

Rows are stored in example order (never shuffled): example ``i`` owns rows
``i * r : (i + 1) * r`` with ``r`` the number of tokens kept per example, so
each layer directory is also an activation cache for SAE training::

    CachedActivations("acts/run/blocks.6", batch_tokens=4096)   # innards.sae.cache

Typical usage::

    acts, targets = extract(model, loader, ["blocks.6", "blocks.11"], reduce=token(0))
    save("acts/run", acts, targets, metadata={"model": "vit_base_patch16_224"})
    acts, targets = load("acts/run", layers=["blocks.6"])
"""

from __future__ import annotations

import json
import math
from collections.abc import (
    Callable,
    Sequence
)
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import (
    load_file,
    save_file
)
from torch import Tensor

RUN_NAME = "run.json"
TARGETS_NAME = "targets.safetensors"
META_NAME = "meta.json"  # per layer, as in innards.sae.cache


def open_writer(
    out_dir: str | Path,
    dtype: torch.dtype | None = None,
    shard_rows: int = 2**20,
    metadata: dict[str, Any] | None = None,
) -> tuple[Callable[[Tensor], None], Callable[[], Path]]:
    """
    Opens an activation cache directory for appending rows.

    The directory has the format of :func:`innards.sae.cache.cache_activations`
    (one ``"activations"`` field), so it can be read by
    :class:`innards.sae.cache.CachedActivations`.

    Args:
        out_dir: Output directory; created if needed.
        dtype: Storage dtype; ``None`` keeps the dtype of the first rows.
        shard_rows: Rows per shard file.
        metadata: JSON-serialisable extra information stored in ``meta.json``.

    Returns:
        ``(write, close)``: ``write(rows)`` appends ``(n, d)`` rows, ``close()``
        writes the remaining rows and ``meta.json`` and returns the directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pending: list[Tensor] = []
    shards: list[dict[str, Any]] = []
    state = {"n_pending": 0, "n_rows": 0, "d": None, "dtype": dtype}

    def flush(final: bool) -> None:
        data = torch.cat(pending) if pending else torch.empty(0, state["d"] or 0)
        pending.clear()
        while data.shape[0] >= shard_rows or (final and data.shape[0] > 0):
            chunk, data = data[:shard_rows], data[shard_rows:]
            name = f"shard_{len(shards):05d}.safetensors"
            save_file({"activations": chunk.contiguous()}, str(out_dir / name))
            shards.append({"file": name, "n_tokens": int(chunk.shape[0])})
        if data.shape[0]:
            pending.append(data)
        state["n_pending"] = data.shape[0]

    def write(rows: Tensor) -> None:
        if state["dtype"] is None:
            state["dtype"] = rows.dtype
        state["d"] = rows.shape[-1]
        pending.append(rows.detach().to("cpu", state["dtype"]))
        state["n_pending"] += rows.shape[0]
        state["n_rows"] += rows.shape[0]
        if state["n_pending"] >= shard_rows:
            flush(final=False)

    def close() -> Path:
        flush(final=True)
        meta = {
            "kind": "tensor",
            "fields": [["activations", state["d"]]],
            "d_in": state["d"],
            "d_out": None,
            "n_tokens": state["n_rows"],
            "dtype": str(state["dtype"]).removeprefix("torch."),
            "shards": shards,
            "metadata": metadata or {},
        }
        (out_dir / META_NAME).write_text(json.dumps(meta, indent=2))
        return out_dir

    return write, close


def write_run(
    out_dir: str | Path,
    n_examples: int,
    layers: dict[str, dict[str, Any]],
    targets: Tensor | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Writes ``run.json`` (and the targets) of a run directory.

    Args:
        out_dir: Run directory.
        n_examples: Number of examples.
        layers: Per layer: ``example_shape`` (shape of one example's
            activation) and optionally ``layout`` and ``grid``.
        targets: One target per example, or ``None``.
        metadata: JSON-serialisable extra information, e.g. model and dataset.

    Returns:
        The run directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if targets is not None:
        save_file({"targets": targets.detach().cpu().contiguous()}, str(out_dir / TARGETS_NAME))
    run = {
        "n_examples": n_examples,
        "layers": {name: {"dir": name, **info} for name, info in layers.items()},
        "targets": targets is not None,
        "metadata": metadata or {},
    }
    (out_dir / RUN_NAME).write_text(json.dumps(run, indent=2))
    return out_dir


def save(
    out_dir: str | Path,
    acts: dict[str, Tensor],
    targets: Tensor | None = None,
    info: dict[str, dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    shard_rows: int = 2**20,
) -> Path:
    """
    Saves in-memory activations as a run directory.

    Args:
        out_dir: Run directory; created if needed.
        acts: Layer name -> activations ``(n_examples, ..., d)``.
        targets: One target per example, or ``None``.
        info: Optional extra per-layer information for ``run.json``, e.g.
            the ``layout`` and ``grid`` from :func:`innards.extract.describe`.
        metadata: JSON-serialisable extra information, e.g. model and dataset.
        dtype: Storage dtype; ``None`` keeps each tensor's dtype.
        shard_rows: Rows per shard file.

    Returns:
        The run directory.

    Raises:
        ValueError: If the layers have different numbers of examples.
    """
    n = {a.shape[0] for a in acts.values()}
    if len(n) != 1:
        raise ValueError(f"layers have different numbers of examples: {n}")
    layers = {}
    for name, act in acts.items():
        layers[name] = {**(info or {}).get(name, {}), "example_shape": list(act.shape[1:])}
        write, close = open_writer(Path(out_dir) / name, dtype, shard_rows, {"layer": name, **layers[name]})
        write(act.reshape(-1, act.shape[-1]))
        close()
    return write_run(out_dir, n.pop(), layers, targets, metadata)


def load_info(run_dir: str | Path) -> dict[str, Any]:
    """
    Reads ``run.json`` of a run directory.

    Args:
        run_dir: Run directory.

    Returns:
        The run description: ``n_examples``, ``layers`` (per layer
        ``example_shape``, ``layout``, ``grid``), ``targets``, ``metadata``.
    """
    return json.loads((Path(run_dir) / RUN_NAME).read_text())


def load_layer(run_dir: str | Path, layer: str, flat: bool = False, device: str | torch.device = "cpu") -> Tensor:
    """
    Loads one layer of a run directory.

    Args:
        run_dir: Run directory.
        layer: Layer name.
        flat: Return rows ``(n_rows, d)`` instead of ``(n_examples, ..., d)``.
        device: Device of the result.

    Returns:
        The activations.
    """
    run = load_info(run_dir)
    layer_dir = Path(run_dir) / run["layers"][layer]["dir"]
    meta = json.loads((layer_dir / META_NAME).read_text())
    rows = torch.cat([load_file(str(layer_dir / s["file"]), device=str(device))["activations"]
                      for s in meta["shards"]])
    if flat:
        return rows
    shape = run["layers"][layer]["example_shape"]
    return rows.reshape(run["n_examples"], *shape)


def load(
    run_dir: str | Path,
    layers: Sequence[str] | None = None,
    flat: bool = False,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, Tensor], Tensor | None]:
    """
    Loads a run directory.

    Args:
        run_dir: Run directory.
        layers: Layers to load; ``None`` loads all.
        flat: Return rows ``(n_rows, d)`` instead of ``(n_examples, ..., d)``.
        device: Device of the result.

    Returns:
        ``(acts, targets)``: layer name -> activations, and the targets
        (``None`` if the run has none).
    """
    run = load_info(run_dir)
    layers = list(run["layers"]) if layers is None else list(layers)
    acts = {name: load_layer(run_dir, name, flat, device) for name in layers}
    targets = None
    if run["targets"]:
        targets = load_file(str(Path(run_dir) / TARGETS_NAME), device=str(device))["targets"]
    return acts, targets


def rows_per_example(example_shape: Sequence[int]) -> int:
    """
    Number of stored rows per example, e.g. the number of kept tokens.

    Args:
        example_shape: Shape of one example's activation, ``(..., d)``.

    Returns:
        ``prod(example_shape[:-1])``.
    """
    return math.prod(example_shape[:-1])
