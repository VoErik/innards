"""
Saving and loading activations.

Two kinds of directories, one shard format:

* a **row cache**: aligned rows in safetensors shards plus ``meta.json``,
  written by :func:`open_writer` / :func:`write_rows`, read in shuffled
  batches by :class:`RowLoader` (e.g. to train an SAE or a probe on tokens).
  Rows can be single tensors, ``(input, target)`` pairs (transcoders) or
  dicts of aligned fields (several layers at once).
* a **run directory** from :func:`innards.extract.extract_to_disk` or
  :func:`save`: one row cache per layer, plus ``run.json`` describing the
  layers (example shapes, layouts, grids) and the targets::

      run.json                    # layers, example shapes, layouts, n_examples, metadata
      targets.safetensors         # optional, one target per example
      blocks.6/                   # a row cache
          meta.json
          shard_00000.safetensors # {"activations": (n_rows, d)}

  Rows are stored in example order: example ``i`` owns rows ``i * r : (i + 1) * r``
  with ``r`` the number of tokens kept per example.

Typical usage::

    acts, targets = extract(model, loader, ["blocks.6", "blocks.11"], reduce=token(0))
    save("acts/run", acts, targets, metadata={"model": "vit_base_patch16_224"})
    acts, targets = load("acts/run", layers=["blocks.6"])
    for batch in RowLoader("acts/run/blocks.6", batch_rows=4096):   # shuffled rows
        ...
"""

from __future__ import annotations

import json
import math
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Sequence
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

Rows = Tensor | tuple[Tensor, Tensor] | dict[str, Tensor]

RUN_NAME = "run.json"
TARGETS_NAME = "targets.safetensors"
META_NAME = "meta.json"  # per row cache


def _fields(rows: Rows) -> tuple[str, list[tuple[str, Tensor]]]:
    """The kind of a row batch and its named fields, flattened to ``(n, d)``."""
    if isinstance(rows, dict):
        kind, fields = "dict", list(rows.items())
    elif isinstance(rows, (tuple, list)):
        kind, fields = "pair", [("activations", rows[0]), ("targets", rows[1])]
    else:
        kind, fields = "tensor", [("activations", rows)]
    return kind, [(name, pack([t.detach()], "* d")[0]) for name, t in fields]


def split_fields(flat: Tensor, kind: str, names: Sequence[str], widths: Sequence[int]) -> Rows:
    """
    Splits side-by-side fields ``(n, sum(widths))`` back into the row structure ``kind``.

    Args:
        flat: Concatenated fields.
        kind: ``"tensor"``, ``"pair"`` (input, target) or ``"dict"``.
        names: Field names, in order.
        widths: Field widths, in order.

    Returns:
        A tensor, an ``(input, target)`` pair or a dict name -> tensor.
    """
    parts = flat.split(list(widths), dim=-1)
    if kind == "tensor":
        return parts[0]
    if kind == "pair":
        return tuple(parts)
    return dict(zip(names, parts))


def open_writer(
    out_dir: str | Path,
    dtype: torch.dtype | None = None,
    shard_rows: int = 2**20,
    metadata: dict[str, Any] | None = None,
) -> tuple[Callable[[Rows], None], Callable[[], Path]]:
    """
    Opens a row cache for appending rows.

    Rows are a tensor ``(..., d)``, an ``(input, target)`` pair of such
    tensors (e.g. for transcoders) or a dict of them (e.g. several layers);
    leading dimensions are flattened. Fields are stored side by side, so row
    ``i`` of every field stays aligned; :class:`RowLoader` serves them back in
    the same structure.

    Args:
        out_dir: Output directory; created if needed.
        dtype: Storage dtype, e.g. ``torch.bfloat16``; ``None`` keeps the
            dtype of the first rows.
        shard_rows: Rows per shard file.
        metadata: JSON-serialisable extra information stored in ``meta.json``.

    Returns:
        ``(write, close)``: ``write(rows)`` appends rows, ``close()`` writes
        the remaining rows and ``meta.json`` and returns the directory.

    Raises:
        ValueError: On ``close`` if nothing was written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pending: list[Tensor] = []
    shards: list[dict[str, Any]] = []
    state: dict[str, Any] = {"n_pending": 0, "n_rows": 0, "dtype": dtype, "kind": None, "names": [], "widths": []}

    def flush(final: bool) -> None:
        data = torch.cat(pending) if pending else None
        pending.clear()
        while data is not None and (data.shape[0] >= shard_rows or (final and data.shape[0] > 0)):
            chunk, data = data[:shard_rows], data[shard_rows:]
            name = f"shard_{len(shards):05d}.safetensors"
            parts = chunk.split(state["widths"], dim=-1)
            save_file({f: t.contiguous() for f, t in zip(state["names"], parts)}, str(out_dir / name))
            shards.append({"file": name, "n_rows": int(chunk.shape[0])})
        if data is not None and data.shape[0]:
            pending.append(data)
        state["n_pending"] = 0 if data is None else data.shape[0]

    def write(rows: Rows) -> None:
        kind, fields = _fields(rows)
        if state["kind"] is None:
            state["kind"], state["names"] = kind, [n for n, _ in fields]
            state["widths"] = [t.shape[-1] for _, t in fields]
            state["dtype"] = state["dtype"] or fields[0][1].dtype
        flat = torch.cat([t.to("cpu", state["dtype"]) for _, t in fields], -1)
        pending.append(flat)
        state["n_pending"] += flat.shape[0]
        state["n_rows"] += flat.shape[0]
        if state["n_pending"] >= shard_rows:
            flush(final=False)

    def close() -> Path:
        if state["kind"] is None:
            raise ValueError("no rows were written")
        flush(final=True)
        widths = state["widths"]
        meta = {
            "kind": state["kind"],
            "fields": [[f, w] for f, w in zip(state["names"], widths)],
            "d_in": widths[0],
            "d_out": widths[1] if state["kind"] == "pair" else None,
            "n_rows": state["n_rows"],
            "dtype": str(state["dtype"]).removeprefix("torch."),
            "shards": shards,
            "metadata": metadata or {},
        }
        (out_dir / META_NAME).write_text(json.dumps(meta, indent=2))
        return out_dir

    return write, close


def write_rows(
    batches: Iterable[Rows],
    out_dir: str | Path,
    dtype: torch.dtype | None = None,
    shard_rows: int = 2**20,
    max_rows: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Writes an iterable of row batches to a row cache.

    E.g. the shuffled rows of :class:`innards.extract.RowStream`, so that
    training several SAEs (or epochs) does not re-run the model.

    Args:
        batches: Row batches (tensors, pairs or dicts, see :func:`open_writer`).
        out_dir: Output directory; created if needed. Existing shards with the
            same names are overwritten.
        dtype: Storage dtype, e.g. ``torch.bfloat16``.
        shard_rows: Rows per shard file.
        max_rows: Stop after this many rows.
        metadata: JSON-serialisable extra information.

    Returns:
        The output directory.
    """
    write, close = open_writer(out_dir, dtype, shard_rows, metadata)
    n = 0
    for rows in batches:
        if max_rows is not None:
            take = max_rows - n
            if isinstance(rows, dict):
                rows = {k: v.reshape(-1, v.shape[-1])[:take] for k, v in rows.items()}
            elif isinstance(rows, (tuple, list)):
                rows = tuple(v.reshape(-1, v.shape[-1])[:take] for v in rows)
            else:
                rows = rows.reshape(-1, rows.shape[-1])[:take]
        write(rows)
        n += _fields(rows)[1][0][1].shape[0]
        if max_rows is not None and n >= max_rows:
            break
    return close()


class RowLoader:
    """
    Shuffled batches of rows from a row cache, re-iterable like a ``DataLoader``.

    Each pass visits the shards in a random order and shuffles rows within
    each shard; leftovers carry over to the next shard, so every batch has
    exactly ``batch_rows`` rows (the last partial batch is dropped). A
    background thread loads the next shard while the current one is served,
    so at most two shards are in memory; on CUDA the shards are pinned and
    copied asynchronously. Caches of pairs or dicts yield pairs or dicts.

    Args:
        path: A row cache, e.g. a layer directory of a run.
        batch_rows: Rows per batch.
        shuffle: Shuffle shard order and rows within shards.
        seed: Seed of the first pass; later passes use ``seed + pass``.
        device: Device the batches are moved to.
        dtype: Dtype of the served batches; ``None`` keeps the storage dtype.

    Attributes:
        batch_rows: Rows per batch (read by the SAE trainer).
        meta: Contents of ``meta.json``.
    """

    def __init__(
        self,
        path: str | Path,
        batch_rows: int = 4096,
        shuffle: bool = True,
        seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        self.path = Path(path)
        self.meta = json.loads((self.path / META_NAME).read_text())
        self.batch_rows = batch_rows
        self.shuffle = shuffle
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        self._pass = 0
        self._kind = self.meta["kind"]
        self._names = [f for f, _ in self.meta["fields"]]
        self._widths = [int(w) for _, w in self.meta["fields"]]

    @property
    def d_in(self) -> int:
        """Width of the (first) field."""
        return int(self.meta["d_in"])

    @property
    def n_rows(self) -> int:
        """Total number of rows."""
        return int(self.meta["n_rows"])

    def __len__(self) -> int:
        return self.n_rows // self.batch_rows

    def _load_shard(self, name: str, g: torch.Generator) -> Tensor:
        tensors = load_file(str(self.path / name))
        data = torch.cat([tensors[f] for f in self._names], -1)  # side by side: rows stay aligned
        if self.shuffle:
            data = data[torch.randperm(data.shape[0], generator=g)]
        if self.device.type == "cuda":
            data = data.pin_memory()
        return data

    def _serve(self, batch: Tensor) -> Rows:
        batch = batch.to(self.device, non_blocking=True)
        batch = batch.to(self.dtype) if self.dtype is not None else batch
        return split_fields(batch, self._kind, self._names, self._widths)

    def __iter__(self) -> Iterator[Rows]:
        g = torch.Generator().manual_seed(self.seed + self._pass)
        self._pass += 1
        shards = self.meta["shards"]
        order = torch.randperm(len(shards), generator=g).tolist() if self.shuffle else list(range(len(shards)))
        bs = self.batch_rows
        carry: Tensor | None = None
        with ThreadPoolExecutor(max_workers=1) as pool:  # one worker: shards are shuffled from ``g`` in order
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
