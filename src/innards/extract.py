"""
Extracting activations with nnsight.

:func:`capture` reads the outputs of several layers in one nnsight trace and
stops the forward pass after the last one. :func:`iter_extract` runs it over
any iterable of batches (a ``DataLoader``, a list of tensors, ...), converts
each output to tokens and applies a reduction (see :mod:`innards.reduce`);
:func:`extract` collects the result in memory and :func:`extract_to_disk`
streams it to a run directory (see :mod:`innards.store`).

Batches can be tensors, ``(input, target, ...)`` tuples or dicts (keyword
arguments of the model); ``input_fn`` and ``target_fn`` override how they are
split. Activations come back as tokens ``(n, tokens, d)``, or ``(n, d)`` after
a pooling reduction; feature maps have one token per spatial position, in
row-major order of the grid that :func:`describe` reports.

nnsight reads module outputs in execution order only, so :func:`describe`
first runs one partial forward pass with plain hooks to find that order (and
each layer's shape and layout).

Typical usage::

    model, preprocess = load("resnet50")
    loader = DataLoader(ImageFolder("val", preprocess), batch_size=64)
    acts, targets = extract(model, loader, ["layer2", "layer4"], reduce=mean_pool())
    extract_to_disk(model, loader, ["layer2", "layer4"], "acts/resnet50_val")
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Sequence
)
from itertools import chain
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

import torch
from nnsight.intervention.envoy import Envoy
from torch import (
    Tensor,
    nn
)

from .layouts import (
    grid,
    infer_layout,
    to_tokens
)
from .models import (
    unwrap,
    wrap
)
from .reduce import Reduce
from .store import (
    open_writer,
    write_run
)

Layers = str | Sequence[str]
Reduction = Reduce | dict[str, Reduce | None] | None


def default_input_fn(batch: Any) -> tuple[tuple, dict]:
    """
    Turns a loader batch into model ``(args, kwargs)``.

    Dicts (e.g. Hugging Face tokenizer output) become keyword arguments;
    tuples or lists such as ``(images, labels)`` pass their first element;
    tensors are passed as is.

    Args:
        batch: A batch from the input loader.

    Returns:
        ``(args, kwargs)`` for ``model(*args, **kwargs)``.
    """
    if isinstance(batch, dict):
        return (), dict(batch)
    if isinstance(batch, (tuple, list)):
        return (batch[0],), {}
    return (batch,), {}


def default_target_fn(batch: Any) -> Tensor | None:
    """
    The targets of a loader batch, if it has any.

    Args:
        batch: A batch from the input loader.

    Returns:
        The second element of a tuple or list, the ``"label"`` / ``"labels"``
        entry of a dict, otherwise ``None``.
    """
    if isinstance(batch, (tuple, list)) and len(batch) > 1:
        return torch.as_tensor(batch[1])
    if isinstance(batch, dict):
        for key in ("label", "labels"):
            if key in batch:
                return torch.as_tensor(batch[key])
    return None


def to_device(obj: Any, device: torch.device | str) -> Any:
    """
    Moves the tensors of a (nested) batch to ``device``.

    Args:
        obj: Tensor, dict, tuple or list; other objects pass through.
        device: Target device.

    Returns:
        The batch on ``device``.
    """
    if isinstance(obj, Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def _as_list(layers: Layers) -> list[str]:
    return [layers] if isinstance(layers, str) else list(layers)


def _first(out: Any) -> Any:
    return out[0] if isinstance(out, (tuple, list)) else out


class _StopForward(Exception):
    pass


def describe(
    model: nn.Module | Envoy,
    batch: Any,
    layers: Layers,
    layouts: str | dict[str, str] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
) -> dict[str, dict[str, Any]]:
    """
    Execution order, output shape and layout of layers, from one partial forward pass.

    Args:
        model: An nnsight model or a plain module.
        batch: An input batch (on the model's device).
        layers: Layer names.
        layouts: Layout of every layer, or per layer; missing layouts are
            inferred with :func:`innards.layouts.infer_layout`.
        input_fn: Maps the batch to ``(args, kwargs)``.

    Returns:
        Layer name -> ``{"layout", "shape", "grid"}``, in execution order
        (the order in which the layers' outputs are complete). ``shape`` is
        one example's output shape, ``grid`` the spatial ``(h, w)`` or ``None``.

    Raises:
        RuntimeError: If a layer did not run.
    """
    module, layers = unwrap(model), _as_list(layers)
    shapes: dict[str, tuple[int, ...]] = {}

    def make_hook(name: str):
        def hook(_mod, _inp, out):
            shapes.setdefault(name, tuple(_first(out).shape))
            if len(shapes) == len(layers):
                raise _StopForward
        return hook

    handles = [module.get_submodule(n).register_forward_hook(make_hook(n)) for n in layers]
    args, kwargs = input_fn(batch)
    try:
        with torch.no_grad():
            module(*args, **kwargs)
    except _StopForward:
        pass
    finally:
        for h in handles:
            h.remove()
    if missing := [n for n in layers if n not in shapes]:
        raise RuntimeError(f"layers did not run during the forward pass: {missing}")

    image = args[0] if args else None
    image_hw = tuple(image.shape[-2:]) if isinstance(image, Tensor) and image.ndim == 4 else None
    info = {}
    for name, shape in shapes.items():
        layout = layouts.get(name) if isinstance(layouts, dict) else layouts
        layout = layout or infer_layout(shape, image_hw)
        info[name] = {"layout": layout, "shape": list(shape[1:]), "grid": grid(shape, layout)}
    return info


_PLANS: WeakKeyDictionary = WeakKeyDictionary()


def plan(
    model: nn.Module | Envoy,
    batch: Any,
    layers: Layers,
    layouts: str | dict[str, str] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
) -> tuple[list[str], dict[str, str]]:
    """
    Execution order and layouts of layers, cached per model.

    Runs :func:`describe` the first time a set of layers is used with a
    model; later calls reuse the result. A single layer with a given layout
    needs no forward pass at all.

    Args:
        model: An nnsight model or a plain module.
        batch: An input batch (on the model's device), used on a cache miss.
        layers: Layer names.
        layouts: Layouts, see :func:`describe`.
        input_fn: Maps the batch to ``(args, kwargs)``.

    Returns:
        ``(order, layouts)``: the layers in execution order and the layout of each.
    """
    layers = _as_list(layers)
    given = {n: layouts.get(n) if isinstance(layouts, dict) else layouts for n in layers}
    if len(layers) <= 1 and all(given.values()):
        return layers, given
    cache = _PLANS.setdefault(unwrap(model), {})
    key = tuple(given.items())
    if key not in cache:
        info = describe(model, batch, layers, layouts, input_fn)
        cache[key] = (list(info), {n: i["layout"] for n, i in info.items()})
    return cache[key]


def capture(
    model: nn.Module | Envoy,
    batch: Any,
    layers: Layers,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
) -> dict[str, Tensor]:
    """
    Raw outputs of several layers, from one nnsight trace.

    The forward pass stops after the last layer. Tuple outputs are reduced to
    their first element. Gradients flow through the outputs unless the call
    is wrapped in ``torch.no_grad``.

    Args:
        model: An nnsight model or a plain module.
        batch: An input batch (on the model's device).
        layers: Layer names, in any order (see :func:`plan`).
        input_fn: Maps the batch to ``(args, kwargs)``.

    Returns:
        Layer name -> output.
    """
    model, layers = wrap(model), _as_list(layers)
    if len(layers) > 1:
        layers = plan(model, batch, layers, input_fn=input_fn)[0]
    args, kwargs = input_fn(batch)
    out: dict[str, Tensor] = {}
    with model.trace(*args, **kwargs) as tracer:
        for name in layers:
            out[name] = _first(model.get(name).output).save()
        tracer.stop()
    return out


def _reduce(act: Tensor, layout: str, reduce: Reduce | None) -> Tensor:
    if reduce is None:
        return act if layout == "bd" else to_tokens(act, layout)
    return reduce(to_tokens(act, layout))


def iter_extract(
    model: nn.Module | Envoy,
    batches: Iterable[Any],
    layers: Layers,
    reduce: Reduction = None,
    layouts: str | dict[str, str] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    target_fn: Callable[[Any], Tensor | None] = default_target_fn,
    device: torch.device | str | None = None,
    info: dict[str, dict[str, Any]] | None = None,
) -> Iterator[tuple[dict[str, Tensor], Tensor | None]]:
    """
    Reduced activations per batch.

    Args:
        model: An nnsight model or a plain module.
        batches: Any iterable of batches, e.g. a ``DataLoader``.
        layers: Layer names.
        reduce: A reduction for every layer, or per layer (missing layers
            keep all tokens); ``None`` keeps all tokens.
        layouts: Layouts, see :func:`describe`.
        input_fn: Maps a batch to model ``(args, kwargs)``.
        target_fn: Maps a batch to its targets, or ``None``.
        device: Device the batches are moved to; defaults to the model's.
        info: The result of :func:`describe`, if already computed.

    Yields:
        ``(acts, targets)`` per batch: layer name -> ``(batch, tokens, d)``
        or ``(batch, d)`` activations on ``device``, and the batch targets.
    """
    model, layers = wrap(model), _as_list(layers)
    if device is None:
        device = next(unwrap(model).parameters()).device
    for batch in batches:
        batch = to_device(batch, device)
        if info is None:
            info = describe(model, batch, layers, layouts, input_fn)
        with torch.no_grad():
            raw = capture(model, batch, list(info), input_fn)
        acts = {}
        for name in layers:
            fn = reduce.get(name) if isinstance(reduce, dict) else reduce
            acts[name] = _reduce(raw[name], info[name]["layout"], fn)
        yield acts, target_fn(batch)


def extract(
    model: nn.Module | Envoy,
    batches: Iterable[Any],
    layers: Layers,
    reduce: Reduction = None,
    layouts: str | dict[str, str] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    target_fn: Callable[[Any], Tensor | None] = default_target_fn,
    device: torch.device | str | None = None,
    out_device: torch.device | str = "cpu",
) -> tuple[dict[str, Tensor], Tensor | None]:
    """
    Activations of a whole dataset, in memory.

    Args:
        model: An nnsight model or a plain module.
        batches: Any iterable of batches, e.g. a ``DataLoader``.
        layers: Layer names.
        reduce: Reduction(s), see :func:`iter_extract`.
        layouts: Layouts, see :func:`describe`.
        input_fn: Maps a batch to model ``(args, kwargs)``.
        target_fn: Maps a batch to its targets, or ``None``.
        device: Device the batches are moved to; defaults to the model's.
        out_device: Device the activations are collected on.

    Returns:
        ``(acts, targets)``: layer name -> ``(n, tokens, d)`` or ``(n, d)``
        activations in batch order, and the targets (``None`` if the batches
        have none).
    """
    acts: dict[str, list[Tensor]] = {}
    targets: list[Tensor] = []
    for batch_acts, batch_targets in iter_extract(model, batches, layers, reduce, layouts, input_fn,
                                                  target_fn, device):
        for name, act in batch_acts.items():
            acts.setdefault(name, []).append(act.to(out_device))
        if batch_targets is not None:
            targets.append(batch_targets.to(out_device))
    return {n: torch.cat(a) for n, a in acts.items()}, torch.cat(targets) if targets else None


def extract_to_disk(
    model: nn.Module | Envoy,
    batches: Iterable[Any],
    layers: Layers,
    out_dir: str | Path,
    reduce: Reduction = None,
    layouts: str | dict[str, str] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    target_fn: Callable[[Any], Tensor | None] = default_target_fn,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    shard_rows: int = 2**20,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Streams the activations of a dataset to a run directory (see :mod:`innards.store`).

    Only one shard per layer is held in memory at a time.

    Args:
        model: An nnsight model or a plain module.
        batches: Any iterable of batches, e.g. a ``DataLoader``. Its order is
            the stored example order, so don't shuffle.
        layers: Layer names.
        out_dir: Run directory; created if needed.
        reduce: Reduction(s), see :func:`iter_extract`.
        layouts: Layouts, see :func:`describe`.
        input_fn: Maps a batch to model ``(args, kwargs)``.
        target_fn: Maps a batch to its targets, or ``None``.
        device: Device the batches are moved to; defaults to the model's.
        dtype: Storage dtype, e.g. ``torch.bfloat16``; ``None`` keeps the
            model's.
        shard_rows: Rows per shard file.
        metadata: JSON-serialisable extra information, e.g. model and dataset
            names, stored in ``run.json`` and every layer's ``meta.json``.

    Returns:
        The run directory.

    Raises:
        ValueError: If ``batches`` is empty.
    """
    model, layers, out_dir = wrap(model), _as_list(layers), Path(out_dir)
    if device is None:
        device = next(unwrap(model).parameters()).device
    batches = iter(batches)
    first = next(batches, None)
    if first is None:
        raise ValueError("batches is empty")
    info = describe(model, to_device(first, device), layers, layouts, input_fn)

    writers: dict[str, tuple[Callable, Callable]] = {}
    layer_info: dict[str, dict[str, Any]] = {}
    targets: list[Tensor] = []
    n_examples = 0
    for acts, batch_targets in iter_extract(model, chain([first], batches), layers, reduce, layouts,
                                            input_fn, target_fn, device, info):
        for name, act in acts.items():
            if name not in writers:
                layer_info[name] = {"layout": info[name]["layout"], "grid": info[name]["grid"],
                                    "output_shape": info[name]["shape"], "example_shape": list(act.shape[1:])}
                writers[name] = open_writer(out_dir / name, dtype, shard_rows,
                                            {"layer": name, **layer_info[name], **(metadata or {})})
            writers[name][0](act.reshape(-1, act.shape[-1]))
        n_examples += next(iter(acts.values())).shape[0]
        if batch_targets is not None:
            targets.append(batch_targets.cpu())
    for _, close in writers.values():
        close()
    return write_run(out_dir, n_examples, {n: layer_info[n] for n in layers},
                     torch.cat(targets) if targets else None, metadata)
