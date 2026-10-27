"""
Loading models and finding their layers.

Every function in :mod:`innards` that touches a model takes an nnsight
:class:`~nnsight.NNsight` (or any other nnsight model, e.g. ``LanguageModel``)
and uses only its public tracing interface. Plain ``nn.Module`` s are wrapped
with :func:`wrap`.

Layers are named by their dotted module path, e.g. ``"blocks.6"`` or
``"stages.2.blocks.1"``; :func:`module_names` lists them.

Typical usage::

    model, preprocess = load("vit_base_patch16_224")
    module_names(model, depth=2)          # ['blocks.0', ..., 'blocks.11', ...]
    stage_names(model)                    # timm's feature stages
"""

from __future__ import annotations

import re
from collections.abc import (
    Callable,
    Iterator
)
from contextlib import contextmanager
from typing import Any

import torch
from nnsight import NNsight
from nnsight.intervention.envoy import Envoy
from torch import nn


def wrap(model: nn.Module | Envoy) -> Envoy:
    """
    Wraps a module in :class:`~nnsight.NNsight`; nnsight models pass through.

    Args:
        model: A PyTorch module or an nnsight model.

    Returns:
        The nnsight model.
    """
    return model if isinstance(model, Envoy) else NNsight(model)


def unwrap(model: nn.Module | Envoy) -> nn.Module:
    """
    The underlying PyTorch module of an nnsight model.

    Args:
        model: An nnsight model or a plain module.

    Returns:
        The PyTorch module.
    """
    return model._module if isinstance(model, Envoy) else model


@contextmanager
def frozen(*modules: nn.Module | Envoy) -> Iterator[None]:
    """
    Disables gradients of the modules' parameters inside the block, restoring them after.

    For gradients that only need to reach activations or inputs (attribution,
    Jacobians), so parameter gradients are neither computed nor accumulated.

    Args:
        *modules: Modules or nnsight models.

    Yields:
        None.
    """
    modules = [unwrap(m) for m in modules]
    flags = [[p.requires_grad for p in m.parameters()] for m in modules]
    for m in modules:
        m.requires_grad_(False)
    try:
        yield
    finally:
        for m, fl in zip(modules, flags):
            for p, f in zip(m.parameters(), fl):
                p.requires_grad_(f)


def layer_name(model: nn.Module | Envoy, layer: str | nn.Module | Envoy) -> str:
    """
    Dotted path of a layer, given as a path, a module or an nnsight envoy.

    Lets functions accept ``model.blocks[6]`` (of the plain or the wrapped
    model) as well as ``"blocks.6"``.

    Args:
        model: The model the layer belongs to.
        layer: The layer.

    Returns:
        The layer's path in ``model``.

    Raises:
        ValueError: If the module is not part of the model.
    """
    if isinstance(layer, str):
        return layer
    module = unwrap(layer)
    for name, m in unwrap(model).named_modules():
        if m is module and name:
            return name
    raise ValueError(f"{type(module).__name__} is not a submodule of the model")


def load(
    name: str,
    pretrained: bool = True,
    device: str | torch.device = "cpu",
    **kwargs: Any,
) -> tuple[Envoy, Callable]:
    """
    Creates a timm model in eval mode, wrapped for nnsight, with its preprocessing.

    Needs ``timm`` (the ``vision`` extra).

    Args:
        name: timm model name, e.g. ``"resnet50"`` or ``"vit_base_patch16_224"``.
        pretrained: Load pretrained weights.
        device: Device of the model.
        **kwargs: Passed to :func:`timm.create_model`, e.g. ``img_size``.

    Returns:
        ``(model, preprocess)``: the nnsight model and timm's evaluation
        transform for it (PIL image -> normalised tensor), ready for a
        ``torchvision`` dataset.
    """
    import timm

    module = timm.create_model(name, pretrained=pretrained, **kwargs).eval().to(device)
    cfg = timm.data.resolve_model_data_config(module)
    return wrap(module), timm.data.create_transform(**cfg, is_training=False)


def module_names(model: nn.Module | Envoy, depth: int | None = None, pattern: str | None = None) -> list[str]:
    """
    Dotted paths of a model's submodules, in definition order.

    Args:
        model: An nnsight model or a plain module.
        depth: Keep only paths with at most this many components, e.g. ``2``
            for ``"blocks.6"`` but not ``"blocks.6.attn"``.
        pattern: Keep only paths matching this regular expression
            (``re.search``), e.g. ``r"blocks\\.\\d+$"``.

    Returns:
        Module paths, without the root.
    """
    names = [n for n, _ in unwrap(model).named_modules() if n]
    if depth is not None:
        names = [n for n in names if n.count(".") < depth]
    if pattern is not None:
        names = [n for n in names if re.search(pattern, n)]
    return names


def stage_names(model: nn.Module | Envoy) -> list[str]:
    """
    The feature stages timm declares for a model (``model.feature_info``).

    For ViTs these are the blocks, for CNNs the downsampling stages; a
    sensible default set of layers for any timm model.

    Args:
        model: A timm model, wrapped or not.

    Returns:
        Module paths of the stages.

    Raises:
        ValueError: If the model has no ``feature_info``.
    """
    info = getattr(unwrap(model), "feature_info", None)
    if not info:
        raise ValueError("model has no feature_info; use module_names instead")
    return [f["module"] for f in info]
