"""
Reductions: what to keep of each example's activation.

A reduction maps tokens ``(batch, tokens, d)`` (see :mod:`innards.layouts`) to
``(batch, d)`` (one vector per example) or ``(batch, tokens', d)`` (a subset
of tokens). Feature maps are tokens too, one per spatial position, so the same
reductions work for ViTs and CNNs. Reductions are plain functions; any
``fn(tokens) -> Tensor`` of those shapes works.

Keeping every token is ``reduce=None`` in :func:`innards.extract.extract`.

Typical usage::

    extract(model, loader, "blocks.6", reduce=token(0))           # CLS token
    extract(model, loader, "layer3", reduce=mean_pool())          # spatial mean
    extract(model, loader, "blocks.6", reduce=drop(range(5)))     # drop CLS + 4 registers
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Sequence
)

from torch import Tensor

from .layouts import keep_positions

Reduce = Callable[[Tensor], Tensor]


def token(position: int) -> Reduce:
    """
    Keeps one token, e.g. ``0`` for the CLS token or ``-1`` for the last.

    Args:
        position: Token position (negative indices allowed).

    Returns:
        Reduction to ``(batch, d)``.
    """
    return lambda tokens: tokens[:, position]


def keep(positions: Sequence[int]) -> Reduce:
    """
    Keeps the given token positions, e.g. register tokens.

    Args:
        positions: Token positions (negative indices allowed).

    Returns:
        Reduction to ``(batch, len(positions), d)``.
    """
    return lambda tokens: tokens[:, list(positions)]


def drop(positions: Sequence[int]) -> Reduce:
    """
    Drops the given token positions, e.g. ``range(model.num_prefix_tokens)``
    for the CLS and register tokens of a timm ViT.

    Args:
        positions: Token positions (negative indices allowed).

    Returns:
        Reduction to ``(batch, tokens - len(positions), d)``.
    """
    positions = list(positions)
    return lambda tokens: tokens[:, keep_positions(tokens.shape[1], positions, tokens.device)]


def mean_pool(exclude: Sequence[int] = ()) -> Reduce:
    """
    Averages over tokens (the spatial mean for feature maps).

    Args:
        exclude: Token positions left out of the mean, e.g. ``(0,)`` for CLS.

    Returns:
        Reduction to ``(batch, d)``.
    """
    exclude = list(exclude)
    return lambda tokens: drop(exclude)(tokens).mean(1)


def max_pool(exclude: Sequence[int] = ()) -> Reduce:
    """
    Maximum over tokens, per feature.

    Args:
        exclude: Token positions left out, e.g. ``(0,)`` for CLS.

    Returns:
        Reduction to ``(batch, d)``.
    """
    exclude = list(exclude)
    return lambda tokens: drop(exclude)(tokens).amax(1)
