"""
Activation layouts and conversion to and from tokens.

Module outputs come in different shapes depending on the architecture:

* ``"btd"``: tokens, ``(batch, tokens, d)``, e.g. ViT / DeiT / EVA blocks or
  language-model hidden states.
* ``"bchw"``: convolutional feature maps ``(batch, channels, h, w)``, e.g.
  ResNet, ConvNeXt, EfficientNet stages.
* ``"bhwc"``: channels-last feature maps ``(batch, h, w, channels)``, e.g.
  Swin or MambaOut stages in timm.
* ``"bd"``: one vector per example, e.g. pooled embeddings.

Everything downstream works on tokens ``(batch, tokens, d)``: every spatial
position of a feature map is a token, in row-major ``(h, w)`` order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from einops import rearrange
from torch import Tensor

LAYOUTS = ("btd", "bchw", "bhwc", "bd")


def to_tokens(act: Tensor, layout: str) -> Tensor:
    """
    Converts an activation to ``(batch, tokens, d)``.

    Args:
        act: Activation in ``layout``.
        layout: One of :data:`LAYOUTS`.

    Returns:
        Tensor of shape ``(batch, tokens, d)``.

    Raises:
        ValueError: If the layout is unknown.
    """
    if layout == "btd":
        return act
    if layout == "bchw":
        return rearrange(act, "b c h w -> b (h w) c")
    if layout == "bhwc":
        return rearrange(act, "b h w c -> b (h w) c")
    if layout == "bd":
        return rearrange(act, "b d -> b 1 d")
    raise ValueError(f"unknown layout {layout!r}, expected one of {LAYOUTS}")


def from_tokens(tokens: Tensor, layout: str, like: Tensor) -> Tensor:
    """
    Inverse of :func:`to_tokens`.

    Args:
        tokens: Tensor of shape ``(batch, tokens, d)``.
        layout: Target layout.
        like: A tensor in the target layout, used for spatial sizes.

    Returns:
        Tensor in ``layout``.

    Raises:
        ValueError: If the layout is unknown.
    """
    if layout == "btd":
        return tokens
    if layout == "bchw":
        return rearrange(tokens, "b (h w) c -> b c h w", h=like.shape[2], w=like.shape[3])
    if layout == "bhwc":
        return rearrange(tokens, "b (h w) c -> b h w c", h=like.shape[1], w=like.shape[2])
    if layout == "bd":
        return rearrange(tokens, "b 1 d -> b d")
    raise ValueError(f"unknown layout {layout!r}, expected one of {LAYOUTS}")


def keep_positions(n_tokens: int, exclude: Sequence[int], device: torch.device) -> Tensor:
    """
    Indices of token positions not in ``exclude`` (negative indices allowed).

    Args:
        n_tokens: Sequence length.
        exclude: Positions to drop, e.g. ``(0,)`` for CLS / BOS.
        device: Device of the result.

    Returns:
        Long tensor of kept positions.
    """
    mask = torch.ones(n_tokens, dtype=torch.bool, device=device)
    if exclude:
        mask[torch.tensor([p % n_tokens for p in exclude], device=device)] = False
    return mask.nonzero().squeeze(-1)


def keep_mask(tokens: Tensor, exclude: Sequence[int] = (), mask: Tensor | None = None) -> Tensor:
    """
    Boolean ``(batch, tokens)`` mask of the tokens an edit applies to.

    Args:
        tokens: Tokens ``(batch, tokens, d)``.
        exclude: Positions left out, e.g. ``(0,)`` for CLS / BOS.
        mask: Optional boolean ``(batch, tokens)`` mask, e.g. non-padding tokens.

    Returns:
        The mask.
    """
    keep = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    keep[:, keep_positions(tokens.shape[1], list(exclude), tokens.device)] = True
    if mask is not None:
        keep &= mask.to(keep.device, torch.bool)
    return keep


def grid(shape: Sequence[int], layout: str) -> tuple[int, int] | None:
    """
    Spatial ``(h, w)`` of an activation shape, or ``None`` for non-spatial layouts.

    Args:
        shape: Activation shape, with or without the batch dimension.
        layout: Layout of the activation.

    Returns:
        ``(h, w)`` for ``"bchw"`` and ``"bhwc"``, otherwise ``None``.
    """
    shape = tuple(shape)[-3:]
    if layout == "bchw":
        return shape[1], shape[2]
    if layout == "bhwc":
        return shape[0], shape[1]
    return None


def _is_grid(h: int, w: int, image_hw: tuple[int, int]) -> bool:
    # some stride s maps the image onto an (h, w) grid, up to one position of rounding per side
    big_h, big_w = image_hw
    lo = max(big_h / (h + 1), big_w / (w + 1))
    hi = min(big_h / (h - 1) if h > 1 else math.inf, big_w / (w - 1) if w > 1 else math.inf)
    return lo < hi


def infer_layout(shape: Sequence[int], image_hw: tuple[int, int] | None = None) -> str:
    """
    Guesses the layout of a module output from its shape.

    2-d outputs are ``"bd"`` and 3-d outputs ``"btd"``. For 4-d outputs the
    spatial pair is the one consistent with downsampling the input image
    (``image_hw``); without it, the pair of equal sizes is taken as spatial
    (square inputs). ``"bchw"`` wins ties, as PyTorch's default.

    Args:
        shape: Output shape including the batch dimension.
        image_hw: Height and width of the input images, if known.

    Returns:
        One of :data:`LAYOUTS`.

    Raises:
        ValueError: If the shape has an unsupported rank or neither 4-d
            layout fits; pass the layout explicitly then.
    """
    shape = tuple(shape)
    if len(shape) == 2:
        return "bd"
    if len(shape) == 3:
        return "btd"
    if len(shape) != 4:
        raise ValueError(f"cannot infer a layout for a {len(shape)}-d output {shape}")
    _, a, b, c = shape
    if image_hw is not None:
        fits = {"bchw": _is_grid(b, c, image_hw), "bhwc": _is_grid(a, b, image_hw)}
    else:
        fits = {"bchw": b == c, "bhwc": a == b}
    if fits["bchw"]:
        return "bchw"
    if fits["bhwc"]:
        return "bhwc"
    raise ValueError(f"cannot tell 'bchw' from 'bhwc' for output {shape}; pass the layout explicitly")
