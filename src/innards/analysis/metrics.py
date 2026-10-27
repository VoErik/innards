"""
Metrics for causal experiments.

Two kinds of plain functions, all returning one value per example ``(batch,)``:

* targets ``fn(out, batch) -> (batch,)``: a property of one model output,
  e.g. a logit; differentiated by the gradient-based tools.
* comparisons ``fn(clean, edited, batch) -> (batch,)``: how an edit changed
  the output, e.g. a KL divergence; :func:`change` turns a target into one.

Outputs are logits ``(batch, classes)``; for other outputs (Hugging Face
models, the last position of a sequence) pass ``output_fn`` to the causal
tools. Classes are given as an index, ``"label"`` (the batch targets, see
``innards.extract.default_target_fn``) or ``"pred"`` (the argmax of the
reference output; :func:`change` uses the clean output as reference).

Typical usage::

    values = effect(model, loader, {"blocks.6": ablate(V)}, change(logit("label")))
    mean, lo, hi = bootstrap_ci(values)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import bootstrap
from torch import Tensor

from ..extract import default_target_fn

Target = Callable[..., Tensor]
Comparison = Callable[[Tensor, Tensor, Any], Tensor]
Cls = int | str


def _index(out: Tensor, batch: Any, cls: Cls, ref: Tensor | None) -> Tensor:
    if cls == "label":
        idx = default_target_fn(batch)
        if idx is None:
            raise ValueError("cls='label' needs batches with targets")
        return idx.to(out.device).long()
    if cls == "pred":
        return (out if ref is None else ref).argmax(-1)
    return torch.full(out.shape[:1], int(cls), device=out.device)


def _pick(values: Tensor, idx: Tensor) -> Tensor:
    return values.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def logit(cls: Cls = "pred") -> Target:
    """
    Target: the logit of a class.

    Args:
        cls: Class index, ``"label"`` or ``"pred"``.

    Returns:
        ``fn(out, batch, ref=None) -> (batch,)``.
    """
    return lambda out, batch, ref=None: _pick(out.float(), _index(out, batch, cls, ref))


def log_prob(cls: Cls = "pred") -> Target:
    """
    Target: the log-probability of a class.

    Args:
        cls: Class index, ``"label"`` or ``"pred"``.

    Returns:
        ``fn(out, batch, ref=None) -> (batch,)``.
    """
    return lambda out, batch, ref=None: _pick(out.float().log_softmax(-1), _index(out, batch, cls, ref))


def prob(cls: Cls = "pred") -> Target:
    """
    Target: the probability of a class.

    Args:
        cls: Class index, ``"label"`` or ``"pred"``.

    Returns:
        ``fn(out, batch, ref=None) -> (batch,)``.
    """
    return lambda out, batch, ref=None: _pick(out.float().softmax(-1), _index(out, batch, cls, ref))


def logit_diff(a: Cls, b: Cls) -> Target:
    """
    Target: ``logit[a] - logit[b]``, e.g. correct vs. a contrast class.

    Args:
        a: Class index, ``"label"`` or ``"pred"``.
        b: Class index, ``"label"`` or ``"pred"``.

    Returns:
        ``fn(out, batch, ref=None) -> (batch,)``.
    """
    return lambda out, batch, ref=None: logit(a)(out, batch, ref) - logit(b)(out, batch, ref)


def correct(cls: Cls = "label") -> Target:
    """
    Target: 1 where the prediction equals the class, else 0 (not differentiable).

    Args:
        cls: Class index or ``"label"``.

    Returns:
        ``fn(out, batch, ref=None) -> (batch,)``.
    """
    return lambda out, batch, ref=None: (out.argmax(-1) == _index(out, batch, cls, ref)).float()


def change(target: Target) -> Comparison:
    """
    Comparison: ``target(edited) - target(clean)``.

    ``"pred"`` classes refer to the clean prediction in both terms.

    Args:
        target: A target, e.g. ``logit("label")`` or ``correct()``.

    Returns:
        ``fn(clean, edited, batch) -> (batch,)``.
    """
    return lambda clean, edited, batch: target(edited, batch, ref=clean) - target(clean, batch, ref=clean)


def kl() -> Comparison:
    """
    Comparison: ``KL(clean || edited)`` of the predicted distributions.

    Returns:
        ``fn(clean, edited, batch) -> (batch,)``.
    """
    return lambda clean, edited, batch: F.kl_div(edited.float().log_softmax(-1), clean.float().log_softmax(-1),
                                                 log_target=True, reduction="none").sum(-1)


def flip() -> Comparison:
    """
    Comparison: 1 where the edit changes the top-1 prediction, else 0.

    Returns:
        ``fn(clean, edited, batch) -> (batch,)``.
    """
    return lambda clean, edited, batch: (clean.argmax(-1) != edited.argmax(-1)).float()


def rep_shift(h: Tensor, h_edit: Tensor) -> tuple[Tensor, Tensor]:
    """
    How much an edit moved each example's representation.

    For activations read with ``run(..., read=[layer])`` with and without an
    edit; tokens are flattened per example.

    Args:
        h: Clean representation ``(n, ...)``.
        h_edit: Edited representation ``(n, ...)``.

    Returns:
        ``(relative, cosine)``: ``||h_edit - h|| / ||h||`` and
        ``1 - cos(h, h_edit)``, both ``(n,)``.
    """
    a, b = h.reshape(h.shape[0], -1).double(), h_edit.reshape(h.shape[0], -1).double()
    return (b - a).norm(dim=1) / a.norm(dim=1), 1 - F.cosine_similarity(a, b, dim=1)


def bootstrap_ci(values: Tensor, confidence: float = 0.95, n_resamples: int = 2000,
                 seed: int = 0) -> tuple[float, float, float]:
    """
    Mean with a percentile-bootstrap confidence interval (scipy).

    Args:
        values: Per-example values ``(n,)``, e.g. from ``effect``.
        confidence: Confidence level.
        n_resamples: Bootstrap resamples.
        seed: Random seed.

    Returns:
        ``(mean, low, high)``.
    """
    x = values.detach().double().cpu().numpy().reshape(-1)
    res = bootstrap((x,), np.mean, confidence_level=confidence, n_resamples=n_resamples, method="percentile",
                    random_state=seed)
    return float(x.mean()), float(res.confidence_interval.low), float(res.confidence_interval.high)
