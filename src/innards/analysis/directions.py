"""
Finding directions in a representation.

Every function returns directions ``V (k, d)`` (or one ``(d,)``), so the result
plugs into the geometry tools (``subspace_angles``, ``project``), the causal
tools (``direction_attribution``) and the edits (``ablate(V)``, ``steer(v)``).
An SAE dictionary is a ``V`` too: ``sae.decoder_weight()``.

Typical usage::

    W, b, classes = linear_probe(X_train, y_train)
    probe_accuracy(X_test, y_test, W, b, classes)
    probe_layers(acts, y)                         # held-out accuracy per layer
    V, accs = inlp(X, y, n_iter=10)               # remove a concept linearly
    logits, _ = run(model, images, {"blocks.6": ablate(V)})

References:
    Alain, G. & Bengio, Y. (2016). Understanding intermediate layers using
        linear classifier probes. arXiv:1610.01644.
    Ravfogel, S. et al. (2020). Null It Out: Guarding Protected Attributes by
        Iterative Nullspace Projection. ACL 2020. (INLP.)
    Kim, B. et al. (2018). Interpretability Beyond Feature Attribution:
        Quantitative Testing with Concept Activation Vectors (TCAV). ICML 2018.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from .geometry import remove


def _np(X: Tensor) -> np.ndarray:
    return X.reshape(X.shape[0], -1).detach().double().cpu().numpy()


def _labels(y: Tensor) -> np.ndarray:
    return y.reshape(-1).detach().cpu().numpy()


def class_means(X: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """
    Mean representation of each class.

    Args:
        X: Representation ``(n, d)``.
        y: Labels ``(n,)``.

    Returns:
        ``(means, classes)``: ``(C, d)`` class means and the ``(C,)`` sorted
        class labels they belong to.
    """
    classes = y.unique()
    return torch.stack([X[y == c].double().mean(0) for c in classes]), classes


def diff_of_means(X: Tensor, y: Tensor, positive, negative=None) -> Tensor:
    """
    Difference between the mean of one class and another (or all others).

    The simplest concept direction; the steering vector of activation
    addition when ``X`` are the activations of contrasting inputs.

    Args:
        X: Representation ``(n, d)``.
        y: Labels ``(n,)`` (or a boolean mask, with ``positive=True``).
        positive: Label of the positive class.
        negative: Label of the negative class; ``None`` uses all other examples.

    Returns:
        The direction ``(d,)`` (not normalised).
    """
    pos = y == positive
    neg = ~pos if negative is None else y == negative
    return X[pos].double().mean(0) - X[neg].double().mean(0)


def linear_probe(
    X: Tensor,
    y: Tensor,
    C: float = 1.0,
    max_iter: int = 1000,
    standardize: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Logistic-regression probe (Alain & Bengio, 2016; scikit-learn).

    Features are standardised for the fit (better conditioning) and the
    weights mapped back, so they apply to the raw representation. A binary
    probe's weight is the concept activation vector of TCAV (Kim et al., 2018).

    Args:
        X: Representation ``(n, d)``.
        y: Labels ``(n,)``.
        C: Inverse L2 regularisation strength.
        max_iter: Optimiser iterations.
        standardize: Standardise features for the fit.

    Returns:
        ``(W, b, classes)``: weights ``(k, d)`` and bias ``(k,)`` on the raw
        representation, with ``k = 1`` for two classes (positive class
        ``classes[1]``) and ``k = C`` otherwise; and the ``(C,)`` class labels.
    """
    Z, labels = _np(X), _labels(y)
    scaler = StandardScaler(with_mean=standardize, with_std=standardize).fit(Z)
    clf = LogisticRegression(C=C, max_iter=max_iter).fit(scaler.transform(Z), labels)
    scale = scaler.scale_ if standardize else np.ones(Z.shape[1])
    mean = scaler.mean_ if standardize else np.zeros(Z.shape[1])
    W = clf.coef_ / scale
    b = clf.intercept_ - W @ mean
    return torch.from_numpy(W), torch.from_numpy(b), torch.from_numpy(clf.classes_)


def probe_accuracy(X: Tensor, y: Tensor, W: Tensor, b: Tensor, classes: Tensor) -> float:
    """
    Accuracy of a linear probe, e.g. on held-out data.

    Args:
        X: Representation ``(n, d)``.
        y: Labels ``(n,)``.
        W: Weights ``(k, d)`` from :func:`linear_probe`.
        b: Bias ``(k,)``.
        classes: Class labels ``(C,)``.

    Returns:
        The accuracy.
    """
    logits = X.reshape(X.shape[0], -1).double().cpu() @ W.double().T + b.double()
    pred = (logits[:, 0] > 0).long() if W.shape[0] == 1 else logits.argmax(1)
    return float((classes[pred] == y.reshape(-1).cpu()).double().mean())


def probe_layers(
    acts: Mapping[str, Tensor],
    y: Tensor,
    test_size: float = 0.2,
    seed: int = 0,
    **probe_kwargs,
) -> dict[str, float]:
    """
    Held-out accuracy of a linear probe at every layer.

    Args:
        acts: Layer name -> representation ``(n, ...)`` (e.g. from
            ``load_acts``, pooled); extra dimensions are flattened.
        y: Labels ``(n,)``.
        test_size: Held-out fraction (same split for every layer, stratified).
        seed: Split seed.
        **probe_kwargs: Passed to :func:`linear_probe`.

    Returns:
        Layer name -> held-out accuracy.
    """
    train, test = train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed,
                                   stratify=_labels(y))
    train, test = torch.from_numpy(train), torch.from_numpy(test)
    out = {}
    for name, X in acts.items():
        X = X.reshape(X.shape[0], -1)
        W, b, classes = linear_probe(X[train], y[train], **probe_kwargs)
        out[name] = probe_accuracy(X[test], y[test], W, b, classes)
    return out


def lda(X: Tensor, y: Tensor, k: int | None = None) -> Tensor:
    """
    Linear discriminant directions (scikit-learn, shrinkage covariance).

    Unlike class-mean differences these account for the within-class
    covariance: they separate classes best relative to the noise.

    Args:
        X: Representation ``(n, d)``.
        y: Labels ``(n,)``.
        k: Number of directions, at most ``C - 1``; ``None`` keeps all.

    Returns:
        Unit directions ``(k, d)``, most discriminative first.
    """
    clf = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(_np(X), _labels(y))
    V = torch.from_numpy(clf.scalings_.T[: k or len(clf.classes_) - 1])
    return V / V.norm(dim=1, keepdim=True)


def inlp(
    X: Tensor,
    y: Tensor,
    n_iter: int = 10,
    test_size: float = 0.2,
    seed: int = 0,
    **probe_kwargs,
) -> tuple[Tensor, Tensor]:
    """
    Iterative nullspace projection (Ravfogel et al., 2020).

    Repeatedly fits a linear probe, adds its directions to ``V`` and removes
    ``span(V)`` from the representation, until the concept can no longer be
    decoded linearly. ``ablate(V)`` then erases the concept from the model,
    to test whether the model uses it.

    Args:
        X: Representation ``(n, d)``.
        y: Labels ``(n,)``.
        n_iter: Number of probes.
        test_size: Held-out fraction for the reported accuracies.
        seed: Split seed.
        **probe_kwargs: Passed to :func:`linear_probe`.

    Returns:
        ``(V, accuracies)``: the removed directions (unit, ``(r, d)``) and the
        held-out probe accuracy before each removal ``(n_iter,)``; it should
        fall to chance.
    """
    X = X.reshape(X.shape[0], -1).double().cpu()
    train, test = train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed,
                                   stratify=_labels(y))
    train, test = torch.from_numpy(train), torch.from_numpy(test)
    V = torch.zeros(0, X.shape[1], dtype=torch.float64)
    accs = []
    for _ in range(n_iter):
        Z = remove(X, V) if len(V) else X
        W, b, classes = linear_probe(Z[train], y[train], **probe_kwargs)
        accs.append(probe_accuracy(Z[test], y[test], W, b, classes))
        V = torch.cat([V, W / W.norm(dim=1, keepdim=True)])
    return V, torch.tensor(accs)
