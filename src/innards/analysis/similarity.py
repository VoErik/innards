"""
Comparing representations: of two layers, two models, or before and after an edit.

Every function takes two representations of the same ``n`` examples,
``X (n, ...)`` and ``Y (n, ...)``; extra dimensions (e.g. tokens) are
flattened into features, and the feature counts may differ. Scalars are
returned as floats.

Typical usage::

    acts, _ = load_acts("acts/run")
    cka(acts["blocks.3"], acts["blocks.9"])
    cka_matrix(acts)                         # all layer pairs
    knn_overlap(acts_model_a["head"], acts_model_b["head"], k=10)

References:
    Kornblith, S. et al. (2019). Similarity of Neural Network Representations
        Revisited. ICML 2019. (Linear CKA.)
    Song, L. et al. (2012). Feature Selection via Dependence Maximization.
        JMLR 13. (Unbiased HSIC, for debiased CKA.)
    Kriegeskorte, N., Mur, M. & Bandettini, P. (2008). Representational
        Similarity Analysis. Frontiers in Systems Neuroscience 2.
    Williams, A. et al. (2021). Generalized Shape Metrics on Neural
        Representations. NeurIPS 2021. (Angular Procrustes distance.)
    Raghu, M. et al. (2017). SVCCA: Singular Vector Canonical Correlation
        Analysis for Deep Learning Dynamics and Interpretability. NeurIPS 2017.
    Huh, M. et al. (2024). The Platonic Representation Hypothesis. ICML 2024.
        (Mutual nearest-neighbour alignment.)
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence
)

import numpy as np
import torch
from scipy.spatial.distance import (
    pdist,
    squareform
)
from scipy.stats import (
    pearsonr,
    spearmanr
)
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


def _flat(X: Tensor) -> Tensor:
    return X.detach().reshape(X.shape[0], -1).double()


def _centered(X: Tensor) -> Tensor:
    X = _flat(X)
    return X - X.mean(0)


def _hsic_unbiased(K: Tensor, L: Tensor) -> Tensor:
    n = K.shape[0]
    K, L = K.clone(), L.clone()
    K.fill_diagonal_(0)
    L.fill_diagonal_(0)
    ones = torch.ones(n, dtype=K.dtype, device=K.device)
    return ((K * L).sum() + K.sum() * L.sum() / ((n - 1) * (n - 2))
            - 2 / (n - 2) * (ones @ K) @ (L @ ones)) / (n * (n - 3))


def cka(X: Tensor, Y: Tensor, debiased: bool = False) -> float:
    """
    Linear centered kernel alignment (Kornblith et al., 2019).

    Invariant to rotations and isotropic scaling of either representation;
    1 for identical representations up to those. The biased estimator is
    inflated for few examples and many features; ``debiased=True`` uses the
    unbiased HSIC (Song et al., 2012), which is close to 0 for unrelated
    representations.

    Args:
        X: Representation ``(n, ...)``.
        Y: Representation ``(n, ...)`` of the same examples.
        debiased: Use the unbiased estimator (needs ``n > 3``; builds
            ``(n, n)`` Gram matrices).

    Returns:
        CKA in ``[0, 1]`` (the debiased estimate can be slightly negative).
    """
    X, Y = _centered(X), _centered(Y)
    n = X.shape[0]
    if debiased:
        K, L = X @ X.T, Y @ Y.T
        return float(_hsic_unbiased(K, L) / (_hsic_unbiased(K, K) * _hsic_unbiased(L, L)).sqrt())
    if X.shape[1] * Y.shape[1] <= n * n:  # feature space is smaller than the Gram matrices
        num = (Y.T @ X).square().sum()
        den = (X.T @ X).square().sum().sqrt() * (Y.T @ Y).square().sum().sqrt()
    else:
        K, L = X @ X.T, Y @ Y.T
        num = (K * L).sum()
        den = K.square().sum().sqrt() * L.square().sum().sqrt()
    return float(num / den)


def cka_matrix(acts: Mapping[str, Tensor] | Sequence[Tensor], debiased: bool = False) -> Tensor:
    """
    CKA between every pair of representations, e.g. all layers of a run.

    Args:
        acts: Layer name -> representation (e.g. from ``load_acts``), or a
            list of representations, all of the same ``n`` examples.
        debiased: Use the unbiased estimator, see :func:`cka`.

    Returns:
        Symmetric ``(L, L)`` matrix, in the order of ``acts``.
    """
    reps = list(acts.values()) if isinstance(acts, Mapping) else list(acts)
    out = torch.eye(len(reps), dtype=torch.float64)
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            out[i, j] = out[j, i] = cka(reps[i], reps[j], debiased)
    return out


def rdm(X: Tensor, metric: str = "correlation") -> Tensor:
    """
    Representational dissimilarity matrix (Kriegeskorte et al., 2008).

    Args:
        X: Representation ``(n, ...)``.
        metric: Any :func:`scipy.spatial.distance.pdist` metric, e.g.
            ``"correlation"`` (1 - Pearson r), ``"cosine"``, ``"euclidean"``.

    Returns:
        Symmetric ``(n, n)`` dissimilarities.
    """
    return torch.from_numpy(squareform(pdist(_flat(X).cpu().numpy(), metric)))


def rsa(X: Tensor, Y: Tensor, metric: str = "correlation", method: str = "spearman") -> float:
    """
    Representational similarity: correlation of the two RDMs' upper triangles.

    Args:
        X: Representation ``(n, ...)``.
        Y: Representation ``(n, ...)`` of the same examples.
        metric: Dissimilarity, see :func:`rdm`.
        method: ``"spearman"`` (rank, the usual choice) or ``"pearson"``.

    Returns:
        The correlation in ``[-1, 1]``.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
    a, b = (pdist(_flat(Z).cpu().numpy(), metric) for Z in (X, Y))
    return float((spearmanr if method == "spearman" else pearsonr)(a, b)[0])


def procrustes(X: Tensor, Y: Tensor) -> float:
    """
    Angular Procrustes distance (Williams et al., 2021).

    Both representations are centered and scaled to unit Frobenius norm (the
    narrower one is zero-padded); the distance is the angle between them after
    the best rotation. It is a metric: 0 for representations equal up to
    rotation, translation and scale, at most pi / 2.

    Args:
        X: Representation ``(n, ...)``.
        Y: Representation ``(n, ...)`` of the same examples.

    Returns:
        The distance in radians.
    """
    X, Y = _centered(X), _centered(Y)
    X, Y = X / X.norm(), Y / Y.norm()
    nuclear = torch.linalg.svdvals(X.T @ Y).sum().clamp(-1.0, 1.0)  # padding does not change X^T Y's singular values
    return float(torch.arccos(nuclear))


def _top_directions(X: Tensor, keep: float | int) -> Tensor:
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    if isinstance(keep, float):
        var = S.square().cumsum(0) / S.square().sum()
        keep = int((var < keep).sum()) + 1
    return U[:, :keep] * S[:keep]


def svcca(X: Tensor, Y: Tensor, keep: float | int = 0.99) -> tuple[float, Tensor]:
    """
    SVCCA (Raghu et al., 2017): CCA between the top singular directions of each.

    Args:
        X: Representation ``(n, ...)``.
        Y: Representation ``(n, ...)`` of the same examples.
        keep: Directions kept per representation: a fraction of variance
            (float) or a count (int).

    Returns:
        ``(mean, correlations)``: the mean canonical correlation and all
        canonical correlations, descending.
    """
    X, Y = _top_directions(_centered(X), keep), _top_directions(_centered(Y), keep)
    qx, qy = torch.linalg.qr(X).Q, torch.linalg.qr(Y).Q
    corr = torch.linalg.svdvals(qx.T @ qy).clamp(max=1.0)
    return float(corr.mean()), corr


def knn_overlap(X: Tensor, Y: Tensor, k: int = 10) -> float:
    """
    Mutual nearest-neighbour alignment (Huh et al., 2024).

    The mean fraction of each example's ``k`` nearest neighbours under ``X``
    that are also among its ``k`` nearest neighbours under ``Y``: local
    agreement, where CKA measures global agreement.

    Args:
        X: Representation ``(n, ...)``.
        Y: Representation ``(n, ...)`` of the same examples.
        k: Number of neighbours.

    Returns:
        The overlap in ``[0, 1]``.
    """

    def neighbours(Z: Tensor) -> np.ndarray:
        Z = _flat(Z).cpu().numpy()
        return NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]

    a, b = neighbours(X), neighbours(Y)
    return float(np.mean([len(set(r) & set(s)) / k for r, s in zip(a, b)]))
