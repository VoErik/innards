"""
Geometry of a representation: dimensionality, spectra, manifolds, class structure.

Representations are ``X (n, d)`` (pooled activations, e.g. from ``extract``
with ``reduce=token(0)``, or tokens flattened to rows); sets of directions are
``V (k, d)`` or a single ``(d,)``. Scalars are returned as floats.

Typical usage::

    V, ratio = pca(X, k=10)
    effective_rank(X), participation_ratio(X), twonn_dim(X)
    nc1, nc2_norm, nc2_angle, nc3, nc4 = neural_collapse(X, y, W=head.weight, b=head.bias)
    subspace_angles(V_probe, V)

References:
    Garrido, Q., Balestriero, R., Najman, L. & LeCun, Y. (2023). RankMe:
        Assessing the Downstream Performance of Pretrained Self-Supervised
        Representations by Their Rank. ICML 2023.
    Gao, P. et al. (2017). A theory of multineuronal dimensionality, dynamics
        and measurement. bioRxiv. (Participation ratio.)
    Stringer, C. et al. (2019). High-dimensional geometry of population
        responses in visual cortex. Nature 571. (Power-law spectra.)
    Agrawal, K. et al. (2022). alpha-ReQ: Assessing Representation Quality in
        Self-Supervised Learning by measuring eigenspectrum decay. NeurIPS 2022.
    Facco, E. et al. (2017). Estimating the intrinsic dimension of datasets by
        a minimal neighborhood information. Scientific Reports 7. (TwoNN.)
    Levina, E. & Bickel, P. (2004). Maximum Likelihood Estimation of Intrinsic
        Dimension. NeurIPS 2004.
    Tenenbaum, J., de Silva, V. & Langford, J. (2000). A Global Geometric
        Framework for Nonlinear Dimensionality Reduction. Science 290. (Isomap
        geodesics.)
    Papyan, V., Han, X. Y. & Donoho, D. (2020). Prevalence of neural collapse
        during the terminal phase of deep learning training. PNAS 117.
    Ethayarajh, K. (2019). How Contextual are Contextualized Word
        Representations? EMNLP 2019. (Anisotropy.)
    Sun, M. et al. (2024). Massive Activations in Large Language Models. COLM
        2024; Darcet, T. et al. (2024). Vision Transformers Need Registers.
        ICLR 2024. (High-norm tokens and outlier dimensions.)
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import torch
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import (
    NearestNeighbors,
    kneighbors_graph
)
from torch import Tensor


def _rows(X: Tensor) -> Tensor:
    return X.detach().reshape(-1, X.shape[-1]).double()


def _2d(V: Tensor) -> Tensor:
    return V.detach().reshape(-1, V.shape[-1]).double()


def orthonormal(V: Tensor) -> Tensor:
    """
    Orthonormal basis of the span of directions.

    Args:
        V: Directions ``(k, d)`` or ``(d,)``.

    Returns:
        ``(r, d)`` orthonormal rows spanning the same subspace (``r <= k``;
        numerically dependent directions are dropped).
    """
    Q, R = torch.linalg.qr(_2d(V).T)
    keep = R.diagonal().abs() > 1e-10 * R.diagonal().abs().max()
    return Q[:, keep].T


def eigenspectrum(X: Tensor) -> Tensor:
    """
    Eigenvalues of the covariance of ``X``, descending.

    Args:
        X: Representation ``(n, d)``.

    Returns:
        ``(min(n, d),)`` eigenvalues.
    """
    X = _rows(X)
    return torch.linalg.svdvals(X - X.mean(0)).square() / max(X.shape[0] - 1, 1)


def pca(X: Tensor, k: int | None = None) -> tuple[Tensor, Tensor]:
    """
    Principal directions.

    Args:
        X: Representation ``(n, d)``.
        k: Number of directions; ``None`` keeps ``min(n, d)``.

    Returns:
        ``(V, ratio)``: directions ``(k, d)`` (unit norm, by decreasing
        variance) and the fraction of variance each explains ``(k,)``.
    """
    X = _rows(X)
    _, S, Vh = torch.linalg.svd(X - X.mean(0), full_matrices=False)
    var = S.square()
    k = len(S) if k is None else k
    return Vh[:k], var[:k] / var.sum()


def effective_rank(X: Tensor, center: bool = False, eps: float = 1e-7) -> float:
    """
    RankMe effective rank (Garrido et al., 2023).

    ``exp`` of the entropy of the normalised singular values: about ``k`` for
    ``k`` equally strong directions. Higher rank of self-supervised embeddings
    predicts better downstream performance.

    Args:
        X: Representation ``(n, d)``.
        center: Center first (RankMe uses the raw embeddings).
        eps: Added to the normalised singular values, as in the paper.

    Returns:
        The effective rank in ``[1, min(n, d)]``.
    """
    X = _rows(X)
    S = torch.linalg.svdvals(X - X.mean(0) if center else X)
    p = S / S.sum() + eps
    return float(torch.exp(-(p * p.log()).sum()))


def participation_ratio(X: Tensor) -> float:
    """
    Participation ratio ``(sum l)^2 / sum l^2`` of the covariance eigenvalues ``l``.

    A linear dimensionality: ``d`` for isotropic data, 1 for one direction.

    Args:
        X: Representation ``(n, d)``.

    Returns:
        The participation ratio.
    """
    lam = eigenspectrum(X)
    return float(lam.sum().square() / lam.square().sum())


def spectrum_decay(X: Tensor, fit: tuple[int, int] | None = None) -> float:
    """
    Power-law exponent ``alpha`` of the eigenspectrum, ``l_i ~ i^-alpha``.

    Stringer et al. (2019) relate ``alpha`` close to 1 to smooth yet high
    dimensional codes; Agrawal et al. (2022) use it to assess
    representations (smaller is higher dimensional).

    Args:
        X: Representation ``(n, d)``.
        fit: Range ``(first, last)`` of eigenvalue indices (1-based) fitted;
            ``None`` uses ``(1, r // 2)`` with ``r`` the number of eigenvalues,
            avoiding the noisy tail.

    Returns:
        ``alpha`` from a least-squares fit in log-log space.
    """
    lam = eigenspectrum(X)
    lo, hi = fit or (1, max(len(lam) // 2, 2))
    idx = torch.arange(lo, hi + 1, dtype=torch.float64)
    y = lam[lo - 1 : hi].clamp_min(1e-300).log()
    slope = np.polyfit(idx.log().numpy(), y.cpu().numpy(), 1)[0]
    return float(-slope)


def _neighbour_distances(X: Tensor, k: int) -> np.ndarray:
    Z = _rows(X).cpu().numpy()
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    return dist[:, 1:]


def twonn_dim(X: Tensor, discard: float = 0.1) -> float:
    """
    TwoNN intrinsic dimension (Facco et al., 2017).

    Uses only the ratio of each point's second to first nearest-neighbour
    distance, so it is robust to density changes and curvature.

    Args:
        X: Representation ``(n, d)``; duplicate points are ignored.
        discard: Fraction of the largest ratios dropped from the fit, as in
            the paper.

    Returns:
        The intrinsic dimension.
    """
    dist = _neighbour_distances(X, 2)
    dist = dist[dist[:, 0] > 0]
    mu = np.sort(dist[:, 1] / dist[:, 0])
    n = len(mu)
    keep = int(n * (1 - discard))
    x = np.log(mu[:keep])
    y = -np.log(1 - np.arange(1, keep + 1) / n)
    return float((x @ y) / (x @ x))  # least squares through the origin


def local_dim(X: Tensor, k: int = 20) -> Tensor:
    """
    Local intrinsic dimension of every point (Levina & Bickel, 2004).

    Maximum-likelihood estimate from the distances to the ``k`` nearest
    neighbours. The mean over points is a global estimate.

    Args:
        X: Representation ``(n, d)``.
        k: Neighbours per estimate.

    Returns:
        ``(n,)`` local dimensions.
    """
    dist = _neighbour_distances(X, k).clip(min=1e-300)
    logs = np.log(dist[:, -1:] / dist[:, :-1]).sum(1)
    return torch.from_numpy((k - 1) / np.maximum(logs, 1e-300))


def geodesic_distances(X: Tensor, k: int = 10) -> Tensor:
    """
    Geodesic distances along the ``k``-nearest-neighbour graph (Isomap).

    Compared with Euclidean distances (``torch.cdist(X, X)``) they show how
    curved the data manifold is: on a flat manifold both agree, on a curved
    one geodesics are longer.

    Args:
        X: Representation ``(n, d)``.
        k: Neighbours per point in the graph.

    Returns:
        ``(n, n)`` distances; ``inf`` between disconnected components.
    """
    graph = kneighbors_graph(_rows(X).cpu().numpy(), k, mode="distance")
    return torch.from_numpy(shortest_path(graph, directed=False))


def _class_stats(X: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    X, y = _rows(X), y.reshape(-1).cpu()
    classes = y.unique()
    means = torch.stack([X[y == c].mean(0) for c in classes])
    mu = X.mean(0)
    within = sum((X[y == c] - m).T @ (X[y == c] - m) for c, m in zip(classes, means)) / X.shape[0]
    between = (means - mu).T @ (means - mu) / len(classes)
    return classes, means, within, between


def fisher_ratio(X: Tensor, y: Tensor) -> float:
    """
    Between- over within-class scatter, ``tr(S_B) / tr(S_W)``.

    Args:
        X: Representation ``(n, d)``.
        y: Labels ``(n,)``.

    Returns:
        The ratio; larger means better separated classes.
    """
    _, _, within, between = _class_stats(X, y)
    return float(between.trace() / within.trace())


def neural_collapse(
    X: Tensor,
    y: Tensor,
    W: Tensor | None = None,
    b: Tensor | None = None,
) -> tuple[float, float, float, float, float]:
    """
    Neural-collapse statistics NC1-NC4 (Papyan et al., 2020).

    All are 0 under full collapse: features collapse to their class means,
    which form a simplex equiangular tight frame aligned with the classifier,
    which then acts as a nearest-class-center rule.

    Args:
        X: Penultimate features ``(n, d)``.
        y: Labels ``(n,)``, values ``0 .. C-1`` matching the rows of ``W``.
        W: Classifier weight ``(C, d)``, for NC3 and NC4.
        b: Classifier bias ``(C,)``, for NC4.

    Returns:
        ``(nc1, nc2_norms, nc2_angles, nc3, nc4)``:

        * ``nc1``: within-class variability, ``tr(S_W S_B^+) / C``.
        * ``nc2_norms``: coefficient of variation of the centered class-mean norms.
        * ``nc2_angles``: mean ``|cos + 1 / (C - 1)|`` between centered class
          means (0 for a simplex ETF).
        * ``nc3``: ``|| W / ||W|| - M / ||M|| ||_F^2`` between the classifier
          and the centered class means (``nan`` without ``W``).
        * ``nc4``: fraction of examples where the classifier and the
          nearest-class-center rule disagree (``nan`` without ``W``).
    """
    classes, means, within, between = _class_stats(X, y)
    C = len(classes)
    nc1 = float((within @ torch.linalg.pinv(between, hermitian=True)).trace() / C)
    M = means - _rows(X).mean(0)
    norms = M.norm(dim=1)
    nc2_norms = float(norms.std() / norms.mean())
    cos = (M / norms[:, None]) @ (M / norms[:, None]).T
    off = ~torch.eye(C, dtype=torch.bool)
    nc2_angles = float((cos[off] + 1 / (C - 1)).abs().mean())
    nc3 = nc4 = float("nan")
    if W is not None:
        W = W.detach().double().cpu()
        nc3 = float((W / W.norm() - M / M.norm()).square().sum())
        logits = _rows(X) @ W.T + (b.detach().double().cpu() if b is not None else 0)
        ncc = torch.cdist(_rows(X), means).argmin(1)
        nc4 = float((logits.argmax(1) != ncc).double().mean())
    return nc1, nc2_norms, nc2_angles, nc3, nc4


def anisotropy(X: Tensor) -> float:
    """
    Mean cosine similarity between different examples (Ethayarajh, 2019).

    Near 0 for isotropic representations, near 1 when all examples point the
    same way (e.g. a dominant mean direction).

    Args:
        X: Representation ``(n, d)``.

    Returns:
        The mean pairwise cosine, computed exactly.
    """
    U = _rows(X)
    U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n = U.shape[0]
    return float((U.sum(0).square().sum() - n) / (n * (n - 1)))


def outlier_dims(X: Tensor, z: float = 6.0) -> tuple[Tensor, Tensor]:
    """
    Feature dimensions with outsized mean magnitude ("massive activations").

    A dimension is an outlier if its mean absolute activation lies more than
    ``z`` robust standard deviations (1.4826 times the median absolute
    deviation) above the median over dimensions. Robust statistics are
    needed: a few huge dimensions would inflate a plain standard deviation
    enough to hide themselves.

    Args:
        X: Representation ``(n, d)`` or tokens ``(n, t, d)``.
        z: Threshold in robust standard deviations.

    Returns:
        ``(dims, scores)``: outlier dimension indices, and the mean absolute
        activation of every dimension ``(d,)``.
    """
    scores = _rows(X).abs().mean(0)
    med = scores.median()
    spread = 1.4826 * (scores - med).abs().median()
    return (scores > med + z * spread).nonzero().squeeze(-1), scores


def high_norm_tokens(tokens: Tensor, factor: float = 10.0) -> Tensor:
    """
    Tokens with outsized norm, e.g. the artifact tokens of ViTs without registers.

    Args:
        tokens: Tokens ``(n, t, d)``.
        factor: A token is flagged if its norm exceeds ``factor`` times the
            median token norm.

    Returns:
        Boolean mask ``(n, t)``.
    """
    norms = tokens.detach().double().norm(dim=-1)
    return norms > factor * norms.median()


def cosine_matrix(V: Tensor, W: Tensor | None = None) -> Tensor:
    """
    Cosine similarities between directions.

    Args:
        V: Directions ``(k, d)`` or ``(d,)``.
        W: Directions ``(m, d)``; ``None`` compares ``V`` with itself.

    Returns:
        ``(k, m)`` cosines.
    """
    V = _2d(V)
    W = V if W is None else _2d(W)
    return (V / V.norm(dim=1, keepdim=True)) @ (W / W.norm(dim=1, keepdim=True)).T


def subspace_angles(U: Tensor, V: Tensor) -> Tensor:
    """
    Principal angles between the spans of two sets of directions.

    Args:
        U: Directions ``(k, d)`` or ``(d,)``.
        V: Directions ``(m, d)`` or ``(d,)``.

    Returns:
        ``(min(k, m),)`` angles in radians, ascending: 0 for shared
        directions, pi / 2 for orthogonal ones.
    """
    angles = scipy.linalg.subspace_angles(_2d(U).T.cpu().numpy(), _2d(V).T.cpu().numpy())
    return torch.from_numpy(np.sort(angles))


def project(X: Tensor, V: Tensor) -> Tensor:
    """
    Component of ``X`` in the span of ``V``.

    Args:
        X: Representation ``(..., d)``.
        V: Directions ``(k, d)`` or ``(d,)``.

    Returns:
        The projection, shaped like ``X`` (in float64).
    """
    Q = orthonormal(V).to(X.device)
    return (X.detach().double() @ Q.T) @ Q


def remove(X: Tensor, V: Tensor) -> Tensor:
    """
    Component of ``X`` orthogonal to the span of ``V`` (``X - project(X, V)``).

    The same operation as :func:`innards.edits.ablate`, on stored activations.

    Args:
        X: Representation ``(..., d)``.
        V: Directions ``(k, d)`` or ``(d,)``.

    Returns:
        The residual, shaped like ``X`` (in float64).
    """
    return X.detach().double() - project(X, V)
