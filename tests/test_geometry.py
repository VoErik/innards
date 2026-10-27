"""Known-answer tests for innards.analysis.similarity / geometry / directions on synthetic data."""

import math

import pytest
import torch

from innards.analysis import directions as D
from innards.analysis import geometry as G
from innards.analysis import similarity as S


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def rotation(d, seed=0):
    return torch.linalg.qr(torch.randn(d, d, generator=gen(seed), dtype=torch.float64)).Q


def low_rank(n=500, d=20, r=3, noise=0.0, seed=0):
    Z = torch.randn(n, r, generator=gen(seed), dtype=torch.float64)
    B = torch.randn(r, d, generator=gen(seed + 1), dtype=torch.float64)
    return Z @ B + noise * torch.randn(n, d, generator=gen(seed + 2), dtype=torch.float64), B


# ---------- similarity ----------


def test_cka_invariances():
    X = torch.randn(200, 30, generator=gen(), dtype=torch.float64)
    Y = 3.0 * X @ rotation(30) + 5.0
    assert S.cka(X, Y) == pytest.approx(1.0, abs=1e-9)
    assert S.cka(X, Y, debiased=True) == pytest.approx(1.0, abs=1e-9)
    noise = torch.randn(200, 30, generator=gen(1), dtype=torch.float64)
    assert S.cka(X, noise, debiased=True) == pytest.approx(0.0, abs=0.05)
    assert S.cka(X, noise) > S.cka(X, noise, debiased=True)          # the biased estimate is inflated


def test_cka_feature_and_gram_forms_agree():
    X = torch.randn(10, 40, generator=gen(), dtype=torch.float64)    # d1 * d2 > n^2: Gram form
    Y = torch.randn(10, 3, generator=gen(1), dtype=torch.float64)    # mixed
    Z = torch.randn(10, 5, generator=gen(2), dtype=torch.float64)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    K, L = Xc @ Xc.T, Yc @ Yc.T
    expected = float((K * L).sum() / (K.norm() * L.norm()))
    assert S.cka(X, Y) == pytest.approx(expected, rel=1e-9)
    assert S.cka(Y, Z) == pytest.approx(S.cka(Z, Y), rel=1e-9)


def test_cka_matrix_and_tokens():
    X = torch.randn(50, 4, 8, generator=gen(), dtype=torch.float64)  # tokens are flattened
    M = S.cka_matrix({"a": X, "b": 2 * X, "c": torch.randn(50, 6, generator=gen(1))})
    assert M.shape == (3, 3) and torch.allclose(M, M.T)
    assert M[0, 1] == pytest.approx(1.0) and torch.allclose(M.diagonal(), torch.ones(3, dtype=M.dtype))


def test_rsa_procrustes_svcca_knn():
    X = torch.randn(80, 10, generator=gen(), dtype=torch.float64)
    Y = 2 * X @ rotation(10) + 1
    assert S.rsa(X, Y, metric="euclidean") == pytest.approx(1.0)
    assert S.rdm(X).shape == (80, 80)
    assert S.procrustes(X, Y) == pytest.approx(0.0, abs=1e-6)
    padded = torch.cat([X, torch.zeros(80, 5, dtype=X.dtype)], 1)    # different widths
    assert S.procrustes(X, padded) == pytest.approx(0.0, abs=1e-6)
    other = torch.randn(80, 10, generator=gen(3), dtype=torch.float64)
    assert S.procrustes(X, other) > 0.5
    mean, corr = S.svcca(X, X @ torch.randn(10, 10, generator=gen(4), dtype=torch.float64), keep=10)
    assert mean == pytest.approx(1.0, abs=1e-6) and corr.shape == (10,)
    assert S.knn_overlap(X, Y, k=5) == pytest.approx(1.0)
    assert S.knn_overlap(X, other, k=5) < 0.3


# ---------- geometry ----------


def test_pca_and_dimensionality():
    X, B = low_rank(r=3)
    V, ratio = G.pca(X, k=5)
    assert V.shape == (5, 20) and ratio[:3].sum() == pytest.approx(1.0)
    assert G.subspace_angles(V[:3], B).max() < 1e-6                  # recovers the planted subspace
    assert G.effective_rank(X, center=True) < 3.01
    iso = torch.randn(20000, 10, generator=gen(), dtype=torch.float64)
    assert G.participation_ratio(iso) == pytest.approx(10, rel=0.02)
    assert G.effective_rank(iso, center=True) == pytest.approx(10, rel=0.02)
    assert G.eigenspectrum(iso).shape == (10,)


def test_spectrum_decay():
    d, alpha = 200, 1.3
    scales = torch.arange(1, d + 1, dtype=torch.float64) ** (-alpha / 2)
    X = torch.randn(5000, d, generator=gen(), dtype=torch.float64) @ rotation(d).T * scales  # eigenvalues ~ i^-alpha
    assert G.spectrum_decay(X @ rotation(d, 1), fit=(1, 50)) == pytest.approx(alpha, abs=0.1)


def test_intrinsic_dimension():
    plane = torch.rand(2000, 2, generator=gen(), dtype=torch.float64) @ torch.randn(2, 12, generator=gen(1),
                                                                                     dtype=torch.float64)
    assert G.twonn_dim(plane) == pytest.approx(2.0, abs=0.25)
    assert G.local_dim(plane, k=20).median() == pytest.approx(2.0, abs=0.3)
    cube = torch.rand(4000, 5, generator=gen(2), dtype=torch.float64)
    assert G.twonn_dim(cube) == pytest.approx(5.0, abs=0.6)


def test_geodesic_distances_on_a_circle():
    t = torch.linspace(0, math.pi, 200, dtype=torch.float64)
    X = torch.stack([t.cos(), t.sin()], 1)                           # half circle, radius 1
    geo = G.geodesic_distances(X, k=4)
    assert geo[0, -1] == pytest.approx(math.pi, rel=0.01)            # arc length, not the chord (2)


def simplex_data(C=4, d=8, n_per=50, noise=0.0, seed=0):
    M = torch.eye(C, dtype=torch.float64) - 1 / C                    # centered simplex ETF in R^C
    M = torch.cat([M, torch.zeros(C, d - C, dtype=torch.float64)], 1) @ rotation(d, seed)
    y = torch.arange(C).repeat_interleave(n_per)
    return M[y] + noise * torch.randn(len(y), d, generator=gen(seed), dtype=torch.float64), y, M


def test_neural_collapse():
    X, y, M = simplex_data()
    nc1, nc2n, nc2a, nc3, nc4 = G.neural_collapse(X, y, W=M, b=torch.zeros(4))
    assert nc1 == pytest.approx(0, abs=1e-12) and nc2n == pytest.approx(0, abs=1e-9)
    assert nc2a == pytest.approx(0, abs=1e-9) and nc3 == pytest.approx(0, abs=1e-9) and nc4 == 0
    Xn, yn, _ = simplex_data(noise=0.3)
    assert G.neural_collapse(Xn, yn)[0] > 0.01
    assert math.isnan(G.neural_collapse(Xn, yn)[3])
    assert G.fisher_ratio(X + 0.1 * torch.randn_like(X), y) > G.fisher_ratio(Xn, yn)


def test_anisotropy_outliers_norms():
    iso = torch.randn(2000, 32, generator=gen(), dtype=torch.float64)
    assert G.anisotropy(iso) == pytest.approx(0.0, abs=0.01)
    assert G.anisotropy(iso + 50) > 0.95
    spiky = iso.clone()
    spiky[:, 7] *= 100
    dims, scores = G.outlier_dims(spiky)
    assert dims.tolist() == [7] and scores.shape == (32,)
    tokens = torch.randn(4, 10, 8, generator=gen())
    tokens[2, 3] *= 100
    mask = G.high_norm_tokens(tokens)
    assert mask.sum() == 1 and mask[2, 3]


def test_directions_algebra():
    U = torch.randn(3, 10, generator=gen(), dtype=torch.float64)
    assert G.subspace_angles(U, torch.randn(3, 3, generator=gen(1), dtype=torch.float64) @ U).max() < 1e-6
    e = torch.eye(10, dtype=torch.float64)
    assert G.subspace_angles(e[:2], e[2:4]).min() == pytest.approx(math.pi / 2)
    X = torch.randn(30, 10, generator=gen(2), dtype=torch.float64)
    assert torch.allclose(G.project(X, U) + G.remove(X, U), X)
    assert torch.allclose(G.remove(X, U) @ U.T, torch.zeros(30, 3, dtype=torch.float64), atol=1e-10)
    assert G.orthonormal(torch.stack([U[0], U[0] * 2, U[1]])).shape == (2, 10)   # dependent rows dropped
    C = G.cosine_matrix(U)
    assert torch.allclose(C.diagonal(), torch.ones(3, dtype=C.dtype))


# ---------- directions ----------


def concept_data(n=600, d=16, seed=0):
    v = torch.zeros(d, dtype=torch.float64)
    v[3] = 1.0
    v = rotation(d, seed) @ v                                        # planted concept direction
    y = torch.arange(n) % 2
    X = torch.randn(n, d, generator=gen(seed), dtype=torch.float64) + 3.0 * (2 * y[:, None] - 1) * v
    return X, y, v


def test_means_and_probe():
    X, y, v = concept_data()
    means, classes = D.class_means(X, y)
    assert classes.tolist() == [0, 1] and means.shape == (2, 16)
    diff = D.diff_of_means(X, y, positive=1, negative=0)
    assert G.cosine_matrix(diff, v).item() > 0.98
    assert torch.allclose(diff, D.diff_of_means(X, y, positive=1))
    W, b, classes = D.linear_probe(X, y)
    assert W.shape == (1, 16) and G.cosine_matrix(W, v).item() > 0.95
    assert D.probe_accuracy(X, y, W, b, classes) > 0.95
    W3, b3, c3 = D.linear_probe(X, y + 5 * (torch.arange(len(y)) % 3 == 0), standardize=False)
    assert W3.shape[0] == len(c3) == 4                                # multiclass labels 0, 1, 5, 6


def test_probe_layers_and_lda():
    X, y, v = concept_data()
    acc = D.probe_layers({"signal": X, "noise": torch.randn(len(y), 16, generator=gen(5))}, y)
    assert acc["signal"] > 0.95 and acc["noise"] < 0.7
    V = D.lda(X, y)
    assert V.shape == (1, 16) and abs(G.cosine_matrix(V, v).item()) > 0.95


def test_inlp_erases_the_concept():
    X, y, v = concept_data()
    V, accs = D.inlp(X, y, n_iter=4)
    assert V.shape == (4, 16)
    assert accs[0] > 0.95 and accs[-1] < 0.65
    assert D.probe_layers({"erased": G.remove(X, V)}, y)["erased"] < 0.65
