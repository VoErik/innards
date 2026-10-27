"""Tests for innards.analysis.metrics and innards.analysis.causal."""

import pytest
import torch
from torch import nn

timm = pytest.importorskip("timm")

from innards.analysis import causal as C
from innards.analysis import geometry as G
from innards.analysis import metrics as M
from innards.edits import (
    ablate,
    run,
    steer
)


class LinearAfterMid(nn.Module):
    """Nonlinear up to ``mid``, linear after it, so first-order estimates at ``mid`` are exact."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed, self.act, self.mid, self.head = nn.Linear(4, 8), nn.ReLU(), nn.Identity(), nn.Linear(8, 3)

    def forward(self, x):
        return self.head(self.mid(self.act(self.embed(x))).mean(1))


@pytest.fixture(scope="module")
def vit():
    torch.manual_seed(0)
    return timm.create_model("test_vit", pretrained=False).eval()


def images(n, seed=0):
    return torch.randn(n, 3, 160, 160, generator=torch.Generator().manual_seed(seed))


# ---------- metrics ----------


def test_metrics():
    clean = torch.tensor([[2.0, 0.0, -1.0], [0.0, 3.0, 1.0]])
    edited = torch.tensor([[0.0, 1.0, -1.0], [0.0, 2.0, 1.0]])
    batch = (None, torch.tensor([0, 2]))
    assert M.logit("label")(clean, batch).tolist() == [2.0, 1.0]
    assert M.logit(1)(clean, batch).tolist() == [0.0, 3.0]
    assert M.logit()(clean, batch).tolist() == [2.0, 3.0]                       # "pred"
    assert M.change(M.logit())(clean, edited, batch).tolist() == [-2.0, -1.0]   # clean's prediction in both
    assert M.logit_diff(0, 1)(clean, batch).tolist() == [2.0, -3.0]
    assert M.correct()(clean, batch).tolist() == [1.0, 0.0]
    assert M.flip()(clean, edited, batch).tolist() == [1.0, 0.0]
    torch.testing.assert_close(M.prob(1)(clean, batch), clean.softmax(-1)[:, 1])
    torch.testing.assert_close(M.log_prob("label")(clean, batch), clean.log_softmax(-1)[[0, 1], [0, 2]])
    assert torch.allclose(M.kl()(clean, clean, batch), torch.zeros(2), atol=1e-6)
    expected = (clean.softmax(-1) * (clean.log_softmax(-1) - edited.log_softmax(-1))).sum(-1)
    torch.testing.assert_close(M.kl()(clean, edited, batch), expected)
    with pytest.raises(ValueError):
        M.logit("label")(clean, clean)                                       # no targets in the batch


def test_rep_shift_and_bootstrap():
    h = torch.randn(5, 3, 4)
    rel, cos = M.rep_shift(h, 2 * h)
    assert torch.allclose(rel, torch.ones(5, dtype=rel.dtype)) and torch.allclose(cos, torch.zeros(5, dtype=cos.dtype),
                                                                                  atol=1e-12)
    values = torch.randn(500, generator=torch.Generator().manual_seed(0)) + 2
    mean, lo, hi = M.bootstrap_ci(values)
    assert lo < mean < hi and lo < 2.0 < hi


# ---------- effects ----------


def test_effect_and_sweep(vit):
    x, y = images(6), torch.arange(6) % 3
    loader = [(x[:3], y[:3]), (x[3:], y[3:])]
    none = C.effect(vit, loader, {}, M.kl())
    assert none.shape == (6,) and none.abs().max() < 1e-6
    v = torch.randn(64, generator=torch.Generator().manual_seed(1))
    swept = C.sweep(vit, loader, lambda a: {"blocks.3": steer(v, a)}, [0.0, 1.0, 4.0], M.kl())
    assert swept.shape == (6, 3) and swept[:, 0].abs().max() < 1e-6
    assert swept[:, 1].mean() < swept[:, 2].mean()
    per_batch = C.effect(vit, loader, lambda batch: {"blocks.3": steer(v, 4.0)}, M.kl())  # edits from the batch
    torch.testing.assert_close(per_batch, swept[:, 2])
    acc = C.effect(vit, loader, {"blocks.3": ablate(v)}, M.change(M.correct()))
    assert set(acc.unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_patch_sweep_restores(vit):
    clean, corrupt = (images(2, 1), torch.tensor([0, 1])), (images(2, 2), torch.tensor([0, 1]))
    metric = M.change(M.logit("label"))
    full = C.patch_sweep(vit, clean, corrupt, ["blocks.1", "blocks.4"], metric)
    with torch.no_grad():
        target = M.logit("label")(vit(clean[0]), clean) - M.logit("label")(vit(corrupt[0]), clean)
    for name in ("blocks.1", "blocks.4"):                                # patching every token restores fully
        torch.testing.assert_close(full[name], target, atol=1e-4, rtol=1e-4)
    per_pos = C.patch_sweep(vit, clean, corrupt, "blocks.4", metric, positions=True)
    assert per_pos["blocks.4"].shape == (2, 101)


# ---------- gradient-based, exact on a model linear after the layer ----------


def test_gradients_match_autograd():
    net, x = LinearAfterMid().eval(), torch.randn(2, 5, 4)
    grads, acts, values = C.gradients(net, x, ["mid", "act"], M.logit(1))
    assert all(p.grad is None for p in net.parameters())                 # parameter gradients untouched
    h = net.act(net.embed(x)).detach().requires_grad_(True)
    net.head(h.mean(1))[:, 1].sum().backward()
    torch.testing.assert_close(grads["mid"], h.grad)
    torch.testing.assert_close(acts["mid"], h.detach())
    assert grads["act"].abs().sum() > 0                                  # earlier layer gets gradient too


def test_attribution_patching_is_exact_when_linear():
    net = LinearAfterMid().eval()
    clean, corrupt = torch.randn(3, 5, 4), torch.randn(3, 5, 4)
    estimate = C.attribution_patching(net, clean, corrupt, "mid", M.logit(0))["mid"]
    actual = C.patch_sweep(net, clean, corrupt, "mid", M.change(M.logit(0)), positions=True)["mid"]
    torch.testing.assert_close(estimate.float(), actual, atol=1e-5, rtol=1e-4)


def test_direction_attribution_is_exact_when_linear():
    net, x = LinearAfterMid().eval(), torch.randn(4, 5, 4)
    V = torch.randn(3, 8, generator=torch.Generator().manual_seed(2))
    attr = C.direction_attribution(net, [x[:2], x[2:]], "mid", V, M.logit(2))
    assert attr.shape == (4, 3)
    with torch.no_grad():
        clean = net(x)[:, 2]
        for i in range(3):
            ablated = run(net, x, {"mid": ablate(V[i])})[0][:, 2]
            torch.testing.assert_close(attr[:, i].float(), clean - ablated, atol=1e-5, rtol=1e-4)


def test_relevant_subspace_is_the_readout():
    net = LinearAfterMid().eval()
    x, y = torch.randn(30, 5, 4), torch.arange(30) % 3
    V, energy = C.relevant_subspace(net, [(x, y)], "mid", M.logit("label"), k=5)
    assert V.shape == (5, 8) and energy[:3].sum() == pytest.approx(1.0, abs=1e-9)
    assert G.subspace_angles(V[:3], net.head.weight).max() < 1e-5      # gradients live in the head's row space
