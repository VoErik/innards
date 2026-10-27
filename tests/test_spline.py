"""Tests for innards.analysis.spline, with exact answers on small ReLU networks."""

import pytest
import torch
from torch import nn

timm = pytest.importorskip("timm")

from innards.analysis import spline as SP


class Planar(nn.Module):
    """One ReLU layer on 2-d inputs: the partition is an arrangement of H lines."""

    def __init__(self, hidden=12):
        super().__init__()
        torch.manual_seed(0)
        self.fc, self.act, self.out = nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 2)

    def forward(self, x):
        return self.out(self.act(self.fc(x)))

    def signs(self, x):
        return (self.fc(x) > 0).detach()


@pytest.fixture(scope="module")
def resnet():
    torch.manual_seed(0)
    return timm.create_model("test_resnet", pretrained=False).eval()


def test_activation_layers(resnet):
    layers = SP.activation_layers(resnet)
    assert "act1" in layers and any(n.endswith("act2") for n in layers)
    assert SP.activation_layers(resnet, pattern=r"^layer1") == [n for n in layers if n.startswith("layer1")]
    assert SP.is_piecewise_linear(resnet)
    vit = timm.create_model("test_vit", pretrained=False)
    assert SP.activation_layers(vit) == [f"blocks.{i}.mlp.act" for i in range(6)]
    assert not SP.is_piecewise_linear(vit)


def test_codes_match_preactivations(resnet):
    layers = ["layer1.0.act1", "layer2.0.act2"]
    x = torch.randn(3, 3, 96, 96, generator=torch.Generator().manual_seed(0))
    ref = {}
    hooks = [resnet.get_submodule(n).register_forward_pre_hook(
        lambda _m, inp, n=n: ref.__setitem__(n, (inp[0] > 0).flatten(2).transpose(1, 2).clone())) for n in layers]
    with torch.no_grad():
        resnet(x)
    for h in hooks:
        h.remove()
    c = SP.codes(resnet, [x[:2], x[2:]], layers)
    for n in layers:
        assert c[n].dtype == torch.bool and torch.equal(c[n], ref[n])  # pre-activation signs, despite in-place ReLU


def test_codes_under_edits():
    from innards.edits import steer

    net = Planar().eval()
    x = torch.randn(6, 2, generator=torch.Generator().manual_seed(3))
    clean = SP.codes(net, [x], "act")["act"]
    assert SP.hamming(clean, SP.codes(net, [x], "act", edits={"fc": steer(torch.zeros(12))})["act"]).sum() == 0
    shift = torch.full((12,), 100.0)
    on = SP.codes(net, [x], "act", edits={"fc": steer(shift)})["act"]
    assert on.all() and torch.allclose(SP.hamming(clean, on), 1 - clean.float().mean((1, 2)))


def test_region_ids_hamming_agreement():
    c = torch.tensor([[1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0]], dtype=torch.bool)
    ids = SP.region_ids(c)
    assert ids[0] == ids[1] and len(ids.unique()) == 3
    assert len(SP.region_ids({"a": c, "b": torch.tensor([[1], [0], [0], [0]], dtype=torch.bool)}).unique()) == 4
    D = SP.hamming(c)
    assert D.shape == (4, 4) and D[0, 1] == 0 and D[0, 3] == pytest.approx(1.0)
    assert SP.hamming(c, c.flip(0)).tolist() == pytest.approx([1.0, 1 / 3, 1 / 3, 1.0])
    within, between = SP.code_agreement(c, torch.tensor([0, 0, 1, 1]))
    assert within == pytest.approx((0 + 2 / 3) / 2) and between > within


def test_path_crossings_count_hyperplanes():
    net = Planar().eval()
    x0, x1 = torch.tensor([[-3.0, -2.0], [0.5, 0.5]]), torch.tensor([[2.5, 3.0], [0.6, 0.4]])
    flips = SP.path_crossings(net, x0, x1, "act", steps=2000)["act"]
    expected = (net.signs(x0) != net.signs(x1)).sum(1)                  # each line crosses each hyperplane once at most
    assert flips.shape == (2, 1999) and torch.equal(flips.sum(1), expected)
    assert SP.path_crossings(net, x0, x0, "act", steps=5)["act"].sum() == 0


def test_local_complexity_grows_with_radius():
    net = Planar().eval()
    x = torch.randn(5, 2, generator=torch.Generator().manual_seed(1))
    tiny = SP.local_complexity(net, x, "act", radius=1e-6, n_dirs=2)["act"]
    huge = SP.local_complexity(net, x, "act", radius=1e3, n_dirs=2)["act"]
    mid = SP.local_complexity(net, x, "act", radius=0.5, n_dirs=2)["act"]
    assert tiny.sum() == 0 and torch.equal(huge, torch.full((5,), 12))  # every line cuts a huge neighbourhood
    assert (tiny <= mid).all() and (mid <= huge).all()


def test_slice_partition_matches_direct_codes():
    net = Planar().eval()
    o, e1, e2 = torch.zeros(2), torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    ids, (u, v) = SP.slice_partition(net, o, e1, e2, "act", res=48, extent=(-2.0, 2.0))
    points = torch.stack([u.reshape(-1), v.reshape(-1)], 1)            # the plane is the input space
    assert torch.equal(ids["act"].reshape(-1), SP.region_ids(net.signs(points)))
    n = int(ids["act"].max()) + 1
    assert 1 < n <= 12 * 13 // 2 + 1                                    # at most H(H+1)/2 + 1 regions for H lines
    edge = SP.boundaries(ids["act"])
    assert edge.dtype == torch.bool and edge.any() and not edge.all()


def test_boundary_distance():
    h = torch.tensor([[3.0, 4.0]])
    W, b = torch.tensor([[1.0, 0.0], [0.0, 2.0]]), torch.tensor([0.0, -2.0])
    assert SP.boundary_distance(h, W, b).tolist() == [[3.0, 3.0]]


def test_jacobian_spectrum_of_linear_maps():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 5), nn.Linear(5, 4)).eval()
    x = torch.randn(3, 6)
    sv = SP.jacobian_spectrum(net, x, "1", k=4)
    expected = torch.linalg.svdvals(net[1].weight @ net[0].weight).detach()
    torch.testing.assert_close(sv, expected.expand(3, 4), atol=1e-5, rtol=1e-4)
    sv_mid = SP.jacobian_spectrum(net, x, "1", src="0", k=4)
    torch.testing.assert_close(sv_mid, torch.linalg.svdvals(net[1].weight).detach().expand(3, 4), atol=1e-5,
                               rtol=1e-4)


def test_jacobian_spectrum_on_a_vit():
    vit = timm.create_model("test_vit", pretrained=False).eval()
    sv = SP.jacobian_spectrum(vit, torch.randn(2, 3, 160, 160), "blocks.3", src="blocks.1", k=3)
    assert sv.shape == (2, 3) and (sv > 0).all() and (sv[:, :-1] >= sv[:, 1:]).all()


def test_in_hull():
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    inside = SP.in_hull(X, torch.tensor([[0.2, 0.2], [0.0, 0.5], [0.8, 0.8], [-0.1, 0.0]]))
    assert inside.tolist() == [True, True, False, False]
