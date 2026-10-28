"""Tests for innards.analysis.steering (and soft codes) against exact references on a small token MLP."""

import pytest
import torch
from torch import nn

from innards.analysis import metrics
from innards.analysis.spline import codes, gate
from innards.analysis.steering import (
    activation_slice,
    decisive,
    shift,
    slope_jumps,
    spline_steer,
    steer_path
)
from innards.edits import run


class TokenMLP(nn.Module):
    """lin1 -> act1 -> lin2 -> act2 -> mean over tokens -> head, on tokens (batch, tokens, d)."""

    def __init__(self, act=nn.ReLU, d=8, hidden=16, classes=5, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lin1, self.act1 = nn.Linear(d, hidden), act()
        self.lin2, self.act2 = nn.Linear(hidden, hidden), act()
        self.head = nn.Linear(hidden, classes)

    def after(self, h):
        """Everything after lin1, for references."""
        return self.head(self.act2(self.lin2(self.act1(h))).mean(1))

    def forward(self, x):
        return self.after(self.lin1(x))


@pytest.fixture
def setup():
    net = TokenMLP().eval()
    x = torch.randn(3, 4, 8, generator=torch.Generator().manual_seed(1))
    v = torch.randn(16, generator=torch.Generator().manual_seed(2))
    return net, x, v, torch.linspace(0, 3, 61)


def test_outputs_slopes_and_flips_are_exact(setup):
    net, x, v, alphas = setup
    path = steer_path(net, x, "lin1", v, alphas, ["act1", "act2"], chunk=20)  # several chunks per path
    h = net.lin1(x).detach()
    ref = torch.stack([net.after(h + a * v) for a in alphas], 1).detach()
    torch.testing.assert_close(path.outputs, ref, atol=1e-5, rtol=1e-5)
    for j in (0, 17, 60):  # exact derivatives d/dalpha f(h + alpha v)
        _, d = torch.func.jvp(lambda a: net.after(h + a * v), (alphas[j],), (torch.tensor(1.0),))
        torch.testing.assert_close(path.slopes[:, j], d.detach(), atol=1e-5, rtol=1e-5)
    z1 = torch.stack([h + a * v for a in alphas], 1).reshape(3, len(alphas), -1)  # act1 pre-activations
    torch.testing.assert_close(path.flips["act1"], ((z1[:, 1:] > 0) != (z1[:, :-1] > 0)).sum(-1))
    assert path.shapes == {"act1": (4, 16), "act2": (4, 16)}
    # act1's pre-activation is linear in alpha, so interpolated crossings are exact: alpha = -h_u / v_u
    c = path.crossings
    first = c.layer == 0
    hu = h.reshape(3, -1)[c.example[first], c.unit[first]]
    vu = v.repeat(4)[c.unit[first]]
    torch.testing.assert_close(c.alpha[first], -hu / vu, atol=1e-5, rtol=1e-5)
    seg = c.segment[first]
    assert ((alphas[seg] <= c.alpha[first]) & (c.alpha[first] <= alphas[seg + 1])).all()
    assert int(first.sum()) == int(path.flips["act1"].sum())


def test_piecewise_linear_between_crossings(setup):
    net, x, v, alphas = setup
    path = steer_path(net, x, "lin1", v, alphas, ["act1", "act2"])
    jumps = slope_jumps(path)
    crossed = sum(path.flips.values()) > 0
    assert (jumps[~crossed] < 1e-5).all()  # no crossing: same affine piece, same slope
    assert (jumps[crossed] > 0).any()
    # within a piece the output moves by exactly slope * step
    step = (path.outputs[:, 1:] - path.outputs[:, :-1]) / (alphas[1] - alphas[0])
    torch.testing.assert_close(step[~crossed], path.slopes[:, :-1][~crossed], atol=1e-4, rtol=1e-4)
    # projections: an index picks a class, a readout vector projects
    torch.testing.assert_close(slope_jumps(path, 2), slope_jumps(path, torch.eye(5)[2]))
    path = steer_path(net, x, "lin1", v, alphas, target=2)
    where, jump, idx = decisive(path, k=3)
    assert where.shape == jump.shape == idx.shape == (3, 3) and (idx >= 0).all()
    assert (jump.abs()[:, :-1] >= jump.abs()[:, 1:]).all()
    torch.testing.assert_close(path.crossings.alpha[idx], where)
    assert (path.crossings.example[idx] == torch.arange(3)[:, None]).all()


def _reference_crossings(net, x, v, lo, hi, n=200_001):
    """Crossings of act1 / act2 from a very fine grid: {(example, layer, unit): sorted alphas}."""
    a = torch.linspace(lo, hi, n)
    h = net.lin1(x).detach()
    out = {}
    with torch.no_grad():
        for e in range(x.shape[0]):
            z1 = h[e].reshape(1, -1) + a[:, None] * v.repeat(4)[None]  # (n, 4 * 16), token-major
            z2 = net.lin2(net.act1(z1.reshape(n, 4, 16))).reshape(n, -1)
            for li, z in enumerate((z1, z2)):
                t, u = ((z[1:] > 0) != (z[:-1] > 0)).nonzero(as_tuple=True)
                for ti, ui in zip(t.tolist(), u.tolist()):
                    z0_, z1_ = z[ti, ui], z[ti + 1, ui]
                    out.setdefault((e, li, ui), []).append(float(a[ti] - z0_ / (z1_ - z0_) * (a[1] - a[0])))
    return out


def test_refinement_finds_double_crossings_and_attributes_kinks(setup):
    net, x, v, _ = setup
    alphas = torch.linspace(-6, 6, 7)  # coarse: many kinks per segment
    path = steer_path(net, x, "lin1", v, alphas, target=2, refine=20)
    assert path.layers == ["act1", "act2"] and path.resolved.all()
    c = path.crossings
    assert c.exact.all() and not c.jump.isnan().any()
    found = {}
    for e, li, u, a in zip(c.example.tolist(), c.layer.tolist(), c.unit.tolist(), c.alpha.tolist()):
        if li >= 0:
            found.setdefault((e, li, u), []).append(a)
    ref = _reference_crossings(net, x, v, -6, 6)
    assert found.keys() == ref.keys()
    for key in ref:
        assert len(found[key]) == len(ref[key]), key
        torch.testing.assert_close(torch.tensor(sorted(found[key])), torch.tensor(ref[key]), atol=1e-3, rtol=0)
    # units switching twice inside one coarse segment are found
    twice = [(e, li, u, s) for (e, li, u), s in
             {(e, li, u): torch.tensor(a) for (e, li, u), a in found.items()}.items()
             if len(s) > 1 and len(set(torch.bucketize(s, alphas).tolist())) < len(s)]
    assert twice, "the example should contain double crossings within a segment"
    # the per-crossing slope changes add up to each segment's slope change
    seg_sum = torch.zeros(3, len(alphas) - 1).index_put_((c.example, c.segment), c.jump, accumulate=True)
    torch.testing.assert_close(seg_sum, slope_jumps(path, 2), atol=1e-4, rtol=1e-4)
    # tracking only act2: act1's kinks are attributed to untracked units (layer -1), sums unchanged
    part = steer_path(net, x, "lin1", v, alphas, layers=["act2"], target=2, refine=20)
    assert part.resolved.all() and (part.crossings.layer == -1).any()
    pc = part.crossings
    seg_sum = torch.zeros(3, len(alphas) - 1).index_put_((pc.example, pc.segment), pc.jump, accumulate=True)
    torch.testing.assert_close(seg_sum, slope_jumps(part, 2), atol=1e-4, rtol=1e-4)
    # without refinement, unresolved segments are flagged and fall back to sign changes
    flat = steer_path(net, x, "lin1", v, alphas, refine=0)
    assert not flat.resolved.all() and not flat.crossings.exact.all()
    assert flat.crossings.jump[~flat.crossings.exact].isnan().all()


def test_soft_codes_and_transition_mass():
    net = TokenMLP(act=nn.GELU).eval()
    x = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(3))
    v = torch.randn(16, generator=torch.Generator().manual_seed(4))
    alphas = torch.linspace(-2, 2, 21)
    soft = codes(net, [x], "act1", soft=True)["act1"]
    torch.testing.assert_close(soft, gate(net.act1)(net.lin1(x)).detach())
    torch.testing.assert_close(soft, 0.5 * (1 + torch.erf(net.lin1(x).detach() / 2 ** 0.5)))
    assert codes(net, [x], "act1")["act1"].dtype == torch.bool
    path = steer_path(net, x, "lin1", v, alphas, ["act1"], soft=True)
    h = net.lin1(x).detach()
    g = torch.stack([gate(net.act1)(h + a * v) for a in alphas], 1).reshape(2, len(alphas), -1)
    torch.testing.assert_close(path.flips["act1"], (g[:, 1:] - g[:, :-1]).abs().sum(-1), atol=1e-5, rtol=1e-5)
    assert gate(nn.SiLU())(torch.tensor(0.0)) == 0.5 and gate(nn.ReLU())(torch.tensor(-1.0)) == 0


def test_exclude_and_mask(setup):
    net, x, v, _ = setup
    h = net.lin1(x).detach()
    out = shift(torch.ones(3, 16), exclude=(0,))(h)
    assert torch.equal(out[:, 0], h[:, 0]) and torch.equal(out[:, 1:], h[:, 1:] + 1)
    mask = torch.tensor([[True, False, True, False]]).expand(3, -1)
    path = steer_path(net, x, "lin1", v, [0.0, 1.0], ["act1"], mask=mask, derivative=False)
    steered = h + mask.unsqueeze(-1) * v
    torch.testing.assert_close(path.outputs[:, 1], net.after(steered).detach(), atol=1e-5, rtol=1e-5)
    assert path.slopes is None
    with pytest.raises(ValueError):
        slope_jumps(path)


def test_activation_slice(setup):
    net, x, v, _ = setup
    v2 = torch.randn(16, generator=torch.Generator().manual_seed(5))
    ids, out, (u, w) = activation_slice(net, x[0], "lin1", v, v2, ["act1", "act2"], res=9, extent=(-2, 2), chunk=20)
    assert ids["act1"].shape == ids["act2"].shape == (9, 9) and out.shape == (9, 9, 5)
    h = net.lin1(x[:1]).detach()
    i, j = 2, 7
    torch.testing.assert_close(out[i, j], net.after(h + u[i, j] * v + w[i, j] * v2)[0].detach(), atol=1e-5,
                               rtol=1e-5)
    # region ids up to act2 refine those up to act1
    assert len(ids["act2"].unique()) >= len(ids["act1"].unique()) > 1


def test_spline_steer(setup):
    net, x, _, _ = setup
    deltas, values = spline_steer(net, x, "lin1", metrics.logit(3), steps=10, step_size=0.3)
    assert deltas.shape == (3, 11, 16) and values.shape == (3, 11)
    assert (deltas[:, 0] == 0).all()
    torch.testing.assert_close(deltas.diff(dim=1).norm(dim=-1), torch.full((3, 10), 0.3))
    assert (values[:, -1] > values[:, 0]).all() and (values.diff(dim=1) > -1e-6).all()  # ascent
    final, _ = run(net, x, {"lin1": shift(deltas[:, -1])})
    torch.testing.assert_close(final[:, 3], values[:, -1], atol=1e-5, rtol=1e-5)
    basis = torch.eye(16)[:3]
    deltas, _ = spline_steer(net, x, "lin1", metrics.logit(3), steps=4, basis=basis)
    assert (deltas[..., 3:] == 0).all()


def test_spline_steer_is_gradient_ascent_on_a_linear_model():
    lin = nn.Sequential(nn.Linear(8, 6), nn.Identity(), nn.Linear(6, 4)).eval()
    x = torch.randn(2, 8)
    deltas, _ = spline_steer(lin, x, "0", metrics.logit(1), steps=5, step_size=0.5)
    w = lin[2].weight[1].detach()
    torch.testing.assert_close(deltas[:, -1], (2.5 * w / w.norm()).expand(2, -1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seed", range(5))
def test_refinement_is_exact_across_networks(seed):
    net = TokenMLP(seed=seed).eval()
    x = torch.randn(3, 4, 8, generator=torch.Generator().manual_seed(100 + seed))
    v = torch.randn(16, generator=torch.Generator().manual_seed(200 + seed))
    alphas = torch.linspace(-6, 6, 7)
    path = steer_path(net, x, "lin1", v, alphas, target=1, refine=20)
    assert path.resolved.float().mean() >= 0.9  # crossings closer than float32 can separate stay unresolved
    c = path.crossings
    found = {}
    for e, li, u, a in zip(c.example.tolist(), c.layer.tolist(), c.unit.tolist(), c.alpha.tolist()):
        found.setdefault((e, li, u), []).append(a)
    ref = _reference_crossings(net, x, v, -6, 6)
    assert found.keys() == ref.keys()
    for key in ref:
        torch.testing.assert_close(torch.tensor(sorted(found[key])), torch.tensor(ref[key]), atol=1e-3, rtol=0)
    seg_sum = torch.zeros(3, len(alphas) - 1).index_put_((c.example, c.segment), c.jump, accumulate=True)
    done = path.resolved
    torch.testing.assert_close(seg_sum[done], slope_jumps(path, 1)[done], atol=1e-5, rtol=1e-5)
    assert c.exact[done[c.example, c.segment]].all() and c.jump[~c.exact].isnan().all()
