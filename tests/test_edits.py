"""Tests for nnsight interventions (innards.edits) and layer resolution."""

import pytest
import torch
from torch import nn

timm = pytest.importorskip("timm")

from innards.edits import (
    ablate,
    clamp,
    compose,
    cross,
    edit_output,
    keep_mask,
    mean_ablate,
    patch,
    rows,
    run,
    scale,
    steer,
    zero
)
from innards.extract import (
    _PLANS,
    capture,
    plan
)
from innards.layouts import to_tokens
from innards.models import (
    layer_name,
    wrap
)
from _hooks import replace_output

LAYERS = {"test_vit": ("blocks.2", "btd"), "test_resnet": ("layer2", "bchw"), "test_mambaout": ("stages.1", "bhwc")}


@pytest.fixture(scope="module", params=list(LAYERS))
def case(request):
    torch.manual_seed(0)
    model = timm.create_model(request.param, pretrained=False).eval()
    layer, layout = LAYERS[request.param]
    x = torch.randn(2, 3, 160, 128, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        clean = model(x)
        tokens = to_tokens(capture(model, x, layer)[layer], layout)
    return model, layer, layout, x, clean, tokens


def test_run_without_edits_is_clean(case):
    model, layer, _, x, clean, tokens = case
    with torch.no_grad():
        out, reads = run(model, x, read=layer)
    assert torch.equal(out, clean)
    assert torch.equal(reads[layer], tokens)


def test_edit_matches_hook(case):
    model, layer, layout, x, *_ = case
    v = torch.randn(case[5].shape[-1])
    with torch.no_grad():
        out, _ = run(model, x, {layer: steer(v, 2.0)})
        # the same edit through a plain forward hook, in the layer's own layout
        shape = (1, -1, 1, 1) if layout == "bchw" else (-1,)
        with replace_output(model.get_submodule(layer), lambda a: a + 2.0 * v.reshape(shape)):
            ref = model(x)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_edit_factories(case):
    model, layer, _, x, clean, tokens = case
    d = tokens.shape[-1]
    v, w = torch.randn(d), torch.randn(d)
    with torch.no_grad():
        read = lambda edit: run(model, x, {layer: edit}, read=layer)[1][layer]
        torch.testing.assert_close(read(steer(v, 3.0)), tokens + 3.0 * v)
        assert read(zero()).abs().max() == 0
        torch.testing.assert_close(read(mean_ablate(v)), v.expand_as(tokens))

        projected = read(ablate(torch.stack([v, v + w])))           # not orthonormal
        torch.testing.assert_close(projected @ torch.stack([v, w]).T, torch.zeros(*tokens.shape[:2], 2),
                                   atol=1e-3, rtol=0)
        torch.testing.assert_close(read(ablate(v)), read(compose(ablate(v), ablate(v))), atol=1e-4, rtol=0)

        both = read(compose(steer(v), steer(w, 2.0)))
        torch.testing.assert_close(both, tokens + v + 2.0 * w)

        kept = read(rows(steer(v), exclude=(0,)))
        torch.testing.assert_close(kept[:, 0], tokens[:, 0])
        torch.testing.assert_close(kept[:, 1:], tokens[:, 1:] + v)

        source = torch.randn_like(tokens)
        one = torch.zeros(tokens.shape[:2], dtype=torch.bool)
        one[:, 3] = True
        patched = read(patch(source, mask=one))
        torch.testing.assert_close(patched[:, 3], source[:, 3])
        torch.testing.assert_close(patched[:, 4:], tokens[:, 4:])


def test_edit_output_in_own_trace(case):
    model, layer, layout, x, *_ = case
    v = torch.randn(case[5].shape[-1])
    nn_model = wrap(model)
    with torch.no_grad():
        with nn_model.trace(x):
            edit_output(nn_model.get(layer), steer(v), layout)
            out = nn_model.output.save()
        ref, _ = run(model, x, {layer: steer(v)})
    assert torch.equal(out, ref)


def test_keep_mask():
    t = torch.zeros(2, 4, 3)
    m = torch.tensor([[True, True, False, True], [True, True, True, True]])
    assert keep_mask(t, (0,), m).tolist() == [[False, True, False, True], [False, True, True, True]]
    assert keep_mask(t).all()


def test_unit_edits():
    x = torch.tensor([[[1.0, -2.0, 0.0, 3.0]]])
    assert zero([1])(x).tolist() == [[[1.0, 0.0, 0.0, 3.0]]]
    assert zero()(x).abs().sum() == 0
    assert scale(0, 2.0)(x).tolist() == [[[2.0, -2.0, 0.0, 3.0]]]
    assert scale(torch.tensor([0, 3]), -1.0)(x).tolist() == [[[-1.0, -2.0, 0.0, -3.0]]]
    assert clamp([2, 3], 7.0)(x).tolist() == [[[1.0, -2.0, 7.0, 7.0]]]
    assert clamp([2, 3], 7.0, only_active=True)(x).tolist() == [[[1.0, -2.0, 0.0, 7.0]]]
    mean = torch.tensor([10.0, 20.0, 30.0, 40.0])
    assert mean_ablate(mean, [0, 2])(x).tolist() == [[[10.0, -2.0, 30.0, 3.0]]]
    assert x.tolist() == [[[1.0, -2.0, 0.0, 3.0]]]  # edits do not modify their input


class Toy(nn.Module):
    """A source / target pair (norm -> mlp) and a tuple-returning layer."""

    class Tup(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            return self.lin(x), "aux"

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.norm, self.mlp, self.tup, self.head = nn.LayerNorm(4), nn.Linear(4, 4), self.Tup(), nn.Linear(4, 2)

    def forward(self, x):
        h = x + self.mlp(self.norm(x))
        h, _ = self.tup(h)
        return self.head(h)[:, -1]


def test_cross_and_tuple_outputs():
    net, x = Toy().eval(), torch.randn(3, 5, 4)
    with torch.no_grad():
        clean = net(x)
        # replace the mlp output by its input (read at norm): output = x + norm(x)
        out, reads = run(net, x, cross("norm", "mlp", lambda xs, ys: xs), read=["mlp", "norm"])
        torch.testing.assert_close(reads["mlp"], reads["norm"])
        torch.testing.assert_close(reads["mlp"], net.norm(x))
        assert not torch.allclose(out, clean)
        same, _ = run(net, x, cross("mlp", "mlp", lambda xs, ys: ys))
        torch.testing.assert_close(same, clean)

        out, reads = run(net, x, {"tup": zero()}, read="tup")
        assert reads["tup"].abs().max() == 0
        torch.testing.assert_close(out, net.head.bias.expand(3, 2))


def test_gradients_flow_through_edits():
    net, x = Toy().eval(), torch.randn(3, 5, 4, requires_grad=True)
    net.requires_grad_(False)
    probe = nn.Linear(4, 4)
    out, _ = run(net, x, {"mlp": lambda t: probe(t)})
    out.sum().backward()
    assert probe.weight.grad is not None and probe.weight.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_layer_name():
    net = Toy()
    assert layer_name(net, "mlp") == "mlp"
    assert layer_name(net, net.tup.lin) == "tup.lin"
    assert layer_name(wrap(net), wrap(net).tup) == "tup"
    with pytest.raises(ValueError):
        layer_name(net, nn.Linear(2, 2))


def test_plan_is_cached():
    net, x = Toy().eval(), torch.randn(1, 5, 4)
    calls = []
    handle = net.norm.register_forward_pre_hook(lambda *_: calls.append(1))
    order, layouts = plan(net, x, ["mlp", "norm"])
    assert order == ["norm", "mlp"] and layouts == {"norm": "btd", "mlp": "btd"}
    n = len(calls)
    assert plan(net, x, ["mlp", "norm"]) == (order, layouts) and len(calls) == n   # no forward pass
    assert plan(net, x, "mlp", layouts="btd") == (["mlp"], {"mlp": "btd"})        # single known layer: none either
    handle.remove()
    assert net in _PLANS
