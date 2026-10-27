"""Tests for innards.analysis.interp."""

import pytest
import torch
from torch import nn

timm = pytest.importorskip("timm")

from innards.analysis import interp as I


@pytest.fixture(scope="module")
def vit():
    torch.manual_seed(0)
    return timm.create_model("test_vit", pretrained=False).eval()


def test_top_k():
    X = torch.tensor([[1.0, 0.0], [3.0, 0.0], [-2.0, 0.0]])
    values, examples = I.top_k(X, torch.tensor([1.0, 0.0]), k=2)
    assert examples.tolist() == [1, 0] and values.tolist() == [3.0, 1.0]
    assert I.top_k(X, torch.tensor([1.0, 0.0]), k=1, largest=False)[1].tolist() == [2]
    T = torch.zeros(3, 4, 2)
    T[2, 1, 0], T[0, 3, 0] = 5.0, 4.0
    values, examples, positions = I.top_k(T, torch.tensor([1.0, 0.0]), k=2)
    assert examples.tolist() == [2, 0] and positions.tolist() == [1, 3]


def test_spatial_map():
    tokens = torch.randn(2, 1 + 3 * 4, 5)
    V = torch.randn(2, 5)
    maps = I.spatial_map(tokens, V, (3, 4), exclude=(0,))
    assert maps.shape == (2, 2, 3, 4)
    torch.testing.assert_close(maps[1, 0, 2, 1], (tokens[1, 1 + 2 * 4 + 1].double() @ V[0].double()))
    assert I.spatial_map(tokens, V[0], (3, 4), exclude=(0,), size=(30, 40)).shape == (2, 30, 40)


class Lin(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin, self.head = nn.Linear(4, 8), nn.Linear(8, 2)

    def forward(self, x):
        return self.head(torch.tanh(self.lin(x)).mean(1))


@pytest.mark.parametrize("method", ["grad_x_input", "integrated_gradients"])
def test_input_attribution_completeness_when_linear(method):
    net, x, v = Lin().eval(), torch.randn(3, 5, 4), torch.randn(8)
    attr = I.input_attribution(net, x, "lin", v, method=method, steps=4)
    with torch.no_grad():
        score = lambda inp: (net.lin(inp) @ v).sum(1)
        expected = score(x) - score(torch.zeros_like(x))
    assert attr.shape == x.shape
    torch.testing.assert_close(attr.sum((1, 2)), expected, atol=1e-5, rtol=1e-4)


def test_input_attribution_on_a_vit(vit):
    x = torch.randn(2, 3, 160, 160)
    attr = I.input_attribution(vit, x, "blocks.2", torch.randn(64), exclude=(0,))
    assert attr.shape == x.shape and attr.abs().sum() > 0
    with pytest.raises(ValueError):
        I.input_attribution(vit, x, "blocks.2", torch.randn(64), method="nope")


def test_readout_effect_and_logit_lens(vit):
    V, W = torch.randn(3, 64), vit.head.weight
    torch.testing.assert_close(I.readout_effect(V, W), V.double() @ W.detach().double().T)
    x = torch.randn(2, 3, 160, 160)
    from innards.extract import capture
    with torch.no_grad():
        last = capture(vit, x, "blocks.5")["blocks.5"]
        torch.testing.assert_close(I.logit_lens(vit, last), vit(x), atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        I.logit_lens(nn.Linear(2, 2), torch.randn(1, 2, 2))


def test_attention_maps_and_rollout(vit):
    x = torch.randn(2, 3, 160, 160)
    names = I.attention_modules(vit)
    assert names == [f"blocks.{i}.attn" for i in range(6)]
    maps = I.attention_maps(vit, x)
    assert list(maps) == names
    heads = vit.blocks[0].attn.num_heads
    for a in maps.values():
        assert a.shape == (2, heads, 101, 101)
        torch.testing.assert_close(a.sum(-1), torch.ones(2, heads, 101))
    assert all(vit.blocks[i].attn.fused_attn for i in range(6))         # restored

    ref = {}
    vit.blocks[2].attn.fused_attn = False
    h = vit.blocks[2].attn.attn_drop.register_forward_pre_hook(lambda _m, inp: ref.__setitem__("a", inp[0]))
    with torch.no_grad():
        vit(x)
    h.remove()
    vit.blocks[2].attn.fused_attn = True
    torch.testing.assert_close(maps["blocks.2.attn"], ref["a"])
    assert list(I.attention_maps(vit, x, "blocks.4.attn")) == ["blocks.4.attn"]

    rollout = I.attention_rollout(maps)
    assert rollout.shape == (2, 101, 101)
    torch.testing.assert_close(rollout.sum(-1), torch.ones(2, 101, dtype=rollout.dtype))
    eye = torch.eye(4).expand(1, 2, 4, 4)
    torch.testing.assert_close(I.attention_rollout([eye, eye]), torch.eye(4, dtype=torch.float64)[None])
    with pytest.raises(ValueError):
        I.attention_rollout([eye], head_fusion="median")


def test_scores_accept_functions_and_direction_sets():
    X = torch.randn(3, 5, 4, generator=torch.Generator().manual_seed(0))
    V = torch.randn(2, 4, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(I.scores(X, V), X.double() @ V.double().T)
    torch.testing.assert_close(I.scores(X, V[0]), X.double() @ V[0].double())
    torch.testing.assert_close(I.scores(X, lambda t: t @ V[0]), I.scores(X, V[0]), atol=1e-6, rtol=1e-6)
    vals, ex, pos = I.top_k(X, lambda t: t[..., 2], k=2)
    assert vals[0] == X[..., 2].max() and X[ex[0], pos[0], 2] == vals[0]
    maps = I.spatial_map(X[:, 1:], lambda t: t[..., :2], grid=(2, 2))
    assert maps.shape == (3, 2, 2, 2)
    single = I.spatial_map(X[:, 1:], lambda t: t[..., 1], grid=(2, 2))
    torch.testing.assert_close(single, maps[:, 1])


def test_top_outputs():
    effects = torch.tensor([[0.5, -1.0, 2.0, 0.0], [1.0, 3.0, -2.0, -0.5]])
    top_v, top_i, bot_v, bot_i = I.top_outputs(effects, n=2)
    assert top_i.tolist() == [[2, 0], [1, 0]] and bot_i.tolist() == [[1, 3], [2, 3]]
    assert top_v.tolist() == [[2.0, 0.5], [3.0, 1.0]] and bot_v.tolist() == [[-1.0, 0.0], [-2.0, -0.5]]
    assert I.top_outputs(effects, n=10)[0].shape == (2, 4)
