"""Tests for model access, extraction and storage (innards.models / layouts / reduce / extract / store)."""

import json

import pytest
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset
)

timm = pytest.importorskip("timm")

from innards.extract import (
    RowStream,
    capture,
    capture_tokens,
    layer_tokens,
    describe,
    extract,
    extract_to_disk
)
from innards.layouts import (
    LAYOUTS,
    from_tokens,
    infer_layout,
    to_tokens
)
from innards.models import (
    load,
    module_names,
    stage_names,
    unwrap,
    wrap
)
from innards.reduce import (
    drop,
    keep,
    max_pool,
    mean_pool,
    token
)
from innards.store import (
    RowLoader,
    write_rows
)
from innards.store import (
    load_info,
    load_layer,
    save
)
from innards.store import load as load_run

# tiny random-weight timm models, one per layout; layers deliberately out of execution order
MODELS = {
    "test_vit": (["blocks.3", "blocks.1.attn", "blocks.1"], "btd"),
    "test_resnet": (["layer3", "layer1"], "bchw"),
    "test_mambaout": (["stages.2", "stages.0"], "bhwc"),
}
IMAGE_HW = (160, 128)


@pytest.fixture(scope="module", params=list(MODELS))
def model_case(request):
    torch.manual_seed(0)
    model = timm.create_model(request.param, pretrained=False).eval()
    return request.param, wrap(model)


def images(n: int) -> torch.Tensor:
    return torch.randn(n, 3, *IMAGE_HW, generator=torch.Generator().manual_seed(n))


# ---------- layouts ----------


@pytest.mark.parametrize("layout,shape", [("btd", (2, 5, 3)), ("bchw", (2, 3, 4, 5)),
                                          ("bhwc", (2, 4, 5, 3)), ("bd", (2, 3))])
def test_tokens_roundtrip(layout, shape):
    act = torch.randn(shape)
    tokens = to_tokens(act, layout)
    assert tokens.shape[0] == 2 and tokens.shape[-1] == 3 and tokens.ndim == 3
    assert torch.equal(from_tokens(tokens, layout, act), act)


def test_tokens_spatial_order_matches():
    # bchw and bhwc views of the same map give the same tokens
    chw = torch.randn(2, 3, 4, 5)
    assert torch.equal(to_tokens(chw, "bchw"), to_tokens(chw.permute(0, 2, 3, 1), "bhwc"))


@pytest.mark.parametrize("shape,image_hw,expected", [
    ((2, 64), None, "bd"),
    ((2, 197, 768), None, "btd"),
    ((2, 64, 56, 56), None, "bchw"),
    ((2, 56, 56, 64), None, "bhwc"),
    ((2, 64, 40, 32), (160, 128), "bchw"),
    ((2, 40, 32, 64), (160, 128), "bhwc"),
    ((2, 64, 8, 7), (225, 224), "bchw"),       # rounding at odd sizes
    ((2, 16, 16, 16), None, "bchw"),           # ambiguous: PyTorch default
])
def test_infer_layout(shape, image_hw, expected):
    assert infer_layout(shape, image_hw) == expected


def test_infer_layout_errors():
    with pytest.raises(ValueError):
        infer_layout((2, 64, 40, 32))           # non-square without image size
    with pytest.raises(ValueError):
        infer_layout((2, 3, 4, 5, 6))
    assert set(LAYOUTS) == {"btd", "bchw", "bhwc", "bd"}


# ---------- models ----------


def test_wrap_unwrap():
    module = nn.Linear(2, 2)
    model = wrap(module)
    assert wrap(model) is model
    assert unwrap(model) is module and unwrap(module) is module


def test_module_and_stage_names():
    model = wrap(timm.create_model("test_vit", pretrained=False))
    names = module_names(model, depth=2)
    assert "blocks.0" in names and "blocks.0.attn" not in names
    assert module_names(model, pattern=r"^blocks\.\d+$") == [f"blocks.{i}" for i in range(6)]
    assert stage_names(model) == [f"blocks.{i}" for i in range(6)]
    with pytest.raises(ValueError):
        stage_names(nn.Linear(2, 2))


def test_load_returns_preprocess():
    from PIL import Image

    model, preprocess = load("test_vit", pretrained=False)
    x = preprocess(Image.new("RGB", (300, 200)))
    assert x.shape == (3, 160, 160)
    with torch.no_grad():
        assert unwrap(model)(x[None]).shape == (1, unwrap(model).num_classes)


# ---------- describe / capture ----------


def test_describe(model_case):
    name, model = model_case
    layers, layout = MODELS[name]
    info = describe(model, images(2), layers)
    assert set(info) == set(layers)
    order = list(info)
    if name == "test_vit":
        assert order == ["blocks.1.attn", "blocks.1", "blocks.3"]   # child completes before parent
    assert all(i["layout"] == layout for i in info.values())
    if layout != "btd":
        assert all(i["grid"] is not None and i["grid"][0] / i["grid"][1] == 1.25 for i in info.values())


def test_capture_matches_hooks(model_case):
    name, model = model_case
    layers, _ = MODELS[name]
    x = images(2)
    order = list(describe(model, x, layers))
    ref = {}
    handles = [unwrap(model).get_submodule(n).register_forward_hook(
        lambda _m, _i, o, n=n: ref.__setitem__(n, o[0] if isinstance(o, tuple) else o)) for n in layers]
    with torch.no_grad():
        out = capture(model, x, order)
        unwrap(model)(x)
    for h in handles:
        h.remove()
    for n in layers:
        assert torch.equal(out[n], ref[n])


def test_describe_missing_layer():
    class Partial(nn.Module):
        def __init__(self):
            super().__init__()
            self.used, self.unused = nn.Linear(2, 2), nn.Linear(2, 2)

        def forward(self, x):
            return self.used(x)

    with pytest.raises(RuntimeError, match="unused"):
        describe(Partial(), torch.randn(1, 2), ["used", "unused"])


def test_layout_override():
    model = timm.create_model("test_resnet", pretrained=False).eval()
    x = images(1)
    assert describe(model, x, "layer1", layouts="bhwc")["layer1"]["layout"] == "bhwc"
    assert describe(model, x, ["layer1"], layouts={"layer1": "bchw"})["layer1"]["layout"] == "bchw"


# ---------- reductions ----------


def test_reductions():
    t = torch.randn(2, 5, 3)
    assert torch.equal(token(0)(t), t[:, 0])
    assert torch.equal(token(-1)(t), t[:, 4])
    assert torch.equal(keep([1, 3])(t), t[:, [1, 3]])
    assert torch.equal(drop([0])(t), t[:, 1:])
    assert torch.equal(drop(range(2))(t), t[:, 2:])
    assert torch.allclose(mean_pool()(t), t.mean(1))
    assert torch.allclose(mean_pool(exclude=(0,))(t), t[:, 1:].mean(1))
    assert torch.equal(max_pool(exclude=(-1,))(t), t[:, :4].amax(1))


# ---------- extract ----------


def test_extract_dataloader(model_case):
    name, model = model_case
    layers, layout = MODELS[name]
    x, y = images(7), torch.arange(7)
    loader = DataLoader(TensorDataset(x, y), batch_size=3)          # last batch is partial
    acts, targets = extract(model, loader, layers)
    assert torch.equal(targets, y)
    info = describe(model, x, layers)
    with torch.no_grad():
        full = capture(model, x, list(info))
    for n in layers:
        assert acts[n].shape[0] == 7 and acts[n].ndim == 3
        assert torch.allclose(acts[n], to_tokens(full[n], info[n]["layout"]), atol=1e-5)


def test_extract_reductions_and_plain_batches():
    model = timm.create_model("test_vit", pretrained=False).eval()
    x = images(4)
    batches = [x[:2], x[2:]]                                         # no targets
    acts, targets = extract(model, batches, ["blocks.1", "blocks.3", "head"],
                            reduce={"blocks.1": token(0), "blocks.3": mean_pool(exclude=(0,))})
    assert targets is None
    full, _ = extract(model, [x], ["blocks.1", "blocks.3"])
    assert torch.allclose(acts["blocks.1"], full["blocks.1"][:, 0], atol=1e-5)
    assert torch.allclose(acts["blocks.3"], full["blocks.3"][:, 1:].mean(1), atol=1e-5)
    assert acts["head"].shape == (4, model.num_classes)              # "bd" stays (n, d)


def test_extract_dict_batches():
    model = timm.create_model("test_resnet", pretrained=False).eval()
    x = images(3)
    acts, targets = extract(model, [{"x": x, "label": torch.tensor([1, 2, 3])}], "layer2",
                            input_fn=lambda b: ((b["x"],), {}), reduce=mean_pool())
    assert acts["layer2"].shape[0] == 3 and acts["layer2"].ndim == 2
    assert targets.tolist() == [1, 2, 3]


# ---------- storage ----------


def test_extract_to_disk_roundtrip(tmp_path, model_case):
    name, model = model_case
    layers, layout = MODELS[name]
    x, y = images(5), torch.arange(5) * 10
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    acts, targets = extract(model, loader, layers)
    run = extract_to_disk(model, loader, layers, tmp_path / "run", shard_rows=64,
                          metadata={"model": name})
    info = load_info(run)
    assert info["n_examples"] == 5 and info["metadata"] == {"model": name}
    assert all(info["layers"][n]["layout"] == layout for n in layers)

    loaded, loaded_targets = load_run(run)
    assert torch.equal(loaded_targets, targets)
    for n in layers:
        assert torch.equal(loaded[n], acts[n])
        assert torch.equal(load_layer(run, n, flat=True), acts[n].reshape(-1, acts[n].shape[-1]))
    meta = json.loads((run / layers[0] / "meta.json").read_text())
    assert len(meta["shards"]) > 1 and meta["metadata"]["layer"] == layers[0]

    subset, _ = load_run(run, layers=layers[:1])
    assert list(subset) == layers[:1]


def test_layer_dir_is_sae_cache(tmp_path):
    model = timm.create_model("test_vit", pretrained=False).eval()
    x = images(4)
    run = extract_to_disk(model, [x], "blocks.2", tmp_path / "run", reduce=drop([0]), shard_rows=100)
    rows = load_layer(run, "blocks.2", flat=True)
    assert rows.shape == (4 * 80, 64)
    cached = torch.cat(list(RowLoader(run / "blocks.2", batch_rows=40, shuffle=False)))
    assert torch.equal(cached, rows)


def test_save_and_load_in_memory(tmp_path):
    acts = {"a": torch.randn(6, 4, 3), "b": torch.randn(6, 5)}
    targets = torch.arange(6)
    save(tmp_path / "run", acts, targets, info={"a": {"layout": "btd"}}, dtype=torch.float16)
    loaded, loaded_targets = load_run(tmp_path / "run")
    assert torch.equal(loaded_targets, targets)
    assert loaded["a"].dtype == torch.float16
    assert torch.allclose(loaded["a"].float(), acts["a"], atol=1e-2)
    assert loaded["b"].shape == (6, 5)
    assert load_info(tmp_path / "run")["layers"]["a"]["layout"] == "btd"
    with pytest.raises(ValueError):
        save(tmp_path / "bad", {"a": torch.randn(2, 3), "b": torch.randn(3, 3)})


# ---------- token rows: streaming and row caches ----------


def test_layer_tokens_and_row_stream_on_channels_last():
    model = timm.create_model("test_mambaout", pretrained=False).eval()
    x = images(4)
    with torch.no_grad():
        tokens = layer_tokens(model, x, "stages.1")
        both = capture_tokens(model, x, ["stages.1", "stages.0"])
    assert tokens.shape == (4, 20 * 16, 32) and torch.equal(both["stages.1"], tokens)   # bhwc -> tokens, inferred
    mask = torch.ones(4, 20 * 16, dtype=torch.bool)
    mask[1] = False                                                                   # drop example 1 entirely
    stream = RowStream(model, [(x, torch.zeros(4))], "stages.1", batch_rows=64, buffer_rows=128, exclude=(0,),
                       mask_fn=lambda b: mask, seed=0)
    rows = torch.cat(list(stream))
    expected = torch.cat([tokens[i, 1:] for i in (0, 2, 3)])
    assert rows.shape[1] == 32 and len(rows) == len(expected) // 64 * 64
    # every served row is one of the kept tokens (shuffled)
    assert torch.isin(rows[:, 0], expected[:, 0]).all() and not torch.isin(rows[:, 0], tokens[1, :, 0]).any()


def test_write_rows_pairs_and_row_loader(tmp_path):
    x = torch.arange(30.0).reshape(10, 3)
    pairs = [(x[i : i + 4], -x[i : i + 4, :2]) for i in range(0, 10, 4)]
    path = write_rows(pairs, tmp_path / "pairs", shard_rows=3, max_rows=9)
    loader = RowLoader(path, batch_rows=3, shuffle=False)
    assert loader.n_rows == 9 and loader.d_in == 3 and loader.meta["d_out"] == 2 and len(loader) == 3
    a, b = map(torch.cat, zip(*loader))
    torch.testing.assert_close(a, x[:9])
    torch.testing.assert_close(b, -x[:9, :2])                                         # fields stay aligned
    shuffled = list(RowLoader(path, batch_rows=3, seed=1))
    sa, sb = map(torch.cat, zip(*shuffled))
    torch.testing.assert_close(sb, -sa[:, :2])
