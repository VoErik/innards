"""Multi-hook harvesting, dict caches and SAEGroup training."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sae.cache import CachedActivations, cache_activations
from sae.data import ActivationStore
from sae.group import SAEGroup
from sae.model import SAE
from sae.trainer import SAETrainer, TrainerConfig
from test_core import synthetic


def tcfg(tmp_path, **kw):
    kw.setdefault("log_every", 10_000)
    return TrainerConfig(output_dir=str(tmp_path), device="cpu", **kw)


def mlp_net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 12), nn.ReLU())


def test_multi_hook_store_and_cache_keep_tokens_aligned(tmp_path):
    net = mlp_net()
    loader = [torch.randn(4, 10, 8) for _ in range(20)]  # (batch, tokens, d) inputs
    store = ActivationStore(net, {"h1": net[1], "h2": net[3]}, loader, batch_tokens=64, buffer_tokens=256,
                            exclude_positions=(0,))
    batch = next(iter(store))
    assert set(batch) == {"h1", "h2"} and batch["h1"].shape == (64, 16) and batch["h2"].shape == (64, 12)
    with torch.no_grad():  # rows are the same tokens after the joint shuffle
        torch.testing.assert_close(net[3](net[2](batch["h1"])), batch["h2"])

    path = cache_activations(store, tmp_path / "multi", shard_tokens=100, dtype=torch.float32)
    cached = CachedActivations(path, batch_tokens=32)
    assert cached.meta["kind"] == "dict" and [f for f, _ in cached.meta["fields"]] == ["h1", "h2"]
    for b in cached:
        with torch.no_grad():
            torch.testing.assert_close(net[3](net[2](b["h1"])), b["h2"])
    assert sum(b["h1"].shape[0] for b in cached) == cached.n_tokens // 32 * 32


def _dict_loader(n=20_000, bs=512):
    a = synthetic(n=n, seed=0)
    b = synthetic(n=n, d_in=24, seed=1) * 0.1  # different width and scale
    return [{"a": a[i : i + bs], "b": b[i : i + bs]} for i in range(0, n - bs + 1, bs)]


def test_group_trains_each_member(tmp_path):
    torch.manual_seed(0)
    group = SAEGroup({"a": SAE(d_in=32, d_sae=256, k=8, dead_tokens_threshold=20_000),
                      "b": SAE(d_in=24, d_sae=128, activation="batchtopk", k=8, dead_tokens_threshold=20_000)})
    data = _dict_loader()
    tr = SAETrainer(group, tcfg(tmp_path, num_epochs=6, k_decay_steps=50, log_every=50))
    tr.train(data)
    metrics = tr.evaluate(data)
    assert metrics["eval/a/fve"] > 0.8 and metrics["eval/b/fve"] > 0.8, metrics
    assert abs(metrics["eval/a/l0"] - 8) < 1e-6 and 4 < metrics["eval/b/l0"] < 16
    assert group["a"].k_current == 8  # annealing restored the configured k

    loaded = SAEGroup.load(tmp_path / "sae")
    assert loaded.names == ["a", "b"]
    x = data[0]["a"]
    torch.testing.assert_close(loaded["a"].reconstruct(x), group["a"].eval().reconstruct(x), atol=1e-4, rtol=1e-4)


def test_group_checkpoint_resume(tmp_path):
    torch.manual_seed(0)
    group = SAEGroup({"a": SAE(d_in=32, d_sae=64, k=4), "b": SAE(d_in=24, d_sae=64, activation="jumprelu")})
    data = _dict_loader(n=8192, bs=256)
    tr = SAETrainer(group, tcfg(tmp_path, total_tokens=30 * 256, checkpoint_every=10))
    tr.train(data)
    resumed = SAETrainer.from_checkpoint(tmp_path / "checkpoint.pt")
    assert isinstance(resumed.model, SAEGroup) and resumed.step == 30
    for name in ("a", "b"):
        torch.testing.assert_close(resumed.model[name].w_enc, group[name].w_enc)


def test_group_from_vit_store(tmp_path):
    from test_core import ToyViT

    torch.manual_seed(0)
    vit = ToyViT()
    loader = DataLoader(TensorDataset(torch.randn(256, 3, 16, 16)), batch_size=32)
    store = ActivationStore(vit, {"block1": vit.block1, "block2": vit.block2}, loader, batch_tokens=256,
                            buffer_tokens=2048, exclude_positions=(0,))
    group = SAEGroup({name: SAE(d_in=32, d_sae=128, k=8) for name in ("block1", "block2")})
    tr = SAETrainer(group, tcfg(tmp_path, total_tokens=40_000))
    tr.train(store)
    metrics = tr.evaluate(store, max_batches=4)
    assert metrics["eval/block1/fve"] > 0.5 and metrics["eval/block2/fve"] > 0.5, metrics
