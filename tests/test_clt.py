"""Cross-layer transcoders."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from innards.edits import run
from innards.sae.clt import CLTConfig, CrossLayerTranscoder, clt_edits, clt_hooks
from innards.extract import RowStream
from innards.sae.model import SAE
from innards.sae.trainer import KLFinetuneConfig, SAETrainer, TrainerConfig


def tcfg(tmp_path, **kw):
    kw.setdefault("log_every", 10_000)
    return TrainerConfig(output_dir=str(tmp_path), device="cpu", **kw)


class ToyResidualNet(nn.Module):
    """Residual stream with pre-norm MLP blocks (no attention) and a mean-pooled head."""

    def __init__(self, d=16, n_layers=3, n_classes=5):
        super().__init__()
        self.embed = nn.Linear(8, d)
        self.lns = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.mlps = nn.ModuleList(nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
                                  for _ in range(n_layers))
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        h = self.embed(x)
        for ln, mlp in zip(self.lns, self.mlps):
            h = h + mlp(ln(h))
        return self.head(h).mean(1)


def cross_layer_data(n=20_000, n_layers=3, d=24, n_true=64, active=3, seed=0):
    """MLP outputs y_m = sum_{l <= m} z_l D[l -> m]: features first readable at layer l keep writing later."""
    g = torch.Generator().manual_seed(seed)
    z = [(torch.rand(n, n_true, generator=g) < active / n_true) * torch.rand(n, n_true, generator=g) * 3
         for _ in range(n_layers)]
    enc = [torch.nn.functional.normalize(torch.randn(n_true, d, generator=g), dim=-1) for _ in range(n_layers)]
    x = torch.stack([z[l] @ enc[l] for l in range(n_layers)], 1)
    dec = {(l, m): torch.nn.functional.normalize(torch.randn(n_true, d, generator=g), dim=-1)
           for l in range(n_layers) for m in range(l, n_layers)}
    y = torch.stack([sum(z[l] @ dec[(l, m)] for l in range(m + 1)) for m in range(n_layers)], 1)
    return x * 4.0, y * 2.0 + 1.0


def loader(x, y, bs=512):
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=True, drop_last=True)


@pytest.mark.parametrize("kw", [dict(activation="topk", k=6), dict(activation="jumprelu", sparsity_coeff=3e-3)])
def test_clt_beats_per_layer_transcoders(tmp_path, kw):
    x, y = cross_layer_data()
    torch.manual_seed(0)
    clt = CrossLayerTranscoder(n_layers=3, d_in=24, d_sae=128, dead_tokens_threshold=20_000, **kw)
    tr = SAETrainer(clt, tcfg(tmp_path, num_epochs=8, lr=3e-3, sparsity_warmup_steps=50))
    tr.train(loader(x, y))
    clt_metrics = tr.evaluate(loader(x, y))
    assert {"eval/layer0/fve", "eval/layer2/l0"} <= set(clt_metrics)

    # Per-layer transcoders only see x_m, but y_m depends on features first visible at layers < m.
    per_layer = []
    for m in range(3):
        torch.manual_seed(0)
        tc = SAE(d_in=24, d_sae=128, transcoder=True, dead_tokens_threshold=20_000, **kw)
        t = SAETrainer(tc, tcfg(tmp_path / f"tc{m}", num_epochs=8, lr=3e-3, sparsity_warmup_steps=50))
        t.train(loader(x[:, m], y[:, m]))
        per_layer.append(t.evaluate(loader(x[:, m], y[:, m]))["eval/fve"])
    assert abs(clt_metrics["eval/layer0/fve"] - per_layer[0]) < 0.15  # layer 0: same information
    for m in (1, 2):
        assert clt_metrics[f"eval/layer{m}/fve"] > per_layer[m] + 0.15, (clt_metrics, per_layer)


def test_clt_save_load_and_checkpoint(tmp_path):
    x, y = cross_layer_data(n=4096)
    torch.manual_seed(0)
    clt = CrossLayerTranscoder(n_layers=3, d_in=24, d_sae=64, activation="batchtopk", k=4, skip_connection=True)
    tr = SAETrainer(clt, tcfg(tmp_path, total_tokens=20 * 512, checkpoint_every=10))
    tr.train(loader(x, y))
    clt.eval()
    loaded = CrossLayerTranscoder.load(tmp_path / "sae.safetensors")
    torch.testing.assert_close(loaded.reconstruct(x[:256]), clt.reconstruct(x[:256]))
    torch.testing.assert_close(loaded.encode(x[:256]), clt.encode(x[:256]))
    resumed = SAETrainer.from_checkpoint(tmp_path / "checkpoint.pt")
    assert isinstance(resumed.model, CrossLayerTranscoder) and resumed.step == 20
    with pytest.raises(ValueError, match="targets"):
        clt(x[:4])
    with pytest.raises(ValueError):
        CLTConfig(n_layers=3, d_in=8, d_sae=16, activation="topk")  # k missing


def test_clt_on_model_store_splice_and_downstream(tmp_path):
    torch.manual_seed(0)
    net = ToyResidualNet()
    inputs = torch.randn(256, 6, 8)
    data = DataLoader(TensorDataset(inputs), batch_size=32)
    store = RowStream(net, data, clt_hooks(list(net.lns), list(net.mlps)), batch_rows=256, buffer_rows=1024)
    batch = next(iter(store))
    clt = CrossLayerTranscoder(n_layers=3, d_in=16, d_sae=128, activation="topk", k=8)
    x, y = clt.batch_tensors(batch)
    assert x.shape == y.shape == (256, 3, 16)
    with torch.no_grad():  # aligned: y_l is MLP_l applied to x_l
        for m in range(3):
            torch.testing.assert_close(net.mlps[m](x[:, m]), y[:, m], atol=1e-5, rtol=1e-5)

    tr = SAETrainer(clt, tcfg(tmp_path, total_tokens=60_000))
    tr.train(store)
    assert tr.evaluate(store, max_batches=3)["eval/fve"] > 0.5
    clt.eval()
    lns, mlps = list(net.lns), list(net.mlps)
    with torch.no_grad():
        clean = net(inputs[:8])
        kept, _ = run(net, inputs[:8], clt_edits(clt, lns, mlps, keep_error=True, model=net))
        torch.testing.assert_close(kept, clean, atol=1e-5, rtol=1e-5)
        replaced, _ = run(net, inputs[:8], clt_edits(clt, lns, mlps, model=net))
        assert not torch.allclose(replaced, clean, atol=1e-6)
        names = [f"lns.{i}" for i in range(3)], [f"mlps.{i}" for i in range(3)]
        torch.testing.assert_close(run(net, inputs[:8], clt_edits(clt, *names))[0], replaced)  # layer names work too
        edited, _ = run(net, inputs[:8], clt_edits(clt, lns, mlps, keep_error=True, model=net,
                                                   edit=lambda layer, a: torch.zeros_like(a) if layer == 0 else a))
        assert not torch.allclose(edited, clean, atol=1e-6)

    ce = lambda out, batch: nn.functional.cross_entropy(out, out.detach().argmax(-1))
    metrics = tr.evaluate_downstream(net, list(net.lns), data, KLFinetuneConfig(), loss_fn=ce,
                                     target=list(net.mlps))
    assert metrics["downstream/loss_ablated"] > metrics["downstream/loss_sae"]
    assert metrics["downstream/kl"] >= 0 and 0.0 < metrics["downstream/loss_recovered"] < 1.5
