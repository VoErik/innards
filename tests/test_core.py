"""Correctness tests. Run with ``python -m pytest test_sae.py -q``."""

import math

import pytest
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from sae.data import (
    ActivationStore,
    from_tokens,
    patch_kept_tokens,
    to_tokens
)
from sae.optim import (
    Signum,
    build_optimizers
)
from sae.model import (
    SAEConfig,
    TopKSAE,
    kmeans
)
from sae.trainer import (
    KLFinetuneConfig,
    SAETrainer,
    TrainerConfig
)


def synthetic(n=20_000, d_in=32, n_true=128, active=4, seed=0, signed=False):
    """Sparse superposition of random directions, with offset and scale."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.nn.functional.normalize(torch.randn(n_true, d_in, generator=g), dim=-1)
    mask = torch.rand(n, n_true, generator=g) < active / n_true
    mags = torch.rand(n, n_true, generator=g) * 3
    if signed:
        mags = mags * torch.randint(0, 2, (n, n_true), generator=g).mul(2).sub(1)
    return ((mask * mags) @ dirs + 2.0) * 7.0


def loader(x, bs=512, shuffle=True):
    return DataLoader(TensorDataset(x), batch_size=bs, shuffle=shuffle, drop_last=True)


def tcfg(tmp_path, **kw):
    kw.setdefault("log_every", 10_000)
    return TrainerConfig(output_dir=str(tmp_path), device="cpu", **kw)


@pytest.mark.parametrize("batchtopk", [False, True])
@pytest.mark.parametrize("abs_topk", [False, True])
def test_sparse_decode_matches_dense(batchtopk, abs_topk):
    torch.manual_seed(0)
    m = TopKSAE(d_in=16, d_sae=64, k=5, as_batchtopk=batchtopk, abs_topk=abs_topk).train()
    out = m(torch.randn(3, 7, 16))
    assert out.acts.shape == (3, 7, 64) and out.recon.shape == (3, 7, 16)
    torch.testing.assert_close(m.decode(out.acts), out.recon, atol=1e-5, rtol=1e-5)
    l0 = (out.acts != 0).sum(-1)
    assert l0.sum() <= 5 * 21 if batchtopk else (l0 <= 5).all()
    assert (out.acts < 0).any() == abs_topk


def test_decoder_gradient_projection_and_norm():
    torch.manual_seed(0)
    m = TopKSAE(d_in=16, d_sae=64, k=4)
    m(torch.randn(32, 16)).loss.backward()
    m.pre_optimizer_step()
    assert (m.w_dec.grad * m.w_dec).sum(-1).abs().max() < 1e-6
    torch.testing.assert_close(m.w_dec.norm(dim=-1), torch.ones(64))


@pytest.mark.parametrize(
    "kw",
    [dict(), dict(as_batchtopk=True), dict(abs_topk=True, as_batchtopk=True),
     dict(archetypal=True, archetypal_n_candidates=64), dict(loss_normalization="fixed")],
)
def test_folding_and_safetensors_roundtrip(tmp_path, kw):
    torch.manual_seed(0)
    x = synthetic(n=2048)
    m = TopKSAE(d_in=32, d_sae=128, k=6, matryoshka=True, **kw)
    m.initialize_from_data(x)
    assert not math.isclose(m.input_scale.item(), 1.0)
    m.train()
    for _ in range(3):
        m(x)  # sets the BatchTopK threshold
    m.eval()
    ref = m(x)

    loaded = TopKSAE.load(m.save(tmp_path / "sae.safetensors"))
    assert loaded.input_scale.item() == 1.0 and loaded.cfg == m.cfg
    got = loaded(x)
    # Latents exactly on a selection boundary may flip after rescaling (float
    # rounding), so allow a handful of rows to differ.
    for a, b in [(got.recon, ref.recon), (got.acts, ref.acts), (loaded.encode(x), ref.acts)]:
        assert torch.isclose(a, b, atol=1e-4, rtol=1e-4).all(-1).float().mean() > 0.995
    torch.testing.assert_close(got.metrics["nmse"], ref.metrics["nmse"], atol=1e-3, rtol=1e-3)
    if kw.get("archetypal"):  # training state survives folding: same decoder in train mode
        loaded.train()
        torch.testing.assert_close(loaded.decoder_weight(), m.w_dec, atol=1e-4, rtol=1e-4)


def test_batchtopk_threshold_inference():
    torch.manual_seed(0)
    x = synthetic(n=8192)
    m = TopKSAE(d_in=32, d_sae=128, k=8, as_batchtopk=True)
    m.initialize_from_data(x)
    m.train()
    for chunk in x.split(512):
        m(chunk)
    m.eval()
    l0 = (m.encode(x) > 0).sum(-1).float().mean().item()
    assert 4 < l0 < 16, l0


def test_matryoshka_defaults():
    cfg = SAEConfig(d_in=32, d_sae=256, k=8, matryoshka=True)
    assert cfg.matryoshka_prefixes == (4, 16, 64, 256)
    out = TopKSAE(cfg).train()(torch.randn(64, 32))
    assert not torch.isclose(out.metrics["matryoshka_loss"], out.metrics["nmse"])


@pytest.mark.parametrize("abs_topk", [False, True])
def test_aux_loss_only_when_dead(abs_topk):
    torch.manual_seed(0)
    m = TopKSAE(d_in=16, d_sae=64, k=2, dead_tokens_threshold=10, abs_topk=abs_topk).train()
    assert m(torch.randn(8, 16)).metrics["aux_loss"] == 0
    m.tokens_since_active.fill_(100)
    out = m(torch.randn(8, 16))
    assert out.metrics["aux_loss"] > 0
    out.loss.backward()
    assert m.w_enc.grad is not None


def test_archetypal_constraints():
    torch.manual_seed(0)
    x = synthetic(n=4096)
    m = TopKSAE(d_in=32, d_sae=128, k=6, archetypal=True, archetypal_n_candidates=64, archetypal_delta=0.5)
    with pytest.raises(RuntimeError):
        m.train()(x[:8])
    m.initialize_from_data(x)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        m(x[:512]).loss.backward()
        opt.step()
        m.post_optimizer_step()
    w = m.arch_weight
    assert (w >= 0).all() and torch.allclose(w.sum(-1), torch.ones(128), atol=1e-5)
    assert (m.arch_relax.norm(dim=-1) <= 0.5 + 1e-5).all()


def test_kmeans_recovers_clusters():
    centers = torch.tensor([[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]])
    pts = torch.cat([c + 0.1 * torch.randn(200, 2) for c in centers])
    found = kmeans(pts, 3)
    assert torch.cdist(centers, found).min(-1).values.max() < 0.5


def test_k_annealing_schedule(tmp_path):
    sae = TopKSAE(d_in=32, d_sae=256, k=8)
    tr = SAETrainer(sae, tcfg(tmp_path, k_decay_steps=100))
    assert tr._k_at(0) == 256 and tr._k_at(50) == round(256 * 0.5 + 8 * 0.5) and tr._k_at(100) == 8


def test_signum_and_muon_build():
    m = TopKSAE(d_in=16, d_sae=64, k=4, archetypal=True, archetypal_n_candidates=16)
    opts = build_optimizers(m, "muon", 1e-3)
    assert len(opts) == 2 and opts[0][1] is False  # Muon on matrices, no warmup
    assert isinstance(build_optimizers(m, "signum", 1e-3)[0][0], Signum)


def test_layout_roundtrip():
    x = torch.randn(2, 5, 3, 4)
    torch.testing.assert_close(from_tokens(to_tokens(x, "bchw"), "bchw", x), x)
    assert to_tokens(x, "bchw").shape == (2, 12, 5)


@pytest.mark.parametrize(
    "kw,tkw,min_fve",
    [
        (dict(), dict(), 0.8),
        (dict(as_batchtopk=True), dict(), 0.8),
        (dict(matryoshka=True), dict(), 0.8),
        (dict(multi_topk=True), dict(), 0.8),
        (dict(abs_topk=True), dict(), 0.8),
        (dict(archetypal=True, archetypal_n_candidates=256), dict(), 0.75),
        (dict(loss_normalization="fixed"), dict(), 0.8),
        (dict(), dict(optimizer="muon"), 0.8),
        (dict(), dict(optimizer="signum"), 0.75),
        (dict(), dict(k_decay_steps=100), 0.8),
        (dict(), dict(ema_decay=0.99), 0.8),
    ],
)
def test_training_learns(tmp_path, kw, tkw, min_fve):
    torch.manual_seed(0)
    x = synthetic(signed=kw.get("abs_topk", False))
    sae = TopKSAE(d_in=32, d_sae=256, k=8, dead_tokens_threshold=20_000, **kw)
    tr = SAETrainer(sae, tcfg(tmp_path, num_epochs=6, **tkw))
    tr.train(loader(x))
    metrics = tr.evaluate(loader(x))
    assert metrics["eval/fve"] > min_fve, metrics
    assert (tmp_path / "sae.safetensors").exists()


def test_tensorboard_and_metadata(tmp_path):
    x = synthetic(n=8192)
    tr = SAETrainer(TopKSAE(d_in=32, d_sae=128, k=8),
                    tcfg(tmp_path, num_epochs=2, log_every=5, hist_every=10, loggers=("tensorboard",),
                         log_dir=str(tmp_path / "tb")))
    tr.train(loader(x))
    assert any((tmp_path / "tb").rglob("events.out.tfevents*"))
    assert TopKSAE.read_metadata(tmp_path / "sae.safetensors")["step"] == str(tr.step)


def test_outlier_filter():
    x = synthetic(n=4096)
    x[::50] *= 100
    tr = SAETrainer(TopKSAE(d_in=32, d_sae=64, k=4), TrainerConfig(device="cpu"))
    tr.initialize(loader(x, shuffle=False))
    assert tr._prepare(x[:500]).shape[0] == 490


class ToyViT(nn.Module):
    """Patch embedding + high-norm CLS token + token-wise MLP blocks + head on
    the mean over all tokens (so patch tokens affect the output)."""

    def __init__(self, d=32, n_classes=10, patch=4):
        super().__init__()
        self.embed = nn.Conv2d(3, d, patch, patch)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 5)  # high-norm CLS
        self.block1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        self.block2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        self.head = nn.Linear(d, n_classes)

    def forward(self, img):
        t = self.embed(img).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1)
        t = t + self.block1(t)
        t = t + self.block2(t)
        return self.head(t.mean(1))


def test_vision_store_train_and_kl_finetune(tmp_path):
    torch.manual_seed(0)
    vit = ToyViT()
    images = torch.randn(512, 3, 16, 16)
    img_loader = DataLoader(TensorDataset(images, torch.zeros(512)), batch_size=32)

    store = ActivationStore(vit, vit.block1, img_loader, batch_tokens=256, buffer_tokens=2048,
                            exclude_positions=(0,))
    batches = list(store)
    assert all(b.shape == (256, 32) for b in batches)
    assert sum(b.shape[0] for b in batches) > 0.9 * 512 * 16  # 16 patches per image, CLS dropped

    sae = TopKSAE(d_in=32, d_sae=128, k=8, as_batchtopk=True)
    tr = SAETrainer(sae, tcfg(tmp_path, total_tokens=60_000))
    tr.train(store)
    assert tr.evaluate(store, max_batches=5)["eval/fve"] > 0.5

    kl_cfg = KLFinetuneConfig(steps=40, exclude_positions=(0,), log_every=10)
    before = tr.evaluate_downstream(vit, vit.block1, img_loader, kl_cfg)
    tr.finetune_kl(vit, vit.block1, img_loader, kl_cfg)
    after = tr.evaluate_downstream(vit, vit.block1, img_loader, kl_cfg)
    assert after["downstream/kl"] < before["downstream/kl"], (before, after)
    assert all(p.requires_grad for p in vit.parameters())  # requires_grad flags restored
    assert (tmp_path / "sae_kl.safetensors").exists()


def test_conv_layout_kl(tmp_path):
    torch.manual_seed(0)
    cnn = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1),
                        nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 5))
    images = torch.randn(128, 3, 8, 8)
    img_loader = DataLoader(TensorDataset(images), batch_size=16)
    store = ActivationStore(cnn, cnn[1], img_loader, batch_tokens=256, buffer_tokens=1024, layout="bchw")
    sae = TopKSAE(d_in=16, d_sae=64, k=4)
    tr = SAETrainer(sae, tcfg(tmp_path, total_tokens=20_000))
    tr.train(store)
    kl_cfg = KLFinetuneConfig(steps=10, layout="bchw", log_every=5)
    tr.finetune_kl(cnn, cnn[1], img_loader, kl_cfg)
    assert tr.evaluate_downstream(cnn, cnn[1], img_loader, kl_cfg)["downstream/top1_agreement"] > 0.3


def test_patch_kept_tokens_respects_masks():
    act = torch.randn(2, 6, 4)
    mask_fn = lambda batch: batch
    batch = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)  # padding in row 0
    out = patch_kept_tokens(act, torch.zeros_like, "btd", exclude_positions=(0,), token_mask_fn=mask_fn,
                            batch=batch)
    assert torch.equal(out[:, 0], act[:, 0]) and torch.equal(out[0, 4:], act[0, 4:])
    assert (out[0, 1:4] == 0).all() and (out[1, 1:] == 0).all()


def test_streaming_without_total_tokens_raises(tmp_path):
    net = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
    store = ActivationStore(net, net[1], [torch.randn(4, 10, 8) for _ in range(8)], batch_tokens=32,
                            buffer_tokens=128)
    tr = SAETrainer(TopKSAE(d_in=16, d_sae=64, k=4), tcfg(tmp_path))
    with pytest.raises(ValueError, match="total_tokens"):
        tr.train(store)
