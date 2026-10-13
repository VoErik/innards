"""Tests for caching, analysis and visualisation."""

import matplotlib

matplotlib.use("Agg")

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sae import analysis as A
from sae import viz
from sae.cache import CachedActivations, cache_activations
from sae.data import ActivationStore
from sae.model import SAE
from sae.trainer import SAETrainer, TrainerConfig


class ToyViT(nn.Module):
    """Patch embedding, CLS token, one attention-free block, mean-pool head."""

    def __init__(self, d=32, n_classes=10):
        super().__init__()
        self.embed = nn.Conv2d(3, d, 4, 4)
        self.cls = nn.Parameter(torch.randn(1, 1, d))
        self.block = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        self.head = nn.Linear(d, n_classes)

    def forward(self, img):
        t = self.embed(img).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1)
        t = t + self.block(t)
        return self.head(t.mean(1))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.manual_seed(0)
    vit = ToyViT()
    images = torch.randn(96, 3, 16, 16)
    loader = DataLoader(TensorDataset(images), batch_size=16, shuffle=False)
    store = ActivationStore(vit, vit.block, loader, batch_tokens=128, buffer_tokens=512, exclude_positions=(0,))
    sae = SAE(d_in=32, d_sae=64, k=4)
    out = tmp_path_factory.mktemp("ck")
    SAETrainer(sae, TrainerConfig(total_tokens=20_000, device="cpu", log_every=10**6, output_dir=str(out))).train(store)
    return vit, images, loader, store, sae.eval()


def test_cache_roundtrip(tmp_path, setup):
    *_, store, _ = setup
    path = cache_activations(store, tmp_path / "cache", shard_tokens=300, metadata={"hook": "block"})
    cached = CachedActivations(path, batch_tokens=100)
    assert cached.meta["metadata"]["hook"] == "block" and cached.d_in == 32
    total = sum(b.shape[0] for b in cached)
    assert total == cached.n_tokens // 100 * 100 or total >= cached.n_tokens - 100
    assert all(b.dtype == torch.bfloat16 for b in cached)
    first = torch.cat(list(CachedActivations(path, batch_tokens=100, shuffle=False)))
    again = torch.cat(list(CachedActivations(path, batch_tokens=100, shuffle=False)))
    torch.testing.assert_close(first, again)


def test_cache_max_tokens(tmp_path, setup):
    *_, store, _ = setup
    path = cache_activations(store, tmp_path / "c2", shard_tokens=256, max_tokens=500)
    assert CachedActivations(path).n_tokens == 500


def test_analyze_dataset(setup):
    vit, images, loader, _, sae = setup
    res = A.analyze_dataset(sae, vit, vit.block, loader, n_top=5, exclude_positions=(0,))
    assert res.stats.n_tokens == 96 * 16
    # mean L0 = sum of frequencies = k for TopK
    assert abs(res.stats.frequency.sum().item() - 4) < 1e-3
    lat = int(res.stats.frequency.argmax())
    entries = res.top.for_latent(lat)
    assert len(entries) == 5 and all(p != 0 for _, _, p in entries)  # CLS never selected
    v, e, p = entries[0]
    acts = A.encode_inputs(sae, vit, vit.block, images[e : e + 1])
    torch.testing.assert_close(acts[0, p, lat], torch.tensor(v), atol=1e-5, rtol=1e-5)
    assert abs(v) == pytest.approx(res.stats.max_activation[lat].item(), rel=1e-5)


def test_dictionary_geometry(setup):
    *_, sae = setup
    v, i = A.max_cosine_similarity(sae, chunk=16)
    assert v.shape == (64,) and (i != torch.arange(64)).all() and (v <= 1 + 1e-5).all()
    nv, ni = A.nearest_latents(sae, 3, n=4)
    assert ni[0] == i[3] and nv[0] == pytest.approx(v[3].item(), abs=1e-5)
    m = A.decoder_similarity_matrix(sae, [1, 2, 3])
    torch.testing.assert_close(m.diagonal(), torch.ones(3), atol=1e-5, rtol=0)
    assert A.encoder_decoder_cosine(sae).shape == (64,)
    assert A.dictionary_embedding(sae).shape == (64, 2)


def test_output_effects(setup):
    vit, *_, sae = setup
    eff = A.output_effects(sae, vit.head.weight, latents=[0, 5], n=3)
    direct = sae.decoder_weight()[5] @ vit.head.weight.T
    assert eff.top_values.shape == (2, 3)
    assert eff.top_values[1, 0].item() == pytest.approx(direct.max().item(), rel=1e-5)


def test_latent_attribution_matches_ablation(setup):
    vit, images, *_, sae = setup
    x = images[:1]
    target_fn = lambda out, _: out[:, 3]
    attr = A.latent_attribution(sae, vit, vit.block, x, target_fn, exclude_positions=(0,))
    # Splicing with the error term reproduces the clean output exactly.
    torch.testing.assert_close(attr.target, vit(x)[:, 3].detach(), atol=1e-5, rtol=1e-5)
    # The model is linear after the block's residual add, so attribution
    # equals the true effect of zero-ablating a latent.
    vals, lats = attr.top_latents(0, n=1)
    lat = int(lats[0])

    def ablate(act):
        tok = act.clone()
        acts = sae.encode(tok[:, 1:])
        err = tok[:, 1:] - sae.decode(acts)
        acts[..., lat] = 0
        tok[:, 1:] = sae.decode(acts) + err
        return tok

    from sae.data import replace_activation

    with torch.no_grad(), replace_activation(vit.block, ablate):
        ablated = vit(x)[:, 3]
    assert (attr.target - ablated).item() == pytest.approx(vals[0].item(), rel=1e-3, abs=1e-5)


def test_patch_maps_and_input_attribution(setup):
    vit, images, *_, sae = setup
    maps = A.latent_patch_maps(sae, vit, vit.block, images[:2], [0, 1], exclude_positions=(0,))
    assert maps.shape == (2, 2, 4, 4)
    single = A.latent_patch_maps(sae, vit, vit.block, images[:2], 1, exclude_positions=(0,))
    torch.testing.assert_close(single, maps[:, 1])
    sal = A.input_attribution(sae, vit, vit.block, images[:2], latent=0, exclude_positions=(0,))
    assert sal.shape == images[:2].shape and sal.abs().sum() > 0
    assert all(p.requires_grad for p in vit.parameters())


def test_viz_smoke(setup):
    vit, images, loader, _, sae = setup
    res = A.analyze_dataset(sae, vit, vit.block, loader, n_top=4, exclude_positions=(0,))
    viz.plot_latent_density(res.stats.frequency)
    viz.plot_max_cosine_similarity(A.max_cosine_similarity(sae)[0])
    viz.plot_dictionary_map(A.dictionary_embedding(sae), color=res.stats.frequency, highlight=[1, 2])
    viz.plot_similarity_matrix(A.decoder_similarity_matrix(sae, [0, 1, 2]), [0, 1, 2])
    eff = A.output_effects(sae, vit.head.weight, latents=[0], n=3)
    viz.plot_output_effects(eff.top_values[0], eff.top_ids[0], eff.bottom_values[0], eff.bottom_ids[0],
                            labels=[f"class {i}" for i in range(10)])
    attr = A.latent_attribution(sae, vit, vit.block, images[:1], lambda o, _: o[:, 0], exclude_positions=(0,))
    v, l = attr.top_latents(0, 5)
    viz.plot_attribution_bars(v, l, error=attr.error_attribution[0].sum().item())
    imgs = [viz.to_display_image(images[i]) for i in range(3)]
    maps = A.latent_patch_maps(sae, vit, vit.block, images[:3], 0, exclude_positions=(0,))
    viz.plot_top_images(imgs, maps, values=[1.0, 2.0, 3.0], suptitle="latent 0")
    html = viz.text_examples_html([(["a", "b<", "c"], [0.0, 1.0, -0.5], 1)], title="latent 0", context=1)
    assert "b&lt;" in html and "rgba(220,60,50,1.000)" in html


def test_latent_attribution_token_mask(setup):
    vit, images, *_, sae = setup
    x = images[:2]
    mask_fn = lambda batch: torch.arange(17)[None].expand(batch.shape[0], -1) != 5  # drop patch 5
    attr = A.latent_attribution(sae, vit, vit.block, x, lambda o, _: o[:, 3], exclude_positions=(0,),
                                token_mask_fn=mask_fn)
    assert attr.acts.shape == (2, 17, sae.cfg.d_sae)
    assert not attr.keep[:, 0].any() and not attr.keep[:, 5].any() and attr.keep[:, 1].all()
    assert (attr.acts[:, 5] == 0).all() and (attr.error_attribution[:, 0] == 0).all()
    torch.testing.assert_close(attr.target, vit(x)[:, 3].detach(), atol=1e-5, rtol=1e-5)
