"""Tests for caching, analysis and visualisation."""

import matplotlib

matplotlib.use("Agg")

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from innards.sae import analysis as A
from innards.sae import viz
from innards.sae.cache import CachedActivations, cache_activations
from innards.sae.data import ActivationStore
from innards.sae.model import SAE
from innards.sae.trainer import SAETrainer, TrainerConfig


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

    from innards.sae.data import replace_activation

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


def test_analyze_dataset_chunking_is_exact(setup):
    vit, _, loader, _, sae = setup
    full = A.analyze_dataset(sae, vit, vit.block, loader, n_top=5, exclude_positions=(0,))
    chunked = A.analyze_dataset(sae, vit, vit.block, loader, n_top=5, exclude_positions=(0,), chunk_tokens=7)
    torch.testing.assert_close(chunked.stats.frequency, full.stats.frequency)
    torch.testing.assert_close(chunked.stats.max_activation, full.stats.max_activation)
    torch.testing.assert_close(chunked.top.values, full.top.values)
    filled = full.top.values != 0
    assert torch.equal(chunked.top.example[filled], full.top.example[filled])
    assert (full.top.position[~filled] == -1).all()


@pytest.mark.parametrize("bs", [64, 1000])  # 1000 > shard size: a batch spans several shards
def test_cached_activations_order_and_prefetch(tmp_path, bs):
    x = torch.arange(5000 * 3, dtype=torch.float32).view(5000, 3)
    path = cache_activations([x[i : i + 300] for i in range(0, 5000, 300)], tmp_path / "c", shard_tokens=700,
                             dtype=torch.float32)
    ordered = list(CachedActivations(path, batch_tokens=bs, shuffle=False))
    assert all(b.shape == (bs, 3) for b in ordered)
    torch.testing.assert_close(torch.cat(ordered), x[: len(ordered) * bs])

    cached = CachedActivations(path, batch_tokens=bs, seed=1)
    for _ in range(2):  # each epoch serves distinct tokens, in a new order
        rows = torch.cat(list(cached))[:, 0].long() // 3
        assert rows.unique().numel() == rows.numel() == 5000 // bs * bs
    for i, _ in enumerate(cached):  # stopping mid-epoch shuts the loader thread down cleanly
        if i == 1:
            break


def test_interventions(setup):
    from innards.edits import rows, run, steer
    from innards.sae.data import patch_kept_tokens, replace_activation
    from innards.sae.interventions import clamp, mean_ablate, sae_edits, scale, zero_ablate

    vit, images, *_, sae = setup
    x = images[:4]

    def spliced(edit=None, **kw):
        return run(vit, x, sae_edits(sae, "block", edit, exclude_positions=(0,), **kw))[0]

    with torch.no_grad():
        clean = vit(x)
        torch.testing.assert_close(spliced(), clean, atol=1e-5, rtol=1e-5)  # error kept, no edit: unchanged
        recon = spliced(keep_error=False)
        with replace_activation(vit.block, lambda a: patch_kept_tokens(a, lambda x, y: sae.reconstruct(x), exclude_positions=(0,))):
            torch.testing.assert_close(recon, vit(x), atol=1e-5, rtol=1e-5)
        by_module, _ = run(vit, x, sae_edits(sae, vit.block, keep_error=False, exclude_positions=(0,), model=vit))
        torch.testing.assert_close(by_module, recon)

        lat = int(sae.encode(vit.block(vit.embed(x).flatten(2).transpose(1, 2))).abs().sum((0, 1)).argmax())
        outs = {name: spliced(edit) for name, edit in [
            ("zero", zero_ablate(lat)), ("clamp0", clamp([lat], 0.0)), ("scale0", scale(lat, 0.0)),
            ("mean0", mean_ablate(lat, torch.zeros(sae.cfg.d_sae)))]}
        assert not torch.allclose(outs["zero"], clean, atol=1e-4)
        for name in ("clamp0", "scale0", "mean0"):
            torch.testing.assert_close(outs[name], outs["zero"])

        # The block output moves by exactly -a * d at every patch token where the latent fired.
        ref = run(vit, x, read="block")[1]["block"]
        got = run(vit, x, sae_edits(sae, "block", zero_ablate(lat), exclude_positions=(0,)), read="block")[1]["block"]
        a = sae.encode(ref)[..., lat]
        expected = ref - a.unsqueeze(-1) * sae.decoder_weight()[lat]
        torch.testing.assert_close(got[:, 1:], expected[:, 1:], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(got[:, 0], ref[:, 0])

        direction = sae.decoder_weight()[lat]
        st = run(vit, x, {"block": rows(steer(direction, 3.0), exclude=(0,))}, read="block")[1]["block"]
        torch.testing.assert_close(st[:, 1:], ref[:, 1:] + 3.0 * direction)
        torch.testing.assert_close(st[:, 0], ref[:, 0])


def test_latent_cache_matches_analysis(tmp_path, setup):
    from innards.sae.latent_cache import LatentCache, cache_latents

    vit, images, loader, _, sae = setup
    # Small shards and chunks so merging across shards is exercised.
    path = cache_latents(sae, vit, vit.block, loader, tmp_path / "lat", exclude_positions=(0,), shard_tokens=200,
                         chunk_tokens=50, metadata={"hook": "block"})
    cache = LatentCache(path)
    assert len(cache.meta["shards"]) > 3 and cache.meta["metadata"]["hook"] == "block"
    ref = A.analyze_dataset(sae, vit, vit.block, loader, n_top=5, exclude_positions=(0,))
    stats = cache.stats()
    assert stats.n_tokens == ref.stats.n_tokens == 96 * 16
    for f in ("frequency", "mean_activation", "max_activation"):
        torch.testing.assert_close(getattr(stats, f), getattr(ref.stats, f))
    top = cache.top_examples(5)
    torch.testing.assert_close(top.values, ref.top.values)
    filled = ref.top.values != 0
    assert torch.equal(top.example[filled], ref.top.example[filled])
    assert torch.equal(top.position[filled], ref.top.position[filled])

    # Dense reference over the whole dataset: (examples, tokens, d_sae) with CLS zeroed.
    dense = A.encode_inputs(sae, vit, vit.block, images)
    dense[:, 0] = 0
    lat = int(stats.frequency.argmax())
    tokens, values = cache.latent_activations(lat)
    example, position = cache.token_index(tokens)
    torch.testing.assert_close(values, dense[example, position, lat])
    assert values.numel() == int((dense[..., lat] != 0).sum())

    bins = cache.quantile_examples(lat, n_bins=4, per_bin=3)
    assert len(bins) == 4 and all(len(b) == 3 for b in bins)
    mags = [[abs(v) for v, _, _ in b] for b in bins]
    assert all(max(mags[i]) <= min(mags[i + 1]) + 1e-6 for i in range(3))  # bins ordered by magnitude
    for v, e, p in bins[-1]:
        assert dense[e, p, lat].item() == pytest.approx(v, rel=1e-5)

    latents = [0, 3, lat, 10]
    active = (dense.reshape(-1, sae.cfg.d_sae)[:, latents] != 0).float()
    expected = active.T @ active
    torch.testing.assert_close(cache.cooccurrence(latents), expected)
    torch.testing.assert_close(cache.cooccurrence(latents, chunk_tokens=7), expected)
