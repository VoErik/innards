"""Tests for caching, analysis and visualisation."""

import matplotlib

matplotlib.use("Agg")

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from innards.sae import analysis as A
from innards.extract import RowStream
from innards.store import RowLoader, write_rows
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
    store = RowStream(vit, loader, vit.block, batch_rows=128, buffer_rows=512, exclude=(0,))
    sae = SAE(d_in=32, d_sae=64, k=4)
    out = tmp_path_factory.mktemp("ck")
    SAETrainer(sae, TrainerConfig(total_tokens=20_000, device="cpu", log_every=10**6, output_dir=str(out))).train(store)
    return vit, images, loader, store, sae.eval()


def test_cache_roundtrip(tmp_path, setup):
    *_, store, _ = setup
    path = write_rows(store, tmp_path / "cache", dtype=torch.bfloat16, shard_rows=300, metadata={"hook": "block"})
    cached = RowLoader(path, batch_rows=100)
    assert cached.meta["metadata"]["hook"] == "block" and cached.d_in == 32
    total = sum(b.shape[0] for b in cached)
    assert total == cached.n_rows // 100 * 100 or total >= cached.n_rows - 100
    assert all(b.dtype == torch.bfloat16 for b in cached)
    first = torch.cat(list(RowLoader(path, batch_rows=100, shuffle=False)))
    again = torch.cat(list(RowLoader(path, batch_rows=100, shuffle=False)))
    torch.testing.assert_close(first, again)


def test_cache_max_rows(tmp_path, setup):
    *_, store, _ = setup
    path = write_rows(store, tmp_path / "c2", shard_rows=256, max_rows=500)
    assert RowLoader(path).n_rows == 500


def test_dictionary_geometry(setup):
    from innards.analysis.geometry import nearest

    *_, sae = setup
    atoms = torch.nn.functional.normalize(sae.decoder_weight().detach(), dim=-1)
    v, i = nearest(atoms, k=1, chunk=16)
    assert v.shape == (64, 1) and (i[:, 0] != torch.arange(64)).all() and (v <= 1 + 1e-5).all()
    nv, ni = nearest(atoms, 3, k=4)
    assert ni[0] == i[3, 0] and nv[0] == pytest.approx(v[3, 0].item(), abs=1e-5)
    assert A.encoder_decoder_cosine(sae).shape == (64,)


def test_output_effects(setup):
    from innards.analysis.interp import readout_effect, top_outputs

    vit, *_, sae = setup
    top_v, top_i, bot_v, bot_i = top_outputs(readout_effect(sae.decoder_weight()[[0, 5]], vit.head.weight), n=3)
    direct = sae.decoder_weight()[5] @ vit.head.weight.T
    assert top_v.shape == bot_i.shape == (2, 3)
    assert top_v[1, 0].item() == pytest.approx(direct.max().item(), rel=1e-5)
    assert bot_v[1, 0].item() == pytest.approx(direct.min().item(), rel=1e-5)


def test_latent_attribution_matches_ablation(setup):
    vit, images, *_, sae = setup
    x = images[:1]
    target_fn = lambda out, _: out[:, 3]
    attr = A.latent_attribution(sae, vit, vit.block, x, target_fn, exclude=(0,))
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

    from _hooks import replace_output

    with torch.no_grad(), replace_output(vit.block, ablate):
        ablated = vit(x)[:, 3]
    assert (attr.target - ablated).item() == pytest.approx(vals[0].item(), rel=1e-3, abs=1e-5)


def test_latent_fn_with_interp(setup):
    from innards.analysis import interp
    from innards.edits import run

    vit, images, *_, sae = setup
    tokens = run(vit, images[:2], read="block")[1]["block"]
    torch.testing.assert_close(A.latent_fn(sae)(tokens), sae.encode(tokens))
    torch.testing.assert_close(A.encode_inputs(sae, vit, "block", images[:2]), sae.encode(tokens))
    maps = interp.spatial_map(tokens, A.latent_fn(sae, [0, 1]), (4, 4), exclude=(0,))
    assert maps.shape == (2, 2, 4, 4)
    single = interp.spatial_map(tokens, A.latent_fn(sae, 1), (4, 4), exclude=(0,))
    torch.testing.assert_close(single, maps[:, 1])
    values, examples, positions = interp.top_k(tokens, A.latent_fn(sae, 1), k=3)
    assert values[0] == sae.encode(tokens)[..., 1].max()
    sal = interp.input_attribution(vit, images[:2], "block", A.latent_fn(sae, 0, pre=True), exclude=(0,))
    assert sal.shape == images[:2].shape and sal.abs().sum() > 0
    assert all(p.requires_grad for p in vit.parameters())


def test_viz_smoke(setup):
    from innards.analysis import plot
    from innards.analysis.dashboard import text_examples_html
    from innards.analysis.units import unit_stats
    from innards.analysis.geometry import nearest
    from innards.analysis.interp import readout_effect, top_outputs

    vit, images, loader, _, sae = setup
    stats, _ = unit_stats(vit, loader, vit.block, A.latent_fn(sae), n_top=4, exclude=(0,))
    atoms = torch.nn.functional.normalize(sae.decoder_weight().detach(), dim=-1)
    plot.histogram(stats.frequency, log10=True, xlabel="log10 firing frequency", title="density")
    plot.histogram(nearest(atoms, k=1)[0], xlabel="max cosine similarity")
    plot.embedding(atoms, values=stats.frequency, annotate=[1, 2], metric="cosine")
    plot.similarity_matrix(atoms[:3] @ atoms[:3].T, labels=["0", "1", "2"], signed=True)
    tv, ti, bv, bi = top_outputs(readout_effect(atoms[[0]], vit.head.weight), n=3)
    plot.bars(torch.cat([tv[0], bv[0].flip(0)]), [f"class {int(i)}" for i in torch.cat([ti[0], bi[0].flip(0)])])
    html = text_examples_html([(["a", "b<", "c"], [0.0, 1.0, -0.5], 1)], title="latent 0", context=1)
    assert "b&lt;" in html and "rgba(208,59,59,1.000)" in html


def test_latent_attribution_token_mask(setup):
    vit, images, *_, sae = setup
    x = images[:2]
    mask_fn = lambda batch: torch.arange(17)[None].expand(batch.shape[0], -1) != 5  # drop patch 5
    attr = A.latent_attribution(sae, vit, vit.block, x, lambda o, _: o[:, 3], exclude=(0,),
                                mask_fn=mask_fn)
    assert attr.acts.shape == (2, 17, sae.cfg.d_sae)
    assert not attr.keep[:, 0].any() and not attr.keep[:, 5].any() and attr.keep[:, 1].all()
    assert (attr.acts[:, 5] == 0).all() and (attr.error_attribution[:, 0] == 0).all()
    torch.testing.assert_close(attr.target, vit(x)[:, 3].detach(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("bs", [64, 1000])  # 1000 > shard size: a batch spans several shards
def test_cached_activations_order_and_prefetch(tmp_path, bs):
    x = torch.arange(5000 * 3, dtype=torch.float32).view(5000, 3)
    path = write_rows([x[i : i + 300] for i in range(0, 5000, 300)], tmp_path / "c", shard_rows=700,
                             dtype=torch.float32)
    ordered = list(RowLoader(path, batch_rows=bs, shuffle=False))
    assert all(b.shape == (bs, 3) for b in ordered)
    torch.testing.assert_close(torch.cat(ordered), x[: len(ordered) * bs])

    cached = RowLoader(path, batch_rows=bs, seed=1)
    for _ in range(2):  # each epoch serves distinct tokens, in a new order
        rows = torch.cat(list(cached))[:, 0].long() // 3
        assert rows.unique().numel() == rows.numel() == 5000 // bs * bs
    for i, _ in enumerate(cached):  # stopping mid-epoch shuts the loader thread down cleanly
        if i == 1:
            break


def test_interventions(setup):
    from _hooks import replace_output
    from innards.edits import rows, run, steer
    from innards.edits import clamp, mean_ablate, scale, zero
    from innards.sae.interventions import sae_edits

    vit, images, *_, sae = setup
    x = images[:4]

    def spliced(edit=None, **kw):
        return run(vit, x, sae_edits(sae, "block", edit, exclude=(0,), **kw))[0]

    with torch.no_grad():
        clean = vit(x)
        torch.testing.assert_close(spliced(), clean, atol=1e-5, rtol=1e-5)  # error kept, no edit: unchanged
        recon = spliced(keep_error=False)
        with replace_output(vit.block, lambda a: torch.cat([a[:, :1], sae.reconstruct(a[:, 1:])], 1)):
            torch.testing.assert_close(recon, vit(x), atol=1e-5, rtol=1e-5)
        by_module, _ = run(vit, x, sae_edits(sae, vit.block, keep_error=False, exclude=(0,), model=vit))
        torch.testing.assert_close(by_module, recon)

        lat = int(sae.encode(vit.block(vit.embed(x).flatten(2).transpose(1, 2))).abs().sum((0, 1)).argmax())
        outs = {name: spliced(edit) for name, edit in [
            ("zero", zero(lat)), ("clamp0", clamp([lat], 0.0)), ("scale0", scale(lat, 0.0)),
            ("mean0", mean_ablate(torch.zeros(sae.cfg.d_sae), lat))]}
        assert not torch.allclose(outs["zero"], clean, atol=1e-4)
        for name in ("clamp0", "scale0", "mean0"):
            torch.testing.assert_close(outs[name], outs["zero"])

        # The block output moves by exactly -a * d at every patch token where the latent fired.
        ref = run(vit, x, read="block")[1]["block"]
        got = run(vit, x, sae_edits(sae, "block", zero(lat), exclude=(0,)), read="block")[1]["block"]
        a = sae.encode(ref)[..., lat]
        expected = ref - a.unsqueeze(-1) * sae.decoder_weight()[lat]
        torch.testing.assert_close(got[:, 1:], expected[:, 1:], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(got[:, 0], ref[:, 0])

        direction = sae.decoder_weight()[lat]
        st = run(vit, x, {"block": rows(steer(direction, 3.0), exclude=(0,))}, read="block")[1]["block"]
        torch.testing.assert_close(st[:, 1:], ref[:, 1:] + 3.0 * direction)
        torch.testing.assert_close(st[:, 0], ref[:, 0])
