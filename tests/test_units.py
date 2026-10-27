"""Tests for innards.analysis.units: one pass and the disk cache, against dense references."""

import pytest
import torch

from innards.analysis.units import (
    cache_meta,
    cache_stats,
    cache_top,
    cache_units,
    cooccurrence,
    entries,
    quantile_examples,
    unit_activations,
    unit_stats
)
from innards.sae import analysis as A
from test_analysis import setup  # noqa: F401  (fixture: toy ViT, images, loader, row stream, SAE)


@pytest.fixture(scope="module")
def dense(setup):  # noqa: F811
    """SAE latents at every token ``(examples, tokens, d_sae)``, CLS zeroed."""
    vit, images, *_, sae = setup
    d = A.encode_inputs(sae, vit, vit.block, images)
    d[:, 0] = 0
    return d


def test_unit_stats_on_sae_latents(setup, dense):  # noqa: F811
    vit, images, loader, _, sae = setup
    stats, top = unit_stats(vit, loader, vit.block, A.latent_fn(sae), n_top=5, exclude=(0,))
    assert stats.n_tokens == 96 * 16
    assert abs(stats.frequency.sum().item() - 4) < 1e-3  # mean L0 = k for TopK
    lat = int(stats.frequency.argmax())
    found = entries(top, lat)
    assert len(found) == 5 and all(p != 0 for _, _, p in found)  # CLS never selected
    for v, e, p in found:
        assert dense[e, p, lat].item() == pytest.approx(v, rel=1e-5)
    assert abs(found[0][0]) == pytest.approx(stats.max[lat].item(), rel=1e-5)
    ref = dense[..., lat].abs().flatten().topk(5).values
    torch.testing.assert_close(top.values[lat].abs(), ref)
    active = dense[:, 1:] != 0
    torch.testing.assert_close(stats.mean, dense[:, 1:].abs().sum((0, 1)) / active.sum((0, 1)).clamp_min(1))


def test_unit_stats_chunking_is_exact(setup):  # noqa: F811
    vit, _, loader, _, sae = setup
    full = unit_stats(vit, loader, vit.block, A.latent_fn(sae), n_top=5, exclude=(0,))
    for per_example in (False, True):
        a = unit_stats(vit, loader, vit.block, A.latent_fn(sae), n_top=5, exclude=(0,), per_example=per_example)
        b = unit_stats(vit, loader, vit.block, A.latent_fn(sae), n_top=5, exclude=(0,), per_example=per_example,
                       chunk_tokens=7)
        for x, y in zip(a[0][:3], b[0][:3]):
            torch.testing.assert_close(x, y)
        torch.testing.assert_close(a[1].values, b[1].values)
        filled = a[1].values != 0
        assert torch.equal(a[1].example[filled], b[1].example[filled])
    assert (full[1].position[full[1].values == 0] == -1).all()


def test_per_example_ranks_examples_by_their_peak(setup, dense):  # noqa: F811
    vit, _, loader, _, sae = setup
    _, top = unit_stats(vit, loader, vit.block, A.latent_fn(sae), n_top=6, per_example=True, exclude=(0,))
    for lat in range(0, 64, 7):
        peak, where = dense[..., lat].abs().max(1)  # per example
        n = int((peak > 0).sum().clamp(max=6))
        found = entries(top, lat)
        assert len(found) == n and len({e for _, e, _ in found}) == n
        torch.testing.assert_close(top.values[lat, :n].abs(), peak.topk(n).values)
        for v, e, p in found:
            assert p == int(where[e]) and dense[e, p, lat].item() == pytest.approx(v, rel=1e-5)


def test_neurons_with_threshold(setup):  # noqa: F811
    vit, images, loader, *_ = setup
    with torch.no_grad():
        tokens = vit.block(torch.cat([vit.cls.expand(96, -1, -1), vit.embed(images).flatten(2).transpose(1, 2)], 1))
    tokens = tokens[:, 1:]
    stats, top = unit_stats(vit, loader, vit.block, threshold=0.1, n_top=3, exclude=(0,))  # units: the 32 dims
    torch.testing.assert_close(stats.frequency, (tokens.abs() > 0.1).float().mean((0, 1)))
    torch.testing.assert_close(stats.max, tokens.abs().amax((0, 1)))
    v, e, p = entries(top, 5)[0]
    assert tokens[e, p - 1, 5].item() == pytest.approx(v, rel=1e-5)
    projected, _ = unit_stats(vit, loader, vit.block, lambda t: t @ torch.eye(32)[[5, 7]].T, exclude=(0,))
    torch.testing.assert_close(projected.max, stats.max[[5, 7]])


def test_cache_matches_one_pass(tmp_path, setup, dense):  # noqa: F811
    vit, _, loader, _, sae = setup
    # Small shards and chunks so merging across shards is exercised.
    path = cache_units(vit, loader, vit.block, tmp_path / "lat", A.latent_fn(sae), exclude=(0,), shard_tokens=200,
                       chunk_tokens=50, metadata={"hook": "block"})
    meta = cache_meta(path)
    assert len(meta["shards"]) > 3 and meta["metadata"]["hook"] == "block" and meta["n_units"] == 64
    for per_example in (False, True):
        ref_stats, ref_top = unit_stats(vit, loader, vit.block, A.latent_fn(sae), n_top=5, exclude=(0,),
                                        per_example=per_example)
        stats, top = cache_stats(path), cache_top(path, 5, per_example=per_example)
        assert stats.n_tokens == ref_stats.n_tokens == 96 * 16
        for x, y in zip(stats[:3], ref_stats[:3]):
            torch.testing.assert_close(x, y)
        torch.testing.assert_close(top.values, ref_top.values)
        filled = ref_top.values != 0
        assert torch.equal(top.example[filled], ref_top.example[filled])
        assert torch.equal(top.position[filled], ref_top.position[filled])

    lat = int(stats.frequency.argmax())
    example, position, values = unit_activations(path, lat)
    torch.testing.assert_close(values, dense[example, position, lat])
    assert values.numel() == int((dense[..., lat] != 0).sum())

    bins = quantile_examples(path, lat, n_bins=4, per_bin=3)
    assert len(bins) == 4 and all(len(b) == 3 for b in bins)
    mags = [[abs(v) for v, _, _ in b] for b in bins]
    assert all(max(mags[i]) <= min(mags[i + 1]) + 1e-6 for i in range(3))  # bins ordered by magnitude
    for v, e, p in bins[-1]:
        assert dense[e, p, lat].item() == pytest.approx(v, rel=1e-5)

    units = [0, 3, lat, 10]
    active = (dense[:, 1:].reshape(-1, 64)[:, units] != 0).float()
    expected = active.T @ active
    torch.testing.assert_close(cooccurrence(path, units), expected)
    torch.testing.assert_close(cooccurrence(path, units, chunk_tokens=7), expected)
