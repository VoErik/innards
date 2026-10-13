"""SAELens format round trips; the cross-check against SAELens itself runs when ``sae_lens`` is installed."""

import json

import pytest
import torch

from sae.interop import load_saelens, save_saelens
from sae.model import SAE

CASES = [
    dict(activation="topk", k=6),
    dict(activation="topk", k=6, abs_topk=True),
    dict(activation="batchtopk", k=6),
    dict(activation="jumprelu"),
    dict(activation="relu"),
    dict(activation="topk", k=6, decoder="archetypal", archetypal_n_candidates=32),
    dict(activation="relu", transcoder=True, d_out=24),
    dict(activation="jumprelu", transcoder=True, d_out=24, skip_connection=True),
    dict(activation="batchtopk", k=6, transcoder=True, d_out=24, skip_connection=True),
]


def _trained(kw, seed=0):
    """An SAE with non-trivial weights, scales, thresholds and skip connection."""
    torch.manual_seed(seed)
    x = torch.randn(2048, 32) * 3 + 1
    y = torch.tanh(x[:, :24]) * 5 if kw.get("transcoder") else None
    m = SAE(d_in=32, d_sae=128, **kw)
    m.initialize_from_data(x, y)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    m.post_optimizer_step()
    m.train()
    for i in range(0, 2048, 256):
        m(x[i : i + 256], target=None if y is None else y[i : i + 256])  # BatchTopK threshold
    return m.eval(), x, y


def _ref(m, x):
    return m.encode(x), m.reconstruct(x)


@pytest.mark.parametrize("kw", CASES)
def test_saelens_roundtrip(tmp_path, kw):
    m, x, _ = _trained(kw)
    path = save_saelens(m, tmp_path / "sae", metadata={"model_name": "toy", "hook_name": "blocks.0.hook_resid_post"})
    cfg = json.loads((path / "cfg.json").read_text())
    assert cfg["metadata"]["hook_name"] == "blocks.0.hook_resid_post"
    loaded, raw = load_saelens(path)
    assert raw["architecture"] == cfg["architecture"]
    for a, b in zip(_ref(loaded, x), _ref(m, x)):
        # latents exactly on a selection boundary may flip after rescaling
        assert torch.isclose(a, b, atol=1e-4, rtol=1e-4).all(-1).float().mean() > 0.995


def test_unsupported_exports(tmp_path):
    with pytest.raises(ValueError, match="transcoder"):
        save_saelens(SAE(d_in=8, d_sae=32, k=4, transcoder=True), tmp_path / "a")
    with pytest.raises(ValueError, match="threshold"):
        save_saelens(SAE(d_in=8, d_sae=32, k=4, activation="batchtopk"), tmp_path / "b")


def test_import_absorbs_bias_and_decoder_norms(tmp_path):
    """A hand-written SAELens 'standard' SAE without pre-encoder bias and with non-unit decoder norms."""
    torch.manual_seed(0)
    from safetensors.torch import save_file

    w_enc, b_enc = torch.randn(16, 64), torch.randn(64) * 0.1
    w_dec, b_dec = torch.randn(64, 16) * 3, torch.randn(16)
    save_file({"W_enc": w_enc, "b_enc": b_enc, "W_dec": w_dec, "b_dec": b_dec}, str(tmp_path / "sae_weights.safetensors"))
    (tmp_path / "cfg.json").write_text(json.dumps({"architecture": "standard", "d_in": 16, "d_sae": 64,
                                                   "apply_b_dec_to_input": False, "metadata": {}}))
    sae, _ = load_saelens(tmp_path)
    x = torch.randn(100, 16)
    acts = torch.relu(x @ w_enc + b_enc)
    torch.testing.assert_close(sae.reconstruct(x), acts @ w_dec + b_dec, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(sae.decoder_weight().norm(dim=-1), torch.ones(64))


@pytest.mark.parametrize("kw", CASES[:5] + CASES[6:])
def test_matches_saelens_library(tmp_path, kw):
    sae_lens = pytest.importorskip("sae_lens")
    m, x, _ = _trained(kw)
    path = save_saelens(m, tmp_path / "sae")
    theirs = sae_lens.SAE.load_from_disk(str(path)).eval()
    with torch.no_grad():
        acts = theirs.encode(x)
        recon = theirs.decode(acts) if not kw.get("skip_connection") else theirs.forward(x)
    ours_acts, ours_recon = _ref(m, x)
    assert torch.isclose(acts, ours_acts, atol=1e-4, rtol=1e-4).all(-1).float().mean() > 0.995
    assert torch.isclose(recon, ours_recon, atol=1e-4, rtol=1e-4).all(-1).float().mean() > 0.995
    # and back: SAELens' own save is read by load_saelens
    theirs.save_model(tmp_path / "again")
    back, _ = load_saelens(tmp_path / "again")
    assert torch.isclose(back.reconstruct(x), ours_recon, atol=1e-4, rtol=1e-4).all(-1).float().mean() > 0.995
