"""HTML views: payload contents and renderers (the pages themselves are checked in a browser by hand)."""

import json
import re

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from innards.analysis.dashboard import (
    dashboard_html,
    image_renderer,
    save_dashboard,
    text_examples_html,
    text_renderer
)
from innards.analysis.units import cache_stats, cache_top, cache_units, entries
from innards.sae.analysis import latent_fn
from innards.sae.dashboard import graph_html
from innards.sae.graph import attribution_graph, sae_site
from innards.sae.model import SAE
from test_graph import ToyMixNet, _transcoders


def payload(page: str) -> dict:
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


class TokenNet(nn.Module):
    def __init__(self, vocab=20, d=16):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, d)
        self.block = nn.Sequential(nn.Linear(d, d), nn.GELU())

    def forward(self, ids):
        return self.block(self.embed(ids))


def test_latent_dashboard_payload(tmp_path):
    net = TokenNet()
    ids = torch.randint(0, 20, (40, 12), generator=torch.Generator().manual_seed(0))
    loader = DataLoader(TensorDataset(ids), batch_size=8)
    with torch.no_grad():
        acts = net.block(net.embed(ids)).reshape(-1, 16)
    sae = SAE(d_in=16, d_sae=32, k=3)
    sae.initialize_from_data(acts)
    sae.eval()
    cache = cache_units(net, loader, net.block, tmp_path / "lat", latent_fn(sae), exclude=(0,))
    words = [[f"w{i}</script>" for i in row.tolist()] for row in ids]  # hostile token text must stay escaped
    page = dashboard_html(cache, text_renderer(words, context=3, sep=" "), n_top=5, directions=sae.decoder_weight(),
                          noun="latent", readout=torch.randn(7, 16), output_labels=[f"class{i}" for i in range(7)],
                                 descriptions={0: "first latent"}, title="Toy <dash>")
    assert page.count("</script>") == 2 and "<title>Toy &lt;dash&gt;</title>" in page
    data = payload(page)
    top, stats = cache_top(cache, 5), cache_stats(cache)
    alive = int((stats.frequency > 0).sum())
    assert len(data["units"]) == alive and data["sep"] == " " and data["n_tokens"] == 40 * 11
    assert data["noun"] == "latent" and data["n_units"] == 32
    with torch.no_grad():
        dense = sae.encode(net.block(net.embed(ids)))  # (examples, tokens, d_sae)
    for entry in data["units"]:
        lat = entry["id"]
        assert [e["v"] for e in entry["top"]] == [round(v, 4) for v, _, _ in entries(top, lat)]
        for e in entry["top"] + [x for b in entry["bins"] for x in b]:  # shown values are the true activations
            assert e["pos"] > 0 and abs(dense[e["ex"], e["pos"], lat].item() - e["v"]) < 1e-3
        first = entry["top"][0]
        t = first["text"]
        assert t["t"][t["c"]] == words[first["ex"]][first["pos"]] and len(t["t"]) <= 7
        assert any(i == t["c"] for i, _ in t["a"])  # the shown token's own activation is included
        assert len(entry["neighbors"]) == 8 and len(entry["effects"]["top"]) == 7  # only 7 outputs
        assert sum(len(b) for b in entry["bins"]) > 0
    assert all(e["desc"] == ("first latent" if e["id"] == 0 else "") for e in data["units"])
    # standalone HTML of the text renderer
    html = text_renderer(words, context=2)(0, 3, {3: 2.0, 4: -1.0}, 2.0)
    assert "tok cur" in html and "&lt;/script&gt;" in html and "rgba(42,120,214" in html
    assert save_dashboard(tmp_path / "d.html", cache, max_units=3).exists()
    assert len(payload((tmp_path / "d.html").read_text())["units"]) == 3


def test_dashboard_per_example_and_neurons(tmp_path):
    net = TokenNet()
    ids = torch.randint(0, 20, (40, 12), generator=torch.Generator().manual_seed(1))
    loader = DataLoader(TensorDataset(ids), batch_size=8)
    cache = cache_units(net, loader, net.block, tmp_path / "neurons", threshold=0.2)  # the 16 dims as units
    data = payload(dashboard_html(cache, n_top=5, per_example=True))
    assert data["noun"] == "unit" and data["n_units"] == 16
    _, top = cache_stats(cache), cache_top(cache, 5, per_example=True)
    for entry in data["units"]:
        assert len({e["ex"] for e in entry["top"]}) == len(entry["top"])  # one entry per example
        assert [e["v"] for e in entry["top"]] == [round(v, 4) for v, _, _ in entries(top, entry["id"])]  # ties: values
        assert entry["neighbors"] == [] and entry["effects"] is None


def test_text_examples_html():
    html = text_examples_html([(["a", "b<", "c"], [0.0, 1.0, -0.5], 1)], title="latent 0", context=1)
    assert "b&lt;" in html and "rgba(208,59,59,1.000)" in html and "rgba(42,120,214,0.500)" in html


def test_image_renderer():
    images = torch.rand(2, 3, 16, 16)
    html = image_renderer(images, grid=(4, 4), n_prefix=1)(1, 6, {6: 3.0, 2: 1.0}, 3.0)
    assert html.startswith('<img class="ex-img"') and "data:image/" in html
    import base64, io
    from PIL import Image
    data = base64.b64decode(html.split("base64,")[1].split('"')[0])
    assert Image.open(io.BytesIO(data)).size == (112, 112)  # stored at display size


def test_graph_html_payload():
    net = ToyMixNet()
    tcs = _transcoders(net, torch.randn(64, 5, 8), activation="topk", k=4)
    sites = [sae_site(tc, net.inp[i], net.mlps[i], name=f"tc{i}") for i, tc in enumerate(tcs)]
    graph = attribution_graph(net, sites, torch.randn(1, 5, 8), lambda out, _: out[:, 0], max_nodes=30).prune(0.9)
    page = graph_html(graph, tokens=["a", "b", "c", "d", "e"], metric_label="out[0]",
                      descriptions=lambda n: f"latent {n.latent}")
    data = payload(page)
    assert data["sites"] == [s for s in ("tc0", "tc1", "tc2") if any(n.site == s for n in graph.nodes)]
    assert len(data["nodes"]) == len(graph.nodes) and data["tokens"] == ["a", "b", "c", "d", "e"]
    assert len(data["edges"]) == int((graph.adjacency != 0).sum())
    assert all(n["desc"] == f"latent {n['latent']}" for n in data["nodes"] if n["kind"] == "latent")
    assert data["metric"] == graph.metric
