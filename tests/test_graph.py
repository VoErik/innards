"""Attribution graphs: exactness on a model that is linear between the replaced MLPs."""

import pytest
import torch
from torch import nn

from sae.clt import CrossLayerTranscoder, splice_clt
from sae.data import capture_activation, splice
from sae.graph import attribution_graph, clt_site, sae_site
from sae.interventions import splice_sae
from sae.model import SAE


class ToyMixNet(nn.Module):
    """Residual stream; each block: causal token mixing, then an MLP read through an Identity hook point."""

    def __init__(self, d=16, n_layers=3, n_out=4):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Linear(8, d)
        self.mix = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n_layers))
        self.inp = nn.ModuleList(nn.Identity() for _ in range(n_layers))
        self.mlps = nn.ModuleList(nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
                                  for _ in range(n_layers))
        self.head = nn.Linear(d, n_out)

    def forward(self, x):
        h = self.embed(x)
        t = h.shape[1]
        causal = torch.tril(torch.ones(t, t)) / torch.arange(1, t + 1)[:, None]  # causal mean over positions
        for mix, inp, mlp in zip(self.mix, self.inp, self.mlps):
            h = h + mix(causal @ h)
            h = h + mlp(inp(h))
        return self.head(h)[:, -1]  # (batch, n_out): readout at the last position


def _pairs(net, layer, inputs):
    with torch.no_grad(), capture_activation(net.inp[layer]) as a, capture_activation(net.mlps[layer]) as b:
        net(inputs)
    return a["act"].reshape(-1, 16), b["act"].reshape(-1, 16)


def _transcoders(net, inputs, **kw):
    tcs = []
    for layer in range(3):
        tc = SAE(d_in=16, d_sae=64, transcoder=True, **kw)
        tc.initialize_from_data(*_pairs(net, layer, inputs))
        tcs.append(tc.eval())
    return tcs


def _pre(net, sae, layer, x, run_ctx):
    """Pre-activations of ``sae`` at layer ``layer``'s MLP input, under ``run_ctx``."""
    with torch.no_grad(), run_ctx, capture_activation(net.inp[layer]) as cap:
        net(x)
        return sae.pre_activations(cap["act"])[0]  # (tokens, d_sae)


# Edges are *direct* effects: intermediate latents and errors stay fixed. Re-running the model after
# an edit recomputes intermediate sites, so only edges between adjacent sites can be checked that way;
# cross-site writes are covered exactly by the completeness checks (the metric is linear in all leaves).


METRIC = lambda out, _: out[:, 1]


@pytest.mark.parametrize("kw", [dict(activation="topk", k=4), dict(activation="relu"),
                                dict(activation="topk", k=4, skip_connection=True)])
def test_transcoder_graph_is_exact(kw):
    net = ToyMixNet()
    torch.manual_seed(1)
    tcs = _transcoders(net, torch.randn(64, 5, 8), **kw)
    if kw.get("skip_connection"):
        for tc in tcs:
            nn.init.normal_(tc.w_skip, std=0.05)
    x = torch.randn(1, 5, 8)
    sites = [sae_site(tc, net.inp[i], net.mlps[i], name=f"tc{i}") for i, tc in enumerate(tcs)]
    graph = attribution_graph(net, sites, x, METRIC, max_nodes=10_000)
    clean = METRIC(net(x), None).item()
    assert graph.metric == pytest.approx(clean, abs=1e-5)  # the spliced model is the clean model

    # Completeness: the metric is linear in the leaves, so attributions sum to m - m(all leaves = 0).
    def zero_leaves(i):  # latents and error 0: the MLP output becomes the bias (+ skip of the actual input)
        return splice(net.inp[i], lambda x_, y_: tcs[i].decode(torch.zeros(len(x_), 64)) + tcs[i].skip(x_),
                      net.mlps[i])
    with torch.no_grad(), zero_leaves(0), zero_leaves(1), zero_leaves(2):
        baseline = METRIC(net(x), None).item()
    if not kw.get("skip_connection"):  # with a skip, the zero-leaf model is not the linearization point
        assert graph.attributions.sum().item() == pytest.approx(clean - baseline, rel=1e-4, abs=1e-5)

    # Latent -> latent edges equal the change of the downstream pre-activation when the upstream latent
    # alone is zeroed (at that token, with every other leaf fixed).
    checked = 0
    for src, dst, w in graph.edges():
        if src.kind != "latent" or dst.kind != "latent" or abs(w) < 1e-4 or checked >= 5:
            continue
        s, d = int(src.site[2:]), int(dst.site[2:])
        if d != s + 1:
            continue

        def edit(acts, row=src.position, lat=src.latent):
            acts = acts.clone()
            acts[row, lat] = 0
            return acts

        ablated = _pre(net, tcs[d], d, x, splice_sae(tcs[s], net.inp[s], edit, target_module=net.mlps[s]))
        before = _pre(net, tcs[d], d, x, splice_sae(tcs[s], net.inp[s], target_module=net.mlps[s]))
        assert before[dst.position, dst.latent] - ablated[dst.position, dst.latent] == pytest.approx(w, rel=1e-4,
                                                                                                    abs=1e-6)
        checked += 1
    assert checked > 0
    assert any(s.position != d.position for s, d, w in graph.edges(1e-6))  # token mixing creates cross-position edges

    # Error -> latent edges: removing one token's error term.
    src, dst, w = next(e for e in graph.edges(1e-4) if e[0].kind == "error" and e[1].kind == "latent"
                       and int(e[1].site[2:]) == int(e[0].site[2:]) + 1)
    s, d = int(src.site[2:]), int(dst.site[2:])

    def drop_error(x_, y_, row=src.position):
        out = y_.clone()
        out[row] = tcs[s].reconstruct(x_[row])
        return out

    ablated = _pre(net, tcs[d], d, x, splice(net.inp[s], drop_error, net.mlps[s]))
    before = _pre(net, tcs[d], d, x, torch.no_grad())
    assert before[dst.position, dst.latent] - ablated[dst.position, dst.latent] == pytest.approx(w, rel=1e-4, abs=1e-6)


def test_clt_graph_is_exact():
    net = ToyMixNet()
    torch.manual_seed(1)
    inputs = torch.randn(64, 5, 8)
    clt = CrossLayerTranscoder(n_layers=3, d_in=16, d_sae=48, activation="topk", k=4)
    pairs = [_pairs(net, layer, inputs) for layer in range(3)]
    clt.initialize_from_data(torch.stack([p[0] for p in pairs], 1), torch.stack([p[1] for p in pairs], 1))
    clt.eval()
    x = torch.randn(1, 5, 8)
    ins, outs = list(net.inp), list(net.mlps)
    graph = attribution_graph(net, [clt_site(clt, ins, outs)], x, METRIC, max_nodes=10_000)
    clean = METRIC(net(x), None).item()
    assert graph.metric == pytest.approx(clean, abs=1e-5)
    with torch.no_grad(), splice_clt(clt, ins, outs, edit=lambda layer, a: torch.zeros_like(a)):
        baseline = METRIC(net(x), None).item()  # all latents and errors 0
    assert graph.attributions.sum().item() == pytest.approx(clean - baseline, rel=1e-4, abs=1e-5)

    def clt_pre(layer, ctx):
        with torch.no_grad(), ctx, capture_activation(net.inp[layer]) as cap:
            net(x)
        x_s = cap["act"][0] * clt.input_scale[layer]
        return x_s @ clt.w_enc[layer] + clt.b_enc[layer]

    sites_seen = set()
    for src, dst, w in graph.edges(1e-4):
        if src.kind != "latent" or dst.kind != "latent":
            continue
        s, d = int(src.site.split(".")[1]), int(dst.site.split(".")[1])
        if (s, d) in sites_seen or d != s + 1:
            continue
        sites_seen.add((s, d))

        def edit(layer, acts, s=s, row=src.position, lat=src.latent):
            if layer == s:
                acts = acts.clone()
                acts[row, lat] = 0
            return acts

        delta = clt_pre(d, splice_clt(clt, ins, outs, keep_error=True))[dst.position, dst.latent] - \
            clt_pre(d, splice_clt(clt, ins, outs, keep_error=True, edit=edit))[dst.position, dst.latent]
        assert delta.item() == pytest.approx(w, rel=1e-4, abs=1e-6)
    assert sites_seen == {(0, 1), (1, 2)}
    assert any(s.site == "clt.0" and d.site == "clt.2" for s, d, _ in graph.edges(1e-6))  # cross-layer edges


def test_residual_sae_graph_prune_and_export():
    net = ToyMixNet()
    torch.manual_seed(1)
    inputs = torch.randn(64, 5, 8)
    saes = []
    for layer in range(2):
        with torch.no_grad(), capture_activation(net.inp[layer]) as cap:
            net(inputs)
        sae = SAE(d_in=16, d_sae=64, k=4)
        sae.initialize_from_data(cap["act"].reshape(-1, 16))
        saes.append(sae.eval())
    x = torch.randn(1, 5, 8)
    sites = [sae_site(s, net.inp[i], name=f"res{i}") for i, s in enumerate(saes)]
    graph = attribution_graph(net, sites, x, METRIC, max_nodes=20)
    assert graph.metric == pytest.approx(METRIC(net(x), None).item(), abs=1e-5)
    latents = [n for n in graph.nodes if n.kind == "latent"]
    errors = [n for n in graph.nodes if n.kind == "error"]
    assert len(latents) == 20 and len(errors) == 2 * 5
    # residual-stream sites cut the stream: res1 latents only get edges from res0 nodes
    for src, dst, _ in graph.edges(0.0):
        assert dst.site == "res1" and src.site == "res0"
    inf = graph.influence()
    assert (inf >= 0).all() and inf.sum() > 0
    pruned = graph.prune(0.5)
    assert 0 < len(pruned.nodes) < len(graph.nodes)
    d = pruned.to_dict()
    assert len(d["nodes"]) == len(pruned.nodes) and all({"source", "target", "weight"} <= set(e) for e in d["edges"])
    assert graph.top_nodes(3)[0].attribution == max((n.attribution for n in latents), key=abs)
