"""Tests for the figure style and innards.analysis.plot."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from innards import style  # noqa: E402
from innards.analysis import plot  # noqa: E402
from innards.analysis.spline import slice_partition  # noqa: E402


@pytest.fixture(autouse=True)
def close():
    yield
    plt.close("all")


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_style_is_scoped_and_complete():
    before = dict(matplotlib.rcParams)
    with style.context():
        assert plt.rcParams["image.cmap"] == "innards_seq"
        assert [to_hex(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]] == style.CATEGORICAL
        assert not plt.rcParams["axes.spines.top"] and plt.rcParams["savefig.bbox"] == "tight"
    assert matplotlib.rcParams["image.cmap"] == before["image.cmap"]
    assert {"innards_seq", "innards_div", "innards_cat"} <= set(matplotlib.colormaps)
    assert style.color(1) == "#eb6834"
    with pytest.raises(IndexError):
        style.color(8)
    assert style.width(1) == 3.5 and style.width(2) == 7.2


def test_layer_curve_series_and_bands():
    layers = ["blocks.0", "blocks.1", "blocks.2"]
    per_example = {l: torch.randn(50, generator=gen(i)) + i for i, l in enumerate(layers)}
    ax = plot.layer_curve({"probe": {l: 0.5 + 0.1 * i for i, l in enumerate(layers)}, "effect": per_example},
                          ylabel="value", reference=0.1, reference_label="chance")
    lines = [l for l in ax.lines if l.get_label() in ("probe", "effect")]
    assert [to_hex(l.get_color()) for l in lines] == style.CATEGORICAL[:2]
    assert len(ax.collections) == 1                      # one bootstrap band, for the per-example series
    assert ax.get_legend() is not None
    assert [t.get_text() for t in ax.get_xticklabels()] == layers
    single = plot.layer_curve({l: 1.0 for l in layers})
    assert single.get_legend() is None                    # one series: no legend box


def test_text_never_takes_data_colors():
    ax = plot.layer_curve({"a": {"x": 1.0, "y": 2.0}, "b": {"x": 2.0, "y": 1.0}}, ylabel="y", title="t")
    assert ax.get_title(loc="left") == "t"
    texts = [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_legend().get_texts()]
    assert not {to_hex(t.get_color()) for t in texts} & set(style.CATEGORICAL)


def test_matrix_spectrum_sweep_distributions_scatter(tmp_path):
    M = torch.rand(4, 4)
    ax = plot.similarity_matrix(M + M.T, labels=list("abcd"), vmax=2)
    assert ax.images[0].get_cmap().name == "innards_seq"
    assert plot.similarity_matrix(M - 0.5, signed=True, vmax=None).images[0].get_cmap().name == "innards_div"

    d = 100
    X = torch.randn(3000, d, generator=gen(), dtype=torch.float64) * torch.arange(1, d + 1) ** -0.5
    ax = plot.spectrum({"layer a": X, "layer b": X * 2})
    assert "α = 1." in ax.get_legend().get_texts()[0].get_text()

    values = torch.randn(40, 4, generator=gen(1)) + torch.arange(4)
    ax = plot.sweep({"v": values, "w": -values}, [0, 1, 2, 4], xlabel="alpha", ylabel="KL")
    assert len(ax.get_legend().get_texts()) == 2

    ax = plot.distributions({"correct": torch.randn(200), "wrong": torch.randn(50) + 1}, xlabel="local complexity")
    assert "median" in ax.get_legend().get_texts()[0].get_text()
    plot.scatter(torch.rand(10), torch.randn(10), annotate=[f"PC{i}" for i in range(10)])

    path = plot.save(ax.figure, tmp_path / "fig.pdf")
    assert path.exists() and path.stat().st_size > 0


def test_patching_layouts():
    per_layer = {"blocks.1": torch.randn(8), "blocks.2": torch.randn(8)}
    assert isinstance(plot.patching(per_layer), plt.Axes)
    per_pos = {"blocks.1": torch.randn(8, 17), "blocks.2": torch.randn(8, 17)}
    ax = plot.patching(per_pos)
    assert ax.images[0].get_array().shape == (2, 17)
    fig = plot.patching(per_pos, grid=(4, 4), exclude=(0,))
    assert isinstance(fig, plt.Figure) and fig.axes[0].images[0].get_array().shape == (4, 4)


def test_embedding_caps_highlighted_classes():
    X, y = torch.randn(300, 8, generator=gen()), torch.arange(300) % 5
    ax = plot.embedding(X, y, class_names=list("abcde"))
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["other", "a", "b", "c"]
    assert to_hex(ax.collections[1].get_facecolor()[0]) == style.CATEGORICAL[0]
    with pytest.raises(ValueError):
        plot.embedding(X, y, highlight=[0, 1, 2, 3])
    plot.embedding(X, method="tsne")


def test_embedding_values_annotate_and_coordinates():
    X = torch.randn(100, 8, generator=gen())
    ax = plot.embedding(X, values=X[:, 0], values_label="x0", annotate=[3, 7])
    assert len(ax.figure.axes) == 2  # color bar
    assert [t.get_text() for t in ax.texts] == ["3", "7"]
    ax = plot.embedding(X[:, :2], method=None)
    np.testing.assert_allclose(ax.collections[0].get_offsets(), X[:, :2].numpy(), rtol=1e-6)
    with pytest.raises(ValueError):
        plot.embedding(X, torch.zeros(100), values=X[:, 0])


def test_bars_and_histogram():
    ax = plot.bars(torch.tensor([2.0, 0.5, -1.0]), ["a", "b", "c"], xlabel="effect")
    colors = [to_hex(p.get_facecolor()) for p in ax.patches]
    assert colors == [style.DIVERGING[8]] * 2 + [style.DIVERGING[2]]
    top = max(ax.get_yticklabels(), key=lambda t: t.get_position()[1])
    assert top.get_text() == "a"  # first value on top
    ax = plot.histogram(torch.tensor([0.0, 0.0, 0.1, 0.01, 1e-3]), log10=True, title="density")
    assert ax.get_title(loc="left") == "density (2 of 5 zero)"
    assert sum(p.get_height() for p in ax.patches) == 3


def test_partition_and_images():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    ids, (u, v) = slice_partition(net, torch.zeros(2), torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), "1",
                                  res=32)
    ax = plot.partition(ids["1"], u, v, fill=u + v, fill_label="u + v")
    assert "regions" in ax.get_title(loc="left")

    imgs = torch.randn(6, 3, 32, 32, generator=gen())
    fig = plot.images(imgs, maps=torch.rand(6, 4, 4), titles=[str(i) for i in range(6)], ncols=3)
    assert len([a for a in fig.axes if a.images]) == 6
    fig = plot.top_examples(imgs, torch.tensor([3, 1]), torch.tensor([2.0, 1.0]), positions=torch.tensor([1, 6]),
                            grid=(4, 4), exclude=1)
    assert len(fig.axes[0].patches) == 2 and len(fig.axes[1].patches) == 2   # boxed patch (ring + line)
