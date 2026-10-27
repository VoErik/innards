"""
Figure style: one look for every plot in innards (needs the ``viz`` extra).

The style file ``innards.mplstyle`` sets clean, publication-style defaults:
print sizes (3.5 in single-column figures, 7-8 pt text), recessive hairline
axes and grids, thin marks, text in ink colors, vector output with embedded
fonts. The colors come from a palette validated for color-vision
deficiencies:

* :data:`CATEGORICAL`: eight hues in a fixed order, for identity (series,
  classes). Assign them in order, never by rank; for scatter plots use at
  most three (beyond that, highlight a few and gray the rest).
* ``innards_seq`` (one hue, light to dark) for magnitudes, and
  ``innards_div`` (blue, gray, red; lightness-matched arms) for signed values
  centred on zero.
* :data:`INK` for text, axes and grids; text never takes a data color.

Typical usage::

    from innards import style

    style.use()                       # for your own figures
    with style.context():             # or temporarily
        fig, ax = plt.subplots()
    fig.savefig("figure.pdf")
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap
)

STYLE_FILE = Path(__file__).with_name("innards.mplstyle")

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
"""Categorical colors (blue, orange, aqua, yellow, magenta, green, violet, red), in their fixed order."""

INK = {
    "primary": "#0b0b0b",    # titles, values
    "secondary": "#52514e",  # axis labels, tick labels, legends
    "muted": "#898781",      # annotations, de-emphasised marks
    "axis": "#c3c2b7",       # spines, ticks, reference lines
    "grid": "#e1e0d9",       # gridlines
    "surface": "#ffffff",    # background, gaps between marks
    "context": "#c9c8c1",    # marks that give context (non-highlighted points)
}
"""Ink colors for everything that is not data."""

SEQUENTIAL = ["#f3f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
"""Blue ramp from near-surface to dark, for magnitudes."""

DIVERGING = ["#104281", "#1c5cab", "#2a78d6", "#6da7ec", "#b7d3f6", "#f0efec",
             "#f6c3bf", "#ed7d77", "#d03b3b", "#a62a2a", "#7a1b1b"]
"""Blue - gray - red, the arms matched in OKLab lightness, for signed values."""

SEQ = LinearSegmentedColormap.from_list("innards_seq", SEQUENTIAL)
DIV = LinearSegmentedColormap.from_list("innards_div", DIVERGING)
CAT = ListedColormap(CATEGORICAL, name="innards_cat")

for _cmap in (SEQ, DIV, CAT):
    if _cmap.name not in mpl.colormaps:
        mpl.colormaps.register(_cmap)


def _params() -> dict:
    params = dict(mpl.rc_params_from_file(str(STYLE_FILE), use_default_template=False))
    params["image.cmap"] = "innards_seq"
    return params


def use() -> None:
    """Applies the innards style globally (for figures created afterwards)."""
    mpl.rcParams.update(_params())


@contextmanager
def context() -> Iterator[None]:
    """
    Applies the innards style inside a ``with`` block.

    Yields:
        None.
    """
    with plt.rc_context(_params()):
        yield


def color(i: int) -> str:
    """
    The ``i``-th categorical color.

    Args:
        i: Slot, ``0 .. 7``.

    Returns:
        Hex color.

    Raises:
        IndexError: Past the eighth slot; fold further series into "other"
            or use small multiples instead of inventing colors.
    """
    if not 0 <= i < len(CATEGORICAL):
        raise IndexError(f"only {len(CATEGORICAL)} categorical colors; fold the rest into 'other' or facet")
    return CATEGORICAL[i]


def width(columns: float = 1.0) -> float:
    """
    Figure width in inches for a print layout.

    Args:
        columns: 1 for a single column (3.5 in), 2 for the full page width (7.2 in).

    Returns:
        The width.
    """
    return 3.5 if columns <= 1 else 7.2 if columns >= 2 else 3.5 + (columns - 1) * 3.7
