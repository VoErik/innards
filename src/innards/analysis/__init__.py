"""
Analysis of representations: geometry, causality and interpretation.

* :mod:`innards.analysis.geometry`: dimensionality, spectra, manifolds, class
  structure, relations between directions and subspaces.
* :mod:`innards.analysis.similarity`: comparing representations (CKA, RSA,
  Procrustes, SVCCA, mutual nearest neighbours).
* :mod:`innards.analysis.directions`: finding directions (class means, probes,
  LDA, INLP).
* :mod:`innards.analysis.spline`: the partition of input space into regions
  and the local affine maps (spline theory of deep networks).
* :mod:`innards.analysis.causal`: effects of interventions, activation and
  attribution patching, causally relevant directions.
* :mod:`innards.analysis.metrics`: targets and comparisons for the causal tools.
* :mod:`innards.analysis.interp`: top examples, spatial maps, input
  attribution, logit lens, attention.
* :mod:`innards.analysis.units`: per-unit statistics and top examples over a
  dataset (neurons, channels, directions, SAE latents), in one pass or from a
  sparse disk cache.
* :mod:`innards.analysis.dashboard`: HTML dashboards of units and inline text views.
* :mod:`innards.analysis.plot`: figures for all of the above (``viz`` extra).

Representations are tensors (e.g. from :func:`innards.extract.extract` or
:func:`innards.store.load`) and directions are ``V (k, d)``, so the modules
compose: directions found by one are measured, ablated or interpreted by the
others.
"""

from . import (
    causal,
    dashboard,
    directions,
    geometry,
    interp,
    metrics,
    similarity,
    spline,
    units
)
