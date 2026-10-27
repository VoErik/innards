"""
Interpretability tooling for vision and language models, built on nnsight.

Model access:

* :mod:`innards.models`: loading timm models (wrapped for nnsight) and finding layers.
* :mod:`innards.layouts`: activation layouts and conversion to tokens.
* :mod:`innards.reduce`: what to keep of each example (CLS token, mean, ...).
* :mod:`innards.extract`: extracting activations over a dataset, in memory, to disk, or
  as a shuffled stream of token rows.
* :mod:`innards.store`: saving and loading activations; row caches and their loader.
* :mod:`innards.edits`: interventions (steering, ablation, patching) applied with nnsight.

Subpackages:

* :mod:`innards.analysis`: geometry, similarity, directions, spline partitions,
  causal effects and interpretation of representations.
* :mod:`innards.sae`: sparse autoencoders, transcoders and their analysis.
"""

from .edits import (
    edit_output,
    run
)
from .extract import (
    RowStream,
    capture,
    capture_tokens,
    describe,
    extract,
    extract_to_disk,
    iter_extract,
    layer_tokens
)
from .models import (
    load as load_model,
    module_names,
    stage_names,
    unwrap,
    wrap
)
from .store import (
    RowLoader,
    load as load_acts,
    load_info,
    load_layer,
    save as save_acts,
    write_rows
)
