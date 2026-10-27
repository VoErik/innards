"""Sparse autoencoders for vision and language models.

Core modules:

* :mod:`innards.sae.model`: the SAE (:class:`SAE`, :class:`SAEConfig`).
* :mod:`innards.sae.activations`: TopK, BatchTopK, JumpReLU and ReLU activations;
  new architectures register here.
* :mod:`innards.sae.decoders`: unit-norm linear and Archetypal dictionaries.
* :mod:`innards.sae.trainer`: training, KL fine-tuning and evaluation.
* :mod:`innards.sae.group`: one SAE per hook point, trained from a single pass.
* :mod:`innards.sae.clt`: cross-layer transcoders replacing all MLPs at once.
* :mod:`innards.sae.graph`: attribution graphs over spliced SAEs, transcoders and CLTs.
* :mod:`innards.sae.dashboard`: self-contained HTML latent dashboards and graph viewers.
* :mod:`innards.sae.optim`: Adam, Muon and Signum (I only use Adam, but sparsify use Muon / Signum as well)
* :mod:`innards.sae.analysis`: latent statistics, top examples, dictionary
  geometry, output effects and attributions.
* :mod:`innards.sae.interventions`: ablating, clamping and steering latents.
* :mod:`innards.sae.latent_cache`: encoding a dataset once and analysing the cached
  latents (statistics, top / quantile examples, co-occurrence).
* :mod:`innards.sae.interop`: reading and writing the SAELens format.

Optional: :mod:`innards.sae.viz` (plots and HTML views; ``viz`` extra).

Training data come from the general machinery of innards: rows streamed from
the model (:class:`innards.extract.RowStream`) or read from a row cache
(:class:`innards.store.RowLoader`, e.g. a layer of
:func:`innards.extract.extract_to_disk`).
"""

from .activations import (
    ACTIVATIONS,
    Activation,
    register_activation
)
from .analysis import (
    analyze_dataset,
    latent_attribution,
    latent_patch_maps,
    max_cosine_similarity,
    output_effects,
)
from .clt import (
    CLTConfig,
    CrossLayerTranscoder,
    clt_edits,
    clt_hooks
)
from .decoders import (
    DECODERS,
    Decoder
)
from .graph import (
    AttributionGraph,
    attribution_graph,
    clt_site,
    sae_site
)
from .group import SAEGroup
from .latent_cache import (
    LatentCache,
    cache_latents
)
from .model import (
    SAE,
    SAEConfig,
    SAEOutput
)
from .trainer import (
    KLFinetuneConfig,
    SAETrainer,
    TrainerConfig
)

__all__ = [
    "ACTIVATIONS",
    "DECODERS",
    "Activation",
    "AttributionGraph",
    "CLTConfig",
    "CrossLayerTranscoder",
    "Decoder",
    "KLFinetuneConfig",
    "LatentCache",
    "SAE",
    "SAEConfig",
    "SAEGroup",
    "SAEOutput",
    "SAETrainer",
    "TrainerConfig",
    "analyze_dataset",
    "attribution_graph",
    "cache_latents",
    "clt_hooks",
    "clt_site",
    "latent_attribution",
    "latent_patch_maps",
    "max_cosine_similarity",
    "output_effects",
    "register_activation",
    "sae_site",
    "clt_edits",
]
__version__ = "0.2.0"
