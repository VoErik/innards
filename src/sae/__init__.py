"""Sparse autoencoders for vision and language models.

Core modules:

* :mod:`sae.model`: the SAE (:class:`SAE`, :class:`SAEConfig`).
* :mod:`sae.activations`: TopK, BatchTopK, JumpReLU and ReLU activations;
  new architectures register here.
* :mod:`sae.decoders`: unit-norm linear and Archetypal dictionaries.
* :mod:`sae.trainer`: training, KL fine-tuning and evaluation.
* :mod:`sae.group`: one SAE per hook point, trained from a single pass.
* :mod:`sae.clt`: cross-layer transcoders replacing all MLPs at once.
* :mod:`sae.graph`: attribution graphs over spliced SAEs, transcoders and CLTs.
* :mod:`sae.data`: activation harvesting from vision and language models.
* :mod:`sae.cache`: caching harvested activations to disk.
* :mod:`sae.optim`: Adam, Muon and Signum (I only use Adam, but sparsify use Muon / Signum as well)
* :mod:`sae.analysis`: latent statistics, top examples, dictionary
  geometry, output effects and attributions.
* :mod:`sae.interventions`: ablating, clamping and steering latents.
* :mod:`sae.latent_cache`: encoding a dataset once and analysing the cached
  latents (statistics, top / quantile examples, co-occurrence).
* :mod:`sae.interop`: reading and writing the SAELens format.

Optional: :mod:`sae.viz` (plots and HTML views; ``viz`` extra).
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
from .cache import (
    CachedActivations,
    cache_activations
)
from .clt import (
    CLTConfig,
    CrossLayerTranscoder,
    clt_hooks,
    splice_clt
)
from .data import ActivationStore
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
    "ActivationStore",
    "AttributionGraph",
    "CLTConfig",
    "CrossLayerTranscoder",
    "CachedActivations",
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
    "cache_activations",
    "cache_latents",
    "clt_hooks",
    "clt_site",
    "latent_attribution",
    "latent_patch_maps",
    "max_cosine_similarity",
    "output_effects",
    "register_activation",
    "sae_site",
    "splice_clt",
]
__version__ = "0.2.0"
