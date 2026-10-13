"""Sparse autoencoders for vision and language models.

Core modules:

* :mod:`sae.model`: the SAE (:class:`SAE`, :class:`SAEConfig`).
* :mod:`sae.activations`: TopK, BatchTopK, JumpReLU and ReLU activations;
  new architectures register here.
* :mod:`sae.decoders`: unit-norm linear and Archetypal dictionaries.
* :mod:`sae.trainer`: training, KL fine-tuning and evaluation.
* :mod:`sae.data`: activation harvesting from vision and language models.
* :mod:`sae.cache`: caching harvested activations to disk.
* :mod:`sae.optim`: Adam, Muon and Signum (I only use Adam, but sparsify use Muon / Signum as well)
* :mod:`sae.analysis`: latent statistics, top examples, dictionary
  geometry, output effects and attributions.

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
from .data import ActivationStore
from .decoders import (
    DECODERS,
    Decoder
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
    "CachedActivations",
    "Decoder",
    "KLFinetuneConfig",
    "SAE",
    "SAEConfig",
    "SAEOutput",
    "SAETrainer",
    "TrainerConfig",
    "analyze_dataset",
    "cache_activations",
    "latent_attribution",
    "latent_patch_maps",
    "max_cosine_similarity",
    "output_effects",
    "register_activation",
]
__version__ = "0.2.0"
