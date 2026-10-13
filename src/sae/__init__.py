"""TopK / BatchTopK sparse autoencoders.

Core modules:

* :mod:`sae.model`: the SAE (:class:`TopKSAE`, :class:`SAEConfig`).
* :mod:`sae.trainer`: training, KL fine-tuning and evaluation.
* :mod:`sae.data`: activation harvesting from vision and language models.
* :mod:`sae.cache`: caching harvested activations to disk.
* :mod:`sae.optim`: Adam, Muon and Signum (I only use Adam, but sparsify use Muon / Signum as well)
* :mod:`sae.analysis`: latent statistics, top examples, dictionary
  geometry, output effects and attributions.

Optional: :mod:`sae.viz` (plots and HTML views; ``viz`` extra).
"""

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
from .model import (
    SAEConfig,
    SAEOutput,
    TopKSAE
)
from .trainer import (
    KLFinetuneConfig,
    SAETrainer,
    TrainerConfig
)

__all__ = [
    "ActivationStore",
    "CachedActivations",
    "KLFinetuneConfig",
    "SAEConfig",
    "SAEOutput",
    "SAETrainer",
    "TopKSAE",
    "TrainerConfig",
    "analyze_dataset",
    "cache_activations",
    "latent_attribution",
    "latent_patch_maps",
    "max_cosine_similarity",
    "output_effects",
]
__version__ = "0.1.0"
