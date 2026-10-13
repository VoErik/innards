"""
Analysis of trained SAEs:

* latent statistics,
* top-activating examples,
* dictionary geometry,
* output effects,
* and attributions.

Everything here is modality-agnostic. Activations are captured with the hook
utilities of :mod:`sae.data`, so the same functions work for ViTs, CNNs
(``layout="bchw"``) and language models.

References:
    nostalgebraist (2020). interpreting GPT: the logit lens. LessWrong.
    Bricken, T. et al. (2023). Towards Monosemanticity: Decomposing Language
        Models With Dictionary Learning. Transformer Circuits Thread.
        (Feature dashboards; latent effects on logits via the unembedding.)
    Nanda, N. (2023). Attribution Patching: Activation Patching At Industrial
        Scale. Blog post, neelnanda.io.
    Syed, A., Rager, C. & Conmy, A. (2023). Attribution Patching Outperforms
        Automated Circuit Discovery. arXiv:2310.10348.
    Marks, S. et al. (2024). Sparse Feature Circuits: Discovering and Editing
        Interpretable Causal Graphs in Language Models. arXiv:2403.19647.
        (SAE error terms in attribution.)
    Simonyan, K., Vedaldi, A. & Zisserman, A. (2013). Deep Inside
        Convolutional Networks: Visualising Image Classification Models and
        Saliency Maps. arXiv:1312.6034.
"""

from __future__ import annotations

import math
from collections.abc import (
    Callable,
    Iterable, Sequence
)
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from einops import (
    einsum,
    rearrange
)
from torch import (
    Tensor,
    nn
)

from .data import (
    _to_device,
    capture_activation,
    default_input_fn,
    keep_positions,
    patch_kept_tokens,
    replace_activation,
    run_capture,
    to_tokens,
    token_keep_mask,
)
from .model import SAE


@dataclass
class LatentStats:
    """
    Per-latent activation statistics over a dataset.

    Attributes:
        frequency: Fraction of kept tokens on which each latent is active,
            Shape: ``(d_sae,)``.
        mean_activation: Mean absolute activation when active, Shape: ``(d_sae,)``.
        max_activation: Maximum absolute activation, Shape: ``(d_sae,)``.
        n_tokens: Number of kept tokens seen.
    """

    frequency: Tensor
    mean_activation: Tensor
    max_activation: Tensor
    n_tokens: int

    @property
    def dead(self) -> Tensor:
        """Indices of latents that never fired."""
        return (self.frequency == 0).nonzero().squeeze(-1)

    def dense(self, threshold: float = 0.1) -> Tensor:
        """
        Indices of latents firing on more than ``threshold`` of tokens.

        Args:
            threshold: Frequency threshold.

        Returns:
            Long tensor of latent indices.
        """
        return (self.frequency > threshold).nonzero().squeeze(-1)


@dataclass
class TopExamples:
    """
    Top-activating tokens per latent, by absolute activation.

    Attributes:
        values: Activations, shape ``(d_sae, n_top)``, sorted by descending
            magnitude.
        example: Index of the input in loader order, ``(d_sae, n_top)``; -1 for
            empty slots.
        position: Token position within that input (in the ``(batch, tokens,
            d)`` layout, counting excluded positions), ``(d_sae, n_top)``;
            -1 for empty slots.
    """

    values: Tensor
    example: Tensor
    position: Tensor

    def for_latent(self, latent: int) -> list[tuple[float, int, int]]:
        """
        ``(value, example, position)`` triples of one latent, best first.

        Args:
            latent: Latent index.

        Returns:
            The non-empty entries.
        """
        out = []
        for v, e, p in zip(
                self.values[latent].tolist(),
                self.example[latent].tolist(),
                self.position[latent].tolist()
        ):
            if e >= 0 and v != 0:
                out.append((v, e, p))
        return out

    def unique_examples(self, latent: int) -> list[tuple[float, int, int]]:
        """
        Like :meth:`for_latent` but keeps only the best token per input.

        Args:
            latent: Latent index.

        Returns:
            One ``(value, example, position)`` triple per input.
        """
        seen, out = set(), []
        for v, e, p in self.for_latent(latent):
            if e not in seen:
                seen.add(e)
                out.append((v, e, p))
        return out


@dataclass
class DatasetAnalysis:
    """
    Result of :func:`analyze_dataset`.

    Attributes:
        stats: Per-latent statistics.
        top: Top-activating tokens per latent.
    """

    stats: LatentStats
    top: TopExamples


@dataclass
class OutputEffects:
    """
    Direct effect of latents on model outputs through a linear readout.

    Attributes:
        latents: Latent indices, shape ``(n_latents,)``.
        top_values: Largest effects, ``(n_latents, n)``.
        top_ids: Output ids (classes / tokens) of the largest effects.
        bottom_values: Most negative effects, ``(n_latents, n)``.
        bottom_ids: Output ids of the most negative effects.
    """

    latents: Tensor
    top_values: Tensor
    top_ids: Tensor
    bottom_values: Tensor
    bottom_ids: Tensor


@dataclass
class Attribution:
    """
    Result of :func:`latent_attribution`.

    Attributes:
        acts: Latent activations, ``(batch, tokens, d_sae)``; zero at
            tokens that were not patched.
        attribution: ``acts * d target / d acts``, same shape: the
            first-order estimate of how much the target drops if the latent
            is zero-ablated at that position. TODO: also add median ablation
        error_attribution: Attribution of the SAE error term, summed over
            ``d_in``, shape ``(batch, tokens)``.
        keep: Boolean ``(batch, tokens)`` mask of the patched tokens.
        target: The target values, shape ``(batch,)``.
    """

    acts: Tensor
    attribution: Tensor
    error_attribution: Tensor
    keep: Tensor
    target: Tensor

    def top_latents(self, example: int = 0, n: int = 10) -> tuple[Tensor, Tensor]:
        """
        Latents with the largest total attribution for one example.

        Args:
            example: Index within the batch.
            n: Number of latents.

        Returns:
            ``(values, latent_ids)``, sorted by absolute value.
        """
        total = self.attribution[example].sum(0)
        idx = total.abs().topk(min(n, total.numel())).indices
        return total[idx], idx


@contextmanager
def _frozen(*modules: nn.Module):
    """Disables gradients of the modules' parameters, restoring them after."""
    flags = [[p.requires_grad for p in m.parameters()] for m in modules]
    for m in modules:
        m.requires_grad_(False)
    try:
        yield
    finally:
        for m, fl in zip(modules, flags):
            for p, f in zip(m.parameters(), fl):
                p.requires_grad_(f)


def tokens_to_grid(values: Tensor, grid: tuple[int, int] | None = None) -> Tensor:
    """
    Reshapes per-patch values ``(..., n_patches)`` to ``(..., h, w)``.

    Args:
        values: Values over patch tokens (excluded tokens such as CLS removed).
        grid: ``(h, w)``; ``None`` assumes a square grid.

    Returns:
        Tensor of shape ``(..., h, w)``.

    Raises:
        ValueError: If no grid is given and the patch count is not square.
    """
    n = values.shape[-1]
    if grid is None:
        side = math.isqrt(n)
        if side * side != n:
            raise ValueError(f"{n} patches do not form a square grid; pass grid=(h, w)")
        grid = (side, side)
    return rearrange(values, "... (h w) -> ... h w", h=grid[0], w=grid[1])


@torch.no_grad()
def encode_inputs(
    sae: SAE,
    model: nn.Module,
    hook_module: nn.Module,
    batch: Any,
    layout: str = "btd",
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
) -> Tensor:
    """
    Latent activations for every token of a batch of model inputs.

    Args:
        sae: The SAE (put in eval mode).
        model: The model the SAE was trained on.
        hook_module: Module whose output the SAE reads.
        batch: Model inputs, already on the right device.
        layout: Activation layout at the hook.
        input_fn: Maps the batch to model ``(args, kwargs)``.

    Returns:
        Activations ``(batch, tokens, d_sae)`` for all positions, including
        ones excluded during training.
    """
    sae.eval()
    model.eval()
    act = run_capture(model, hook_module, batch, input_fn, stop=True)
    return sae.encode(to_tokens(act, layout).to(sae.w_enc.dtype))


@torch.no_grad()
def analyze_dataset(
    sae: SAE,
    model: nn.Module,
    hook_module: nn.Module,
    input_loader: Iterable,
    n_top: int = 16,
    layout: str = "btd",
    exclude_positions: Sequence[int] = (),
    token_mask_fn: Callable[[Any], Tensor] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
    chunk_tokens: int = 4096,
) -> DatasetAnalysis:
    """
    One pass over a dataset collecting latent statistics and top examples.

    The loader must be deterministic (``shuffle=False``) so that
    ``TopExamples.example`` indices can be mapped back to inputs. Only kept
    tokens are encoded, ``chunk_tokens`` at a time, so memory is bounded by
    ``chunk_tokens * d_sae`` whatever the batch size and sequence length.

    Args:
        sae: The SAE.
        model: The model the SAE was trained on.
        hook_module: Module whose output the SAE reads.
        input_loader: Iterable of model inputs.
        n_top: Number of top tokens kept per latent.
        layout: Activation layout at the hook.
        exclude_positions: Positions to ignore, as in training.
        token_mask_fn: Optional ``fn(batch) -> bool (batch, tokens)`` mask,
            e.g. from the attention mask.
        input_fn: Maps a batch to model ``(args, kwargs)``.
        device: Device to run on; defaults to the SAE's device.
        max_batches: Optional cap on the number of batches.
        chunk_tokens: Tokens encoded at once.

    Returns:
        A :class:`DatasetAnalysis`.
    """
    device = torch.device(device) if device is not None else sae.device
    d_sae = sae.cfg.d_sae
    count = torch.zeros(d_sae, device=device)
    total = torch.zeros(d_sae, device=device)
    peak = torch.zeros(d_sae, device=device)
    top_v = torch.zeros(0, d_sae, device=device)
    top_e = torch.zeros(0, d_sae, dtype=torch.long, device=device)
    top_p = torch.zeros(0, d_sae, dtype=torch.long, device=device)
    n_tokens = 0
    offset = 0

    sae.eval()
    model.eval()
    for i, batch in enumerate(input_loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = _to_device(batch, device)
        tokens = to_tokens(run_capture(model, hook_module, batch, input_fn, stop=True), layout)
        b, t, _ = tokens.shape
        keep = token_keep_mask(tokens, exclude_positions, token_mask_fn, batch)
        flat_idx = rearrange(keep, "b t -> (b t)").nonzero().squeeze(-1)  # into the (b t) grid
        kept = tokens[keep].to(sae.w_enc.dtype)
        n_tokens += kept.shape[0]

        for start in range(0, kept.shape[0], chunk_tokens):
            acts = sae.encode(kept[start : start + chunk_tokens]).float()
            mags = acts.abs()
            count += (mags > 0).sum(0)
            total += mags.sum(0)
            peak = torch.maximum(peak, mags.max(0).values)

            _, idx = mags.topk(min(n_top, mags.shape[0]), dim=0)
            where = flat_idx[start + idx]
            cand_v = torch.cat([top_v, acts.gather(0, idx)])
            cand_e = torch.cat([top_e, offset + where // t])
            cand_p = torch.cat([top_p, where % t])
            sel = cand_v.abs().topk(min(n_top, cand_v.shape[0]), dim=0).indices
            top_v, top_e, top_p = cand_v.gather(0, sel), cand_e.gather(0, sel), cand_p.gather(0, sel)
        offset += b

    empty = top_v == 0
    top_e, top_p = top_e.masked_fill(empty, -1), top_p.masked_fill(empty, -1)
    stats = LatentStats(
        frequency=(count / max(n_tokens, 1)).cpu(),
        mean_activation=(total / count.clamp_min(1)).cpu(),
        max_activation=peak.cpu(),
        n_tokens=n_tokens,
    )
    top = TopExamples(
        values=top_v.T.contiguous().cpu(),
        example=top_e.T.contiguous().cpu(),
        position=top_p.T.contiguous().cpu()
    )
    return DatasetAnalysis(stats, top)


@torch.no_grad()
def _unit_atoms(sae: SAE) -> Tensor:
    return F.normalize(sae.decoder_weight().detach().float(), dim=-1)


@torch.no_grad()
def max_cosine_similarity(sae: SAE, chunk: int = 4096) -> tuple[Tensor, Tensor]:
    """
    Highest cosine similarity of each decoder atom to any other atom.

    Values near 1 indicate redundant (duplicated or split) latents.

    Args:
        sae: The SAE.
        chunk: Rows per chunk, bounding memory to ``chunk * d_sae``.

    Returns:
        ``(values, indices)``, each of shape ``(d_sae,)``.
    """
    atoms = _unit_atoms(sae)
    vals, idx = [], []
    for start in range(0, atoms.shape[0], chunk):
        sims = einsum(atoms[start : start + chunk], atoms, "c d, s d -> c s")
        rows = torch.arange(sims.shape[0], device=sims.device)
        sims[rows, rows + start] = -2.0  # exclude self
        v, i = sims.max(-1)
        vals.append(v)
        idx.append(i)
    return torch.cat(vals).cpu(), torch.cat(idx).cpu()


@torch.no_grad()
def nearest_latents(sae: SAE, latent: int, n: int = 10) -> tuple[Tensor, Tensor]:
    """
    Latents whose decoder atoms are most similar to ``latent``'s.

    Args:
        sae: The SAE.
        latent: Query latent.
        n: Number of neighbours.

    Returns:
        ``(cosine_similarities, latent_ids)``, excluding the query itself.
    """
    atoms = _unit_atoms(sae)
    sims = einsum(atoms[latent], atoms, "d, s d -> s")
    sims[latent] = -2.0
    v, i = sims.topk(min(n, sims.numel() - 1))
    return v.cpu(), i.cpu()


@torch.no_grad()
def decoder_similarity_matrix(sae: SAE, latents: Sequence[int] | Tensor) -> Tensor:
    """
    Pairwise cosine similarity of a subset of decoder atoms.

    Args:
        sae: The SAE.
        latents: Latent indices.

    Returns:
        Matrix of shape ``(n, n)``.
    """
    atoms = _unit_atoms(sae)[torch.as_tensor(latents, device=sae.device)]
    return einsum(atoms, atoms, "a d, b d -> a b").cpu()


@torch.no_grad()
def encoder_decoder_cosine(sae: SAE) -> Tensor:
    """
    Cosine similarity between each latent's encoder and decoder direction.

    Low values are expected for latents that must suppress interference from
    related latents (feature absorption; Chanin et al., 2024).

    Args:
        sae: The SAE.

    Returns:
        Tensor of shape ``(d_sae,)``.
    """
    enc = F.normalize(sae.w_enc.detach().float().T, dim=-1)
    return einsum(enc, _unit_atoms(sae), "s d, s d -> s").cpu()


@torch.no_grad()
def dictionary_embedding(sae: SAE, method: str = "pca", n_components: int = 2, seed: int = 0) -> Tensor:
    """
    Low-dimensional embedding of the unit-normalised decoder atoms.

    Args:
        sae: The SAE.
        method: ``"pca"`` (torch only) or ``"umap"`` (needs ``umap-learn``;
            cosine metric).
        n_components: Output dimensionality.
        seed: Random seed.

    Returns:
        Coordinates of shape ``(d_sae, n_components)``.

    Raises:
        ValueError: If the method is unknown.
    """
    atoms = _unit_atoms(sae).cpu()
    if method == "pca":
        torch.manual_seed(seed)
        centred = atoms - atoms.mean(0, keepdim=True)
        _, _, v = torch.pca_lowrank(centred, q=n_components, center=False)
        return centred @ v[:, :n_components]
    if method == "umap":
        import umap

        coords = umap.UMAP(
            n_components=n_components,
            metric="cosine",
            random_state=seed
        ).fit_transform(
            atoms.numpy()
        )
        return torch.from_numpy(coords)
    raise ValueError(f"unknown method {method!r}")


@torch.no_grad()
def output_effects(
    sae: SAE,
    readout: Tensor,
    latents: Sequence[int] | Tensor | None = None,
    n: int = 10,
) -> OutputEffects:
    """
    Direct effect of each latent's decoder atom on a linear readout.

    For language models pass the unembedding (``(vocab, d_model)``), which is
    the logit lens applied to decoder atoms (nostalgebraist, 2020).
    For a ViT classifier pass the classification head weight (``(n_classes, d)``);
    this is only exact for the final residual stream and ignores the final normalization layer.

    Args:
        sae: The SAE.
        readout: Readout matrix of shape ``(n_outputs, d_in)``.
        latents: Latents to analyse; ``None`` means all.
        n: Number of top and bottom outputs per latent.

    Returns:
        An :class:`OutputEffects`.
    """
    idx = torch.arange(sae.cfg.d_sae) if latents is None else torch.as_tensor(latents)
    atoms = sae.decoder_weight().detach().float()[idx.to(sae.device)]
    effects = einsum(atoms, readout.to(atoms), "l d, o d -> l o")
    top_v, top_i = effects.topk(n, dim=-1)
    bot_v, bot_i = effects.topk(n, dim=-1, largest=False)
    return OutputEffects(
        latents=idx.cpu(),
        top_values=top_v.cpu(),
        top_ids=top_i.cpu(),
        bottom_values=bot_v.cpu(),
        bottom_ids=bot_i.cpu()
    )


def latent_attribution(
    sae: SAE,
    model: nn.Module,
    hook_module: nn.Module,
    batch: Any,
    target_fn: Callable[[Any, Any], Tensor],
    layout: str = "btd",
    exclude_positions: Sequence[int] = (),
    token_mask_fn: Callable[[Any], Tensor] | None = None,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
) -> Attribution:
    """
    Attributes a scalar model output to SAE latents at every position.

    The SAE is spliced in as ``x = decode(encode(x)) + error`` with the error
    term kept (Marks et al., 2024), so the forward pass is exactly the clean
    model. The attribution ``a * dT/da`` is the attribution-patching estimate
    (Nanda, 2023; Syed et al., 2023) of the change in the target ``T`` when
    latent ``a`` is zero-ablated.

    Args:
        sae: The SAE.
        model: The model the SAE was trained on.
        hook_module: Module whose output the SAE reads.
        batch: Model inputs, already on the right device.
        target_fn: ``fn(model_output, batch) -> (batch,)`` scalar per
            example, e.g. the logit of the predicted class, or of the correct
            next token at the last position.
        layout: Activation layout at the hook.
        exclude_positions: Positions left unpatched (e.g. CLS / BOS).
        token_mask_fn: Optional ``fn(batch) -> bool (batch, tokens)`` mask of
            tokens to patch, e.g. non-padding tokens.
        input_fn: Maps the batch to model ``(args, kwargs)``.

    Returns:
        An :class:`Attribution`.
    """
    sae.eval()
    model.eval()
    store: dict[str, Tensor] = {}

    def splice(x: Tensor) -> Tensor:
        x = x.detach().to(sae.w_enc.dtype)
        with torch.no_grad():
            acts = sae.encode(x)
        acts.requires_grad_(True)
        recon = sae.decode(acts)
        error = (x - recon.detach()).detach().requires_grad_(True)
        store.update(acts=acts, error=error)
        return recon + error

    def patch(act: Tensor) -> Tensor:
        store["keep"] = token_keep_mask(to_tokens(act, layout), exclude_positions, token_mask_fn, batch)
        return patch_kept_tokens(act, splice, layout, exclude_positions, token_mask_fn, batch)

    args, kwargs = input_fn(batch)
    with _frozen(model, sae), replace_activation(hook_module, patch):
        target = target_fn(model(*args, **kwargs), batch)
        target.sum().backward()
    acts, error, keep = store["acts"], store["error"], store["keep"]

    def scatter(values: Tensor) -> Tensor:
        full = values.new_zeros(*keep.shape, *values.shape[1:])
        full[keep] = values
        return full

    return Attribution(
        acts=scatter(acts.detach()),
        attribution=scatter((acts * acts.grad).detach()),
        error_attribution=scatter((error * error.grad).sum(-1).detach()),
        keep=keep,
        target=target.detach(),
    )


def input_attribution(
    sae: SAE,
    model: nn.Module,
    hook_module: nn.Module,
    inputs: Tensor,
    latent: int,
    layout: str = "btd",
    exclude_positions: Sequence[int] = (),
    use_preactivation: bool = True,
) -> Tensor:
    """
    Gradient x input attribution of a latent to the model's input.

    For images this is a pixel-level saliency map (Simonyan et al., 2013) of
    the latent's summed activation over all kept positions.

    Args:
        sae: The SAE.
        model: The model, called as ``model(inputs)``.
        hook_module: Module whose output the SAE reads.
        inputs: Float input tensor, e.g. images ``(batch, 3, H, W)``.
        latent: Latent index.
        layout: Activation layout at the hook.
        exclude_positions: Positions to ignore.
        use_preactivation: Differentiate the encoder pre-activation instead of
            the post-activation value, so the map is informative even where
            the latent is not selected.

    Returns:
        Attribution with the shape of ``inputs``.
    """
    sae.eval()
    model.eval()
    x = inputs.detach().clone().requires_grad_(True)
    with _frozen(model, sae):
        with capture_activation(hook_module, stop=False) as store:
            model(x)
        tokens = to_tokens(store["act"], layout)
        pos = keep_positions(tokens.shape[1], exclude_positions, tokens.device)
        kept = tokens[:, pos].to(sae.w_enc.dtype)
        values = sae.pre_activations(kept) if use_preactivation else sae.encode(kept)
        values[..., latent].sum().backward()
    return (x.grad * x).detach()


@torch.no_grad()
def latent_patch_maps(
    sae: SAE,
    model: nn.Module,
    hook_module: nn.Module,
    images: Tensor,
    latents: int | Sequence[int],
    exclude_positions: Sequence[int] = (0,),
    grid: tuple[int, int] | None = None,
    layout: str = "btd",
) -> Tensor:
    """
    Activation maps of latents over the patch grid of each image.

    Args:
        sae: The SAE.
        model: A ViT (``layout="btd"``) or CNN (``layout="bchw"``).
        hook_module: Module whose output the SAE reads.
        images: Images ``(batch, 3, H, W)`` on the model's device.
        latents: One latent or a list of latents.
        exclude_positions: Non-patch tokens to drop (CLS, registers). Ignored
            for ``layout="bchw"``.
        grid: Patch grid ``(h, w)``; ``None`` infers a square grid (ViTs) or
            uses the feature map size (CNNs).
        layout: Activation layout at the hook.

    Returns:
        Maps of shape ``(batch, h, w)`` for one latent, or
        ``(batch, n_latents, h, w)`` for a list.
    """
    act = run_capture(model, hook_module, images, default_input_fn, stop=True)
    if layout == "bchw":
        grid = grid or (act.shape[2], act.shape[3])
        exclude_positions = ()
    acts = sae.eval().encode(to_tokens(act, layout).to(sae.w_enc.dtype))
    pos = keep_positions(acts.shape[1], exclude_positions, acts.device)
    single = isinstance(latents, int)
    sel = acts[:, pos][..., [latents] if single else list(latents)]
    maps = tokens_to_grid(rearrange(sel, "b t l -> b l t"), grid)
    return maps[:, 0] if single else maps
