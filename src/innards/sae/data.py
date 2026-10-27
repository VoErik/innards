"""
Activation harvesting for SAE training on vision and language models.

:class:`ActivationStore` runs a frozen model over an input loader, captures the
output of one module with a forward hook, drops excluded tokens and serves
shuffled token batches from a large buffer, so a batch mixes tokens from many
images or sequences. The same hook utilities splice the SAE into the model for
KL fine-tuning (see :meth:`innards.sae.trainer.SAETrainer.finetune_kl`).

Three activation layouts are supported:

* ``"btd"``: tokens, ``(batch, tokens, d)``, e.g. ViT / CLIP residual streams
  or language-model hidden states.
* ``"bchw"``: convolutional feature maps ``(batch, channels, h, w)``; every
  spatial position is a token.
* ``"bd"``: one vector per example, e.g. pooled embeddings.

Excluding tokens:
Some tokens might want to be excluded, e.g.

Language Models:
Gemma Scope excludes BOS, EOS and padding tokens (Lieberum
et al., 2024), whose norms are far larger than typical tokens (Gao et al.,
2024, App. F.2). For token-ID based exclusion in language models use ``token_mask_fn``.

ViTs:
In ViTs, the CLS token and REG tokens (Darcet et al., 2024) play a similar role
(exclude with ``exclude_positions``).

References:
    Gao, L. et al. (2024). Scaling and evaluating sparse autoencoders.
        arXiv:2406.04093.
    Lieberum, T. et al. (2024). Gemma Scope: Open Sparse Autoencoders
        Everywhere All At Once on Gemma 2. arXiv:2408.05147.
    Darcet, T., Oquab, M., Mairal, J. & Bojanowski, P. (2024). Vision
        Transformers Need Registers. ICLR 2024, arXiv:2309.16588.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Sequence
)
from contextlib import contextmanager
from typing import Any

import torch
from einops import rearrange
from torch import (
    Tensor,
    nn
)

LAYOUTS = ("btd", "bchw", "bd")


def to_tokens(act: Tensor, layout: str) -> Tensor:
    """
    Converts a captured activation to ``(batch, tokens, d)``.

    Args:
        act: Activation in ``layout``.
        layout: One of ``"btd"``, ``"bchw"``, ``"bd"``.

    Returns:
        Tensor of shape ``(batch, tokens, d)``.

    Raises:
        ValueError: If the layout is unknown.
    """
    if layout == "btd":
        return act
    if layout == "bchw":
        return rearrange(act, "b c h w -> b (h w) c")
    if layout == "bd":
        return rearrange(act, "b d -> b 1 d")
    raise ValueError(f"unknown layout {layout!r}, expected one of {LAYOUTS}")


def from_tokens(tokens: Tensor, layout: str, like: Tensor) -> Tensor:
    """
    Inverse of :func:`to_tokens`.

    Args:
        tokens: Tensor of shape ``(batch, tokens, d)``.
        layout: Target layout.
        like: A tensor in the target layout, used for spatial sizes.

    Returns:
        Tensor in ``layout``.

    Raises:
        ValueError: If the layout is unknown.
    """
    if layout == "btd":
        return tokens
    if layout == "bchw":
        return rearrange(tokens, "b (h w) c -> b c h w", h=like.shape[2], w=like.shape[3])
    if layout == "bd":
        return rearrange(tokens, "b 1 d -> b d")
    raise ValueError(f"unknown layout {layout!r}, expected one of {LAYOUTS}")


def keep_positions(n_tokens: int, exclude: Sequence[int], device: torch.device) -> Tensor:
    """
    Indices of token positions not in ``exclude`` (negative indices allowed).

    Args:
        n_tokens: Sequence length.
        exclude: Positions to drop, e.g. ``(0,)`` for CLS / BOS.
        device: Device of the result.

    Returns:
        Long tensor of kept positions.
    """
    mask = torch.ones(n_tokens, dtype=torch.bool, device=device)
    if exclude:
        mask[torch.tensor([p % n_tokens for p in exclude], device=device)] = False
    return mask.nonzero().squeeze(-1)


def token_keep_mask(
    tokens: Tensor,
    exclude_positions: Sequence[int] = (),
    token_mask_fn: Callable[[Any], Tensor] | None = None,
    batch: Any = None,
) -> Tensor:
    """
    Boolean ``(batch, tokens)`` mask of tokens to keep.

    Args:
        tokens: Activations in ``(batch, tokens, d)`` layout.
        exclude_positions: Positions to drop, e.g. ``(0,)`` for CLS / BOS.
        token_mask_fn: Optional ``fn(batch) -> bool (batch, tokens)``.
        batch: The model input batch passed to ``token_mask_fn``.

    Returns:
        The keep mask.
    """
    keep = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    if exclude_positions:
        keep[:] = False
        keep[:, keep_positions(tokens.shape[1], exclude_positions, tokens.device)] = True
    if token_mask_fn is not None:
        keep &= token_mask_fn(batch).to(keep.device, torch.bool)
    return keep


def default_input_fn(batch: Any) -> tuple[tuple, dict]:
    """
    Turns a loader batch into model ``(args, kwargs)``.

    Dicts (e.g. Hugging Face tokenizer output) become keyword arguments;
    tuples or lists such as ``(images, labels)`` pass their first element;
    tensors are passed as is.

    Args:
        batch: A batch from the input loader.

    Returns:
        ``(args, kwargs)`` for ``model(*args, **kwargs)``.
    """
    if isinstance(batch, dict):
        return (), dict(batch)
    if isinstance(batch, (tuple, list)):
        return (batch[0],), {}
    return (batch,), {}


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


class _StopForward(Exception):
    pass


@contextmanager
def capture_activation(module: nn.Module, stop: bool = False) -> Iterator[dict[str, Tensor]]:
    """
    Captures the output of ``module`` during a forward pass.

    Tuple outputs are reduced to their first element.

    Args:
        module: The module to hook.
        stop: Abort the forward pass right after the module, saving the cost
            of the remaining layers.

    Yields:
        A dict whose ``"act"`` entry holds the captured tensor after the
        forward pass.
    """
    store: dict[str, Tensor] = {}

    def hook(_mod, _inp, out):
        store["act"] = out[0] if isinstance(out, (tuple, list)) else out
        if stop:
            raise _StopForward

    handle = module.register_forward_hook(hook)
    try:
        yield store
    finally:
        handle.remove()


@contextmanager
def replace_activation(module: nn.Module, fn: Callable[[Tensor], Tensor]) -> Iterator[None]:
    """
    Replaces the output of ``module`` with ``fn(output)`` during forward passes.

    Args:
        module: The module to hook.
        fn: Function applied to the module output (first element for tuples).

    Yields:
        None.
    """

    def hook(_mod, _inp, out):
        if isinstance(out, tuple):
            return (fn(out[0]), *out[1:])
        if isinstance(out, list):
            return [fn(out[0]), *out[1:]]
        return fn(out)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def patch_kept_tokens(
    act: Tensor,
    fn: Callable[[Tensor, Tensor], Tensor],
    layout: str = "btd",
    exclude_positions: Sequence[int] = (),
    token_mask_fn: Callable[[Any], Tensor] | None = None,
    batch: Any = None,
    source: Tensor | None = None,
) -> Tensor:
    """
    Replaces the kept tokens of a captured activation by ``fn(x, y)``.

    Excluded positions (CLS, BOS, ...) and tokens masked out by
    ``token_mask_fn`` (e.g. padding) pass through unchanged.

    Args:
        act: Module output in ``layout``; ``y`` are its kept tokens.
        fn: Maps kept tokens ``x, y`` (both ``(n, d)``) to replacements for ``y``.
        layout: Activation layout.
        exclude_positions: Positions left unchanged.
        token_mask_fn: Optional ``fn(batch) -> bool (batch, tokens)`` mask of
            tokens to replace.
        batch: The model input batch passed to ``token_mask_fn``.
        source: Optional activation (same layout and token grid) whose kept
            tokens are ``x``, e.g. a transcoder's input; ``None`` uses ``act``.

    Returns:
        The patched activation, in ``layout``.
    """
    tokens = to_tokens(act, layout)
    keep = token_keep_mask(tokens, exclude_positions, token_mask_fn, batch)
    y = tokens[keep]
    x = y if source is None else to_tokens(source, layout)[keep]
    patched = tokens.clone()
    patched[keep] = fn(x, y).to(tokens.dtype)
    return from_tokens(patched, layout, act)


@contextmanager
def splice(
    hook_module: nn.Module,
    fn: Callable[[Tensor, Tensor], Tensor],
    target_module: nn.Module | None = None,
    layout: str = "btd",
    exclude_positions: Sequence[int] = (),
    token_mask_fn: Callable[[Any], Tensor] | None = None,
    batch: Any = None,
) -> Iterator[None]:
    """
    Replaces kept tokens of ``target_module``'s output by ``fn(x, y)`` during forward passes.

    ``x`` are the kept tokens of ``hook_module``'s output and ``y`` those of
    ``target_module``'s output. For an SAE both are the same module, so
    ``x is y``; for a transcoder ``hook_module`` is the input (e.g. the norm
    before an MLP) and ``target_module`` the output it predicts (the MLP).
    ``hook_module`` must run before ``target_module``.

    Args:
        hook_module: Module whose output is read.
        fn: Maps kept tokens ``x, y`` to replacements for ``y``.
        target_module: Module whose output is replaced; ``None`` means
            ``hook_module``.
        layout: Activation layout at both modules.
        exclude_positions: Positions left unchanged.
        token_mask_fn: Optional ``fn(batch) -> bool (batch, tokens)`` mask of
            tokens to replace.
        batch: The model input batch passed to ``token_mask_fn``.

    Yields:
        None.

    Raises:
        RuntimeError: If ``target_module`` runs before ``hook_module``.
    """
    if target_module is None or target_module is hook_module:
        with replace_activation(hook_module, lambda act: patch_kept_tokens(
                act, fn, layout, exclude_positions, token_mask_fn, batch)):
            yield
        return

    store: dict[str, Tensor] = {}

    def read(act: Tensor) -> Tensor:
        store["x"] = act
        return act

    def write(act: Tensor) -> Tensor:
        if "x" not in store:
            raise RuntimeError("target_module ran before hook_module")
        return patch_kept_tokens(act, fn, layout, exclude_positions, token_mask_fn, batch, source=store.pop("x"))

    with replace_activation(hook_module, read), replace_activation(target_module, write):
        yield


def split_fields(
    flat: Tensor, kind: str, names: Sequence[str], widths: Sequence[int]
) -> Tensor | tuple[Tensor, ...] | dict[str, Tensor]:
    """
    Splits feature-concatenated activations into the batch structure ``kind``.

    Args:
        flat: Tensor ``(n, sum(widths))``.
        kind: ``"tensor"``, ``"pair"`` (input, target) or ``"dict"``.
        names: Field names, in order.
        widths: Field widths, in order.

    Returns:
        A tensor, a tuple of tensors, or a dict name -> tensor.
    """
    parts = flat.split(list(widths), dim=-1)
    if kind == "tensor":
        return parts[0]
    if kind == "pair":
        return tuple(parts)
    return dict(zip(names, parts))


def run_capture(
    model: nn.Module,
    hook_module: nn.Module,
    batch: Any,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    stop: bool = True,
) -> Tensor:
    """
    Runs ``model`` on ``batch`` and returns the output of ``hook_module``.

    Args:
        model: The model.
        hook_module: Module whose output to capture.
        batch: Input batch (already on the right device).
        input_fn: Maps the batch to ``(args, kwargs)``.
        stop: Stop the forward pass after ``hook_module``.

    Returns:
        The captured activation.
    """
    return run_capture_many(model, [hook_module], batch, input_fn, stop)[0]


def run_capture_many(
    model: nn.Module,
    modules: Sequence[nn.Module],
    batch: Any,
    input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
    stop: bool = True,
) -> list[Tensor]:
    """
    Runs ``model`` once and returns the outputs of several modules.

    Args:
        model: The model.
        modules: Modules whose outputs to capture (tuple outputs are reduced
            to their first element).
        batch: Input batch (already on the right device).
        input_fn: Maps the batch to ``(args, kwargs)``.
        stop: Stop the forward pass once every module has run.

    Returns:
        The captured activations, in the order of ``modules``.

    Raises:
        RuntimeError: If a module did not run.
    """
    store: dict[int, Tensor] = {}

    def make_hook(i: int):
        def hook(_mod, _inp, out):
            store[i] = out[0] if isinstance(out, (tuple, list)) else out
            if stop and len(store) == len(modules):
                raise _StopForward
        return hook

    handles = [m.register_forward_hook(make_hook(i)) for i, m in enumerate(modules)]
    args, kwargs = input_fn(batch)
    try:
        model(*args, **kwargs)
    except _StopForward:
        pass
    finally:
        for h in handles:
            h.remove()
    if len(store) != len(modules):
        raise RuntimeError("some hooked modules did not run during the forward pass")
    return [store[i] for i in range(len(modules))]


class ActivationStore:
    """
    Iterable of shuffled activation batches harvested from a frozen model.

    For transcoders, set ``target_module`` to also capture the output to
    predict; batches are then ``(input, target)`` pairs. To train on several
    hook points at once, pass a dict of modules as ``hook_module``; batches
    are then dicts of the same keys. All hooks are captured in one forward
    pass and shuffled jointly, so row ``i`` of every entry is the same token.

    Fills a buffer of ``buffer_tokens`` tokens, shuffles it, serves batches
    until half of it is used, then refills, so consecutive batches mix tokens
    from many inputs. One pass over ``input_loader`` is one epoch; iterate
    again for more (the trainer does this when ``total_tokens`` is set).

    Args:
        model: Frozen model; run under ``torch.no_grad`` and in eval mode.
        hook_module: Module whose output is the activation to train on (the
            transcoder input when ``target_module`` is set), or a dict of
            name -> module for several hook points with the same token grid.
        input_loader: Iterable of model inputs (images, token dicts, ...).
        batch_tokens: Tokens per served batch.
        buffer_tokens: Shuffle-buffer size in tokens.
        layout: Activation layout, see :data:`LAYOUTS`.
        exclude_positions: Token positions to drop, e.g. ``(0,)`` for CLS or
            BOS, or ``range(5)`` for CLS plus four ViT registers.
        token_mask_fn: Optional ``fn(batch) -> bool (batch, tokens)``
            mask of tokens to keep, e.g. based on token IDs or an attention
            mask. A token is kept only if both this mask and
            ``exclude_positions`` keep it.
        input_fn: Maps a loader batch to model ``(args, kwargs)``.
        device: Device for the model inputs and the buffer.
        storage_dtype: Dtype of the buffer, e.g. ``torch.bfloat16`` to halve
            memory. ``None`` keeps the model's dtype.
        autocast_dtype: Optional autocast dtype for the model forward.
        seed: Shuffle seed.
        target_module: For transcoders: module whose output is the target.
            Batches are then ``(x, y)`` tuples of matching tokens. It must run
            after ``hook_module``; the forward pass stops after it.

    Attributes:
        batch_tokens: Tokens per served batch (read by the trainer).

    Raises:
        ValueError: If ``layout`` is unknown.
    """

    def __init__(
        self,
        model: nn.Module,
        hook_module: nn.Module | dict[str, nn.Module],
        input_loader: Iterable,
        batch_tokens: int = 4096,
        buffer_tokens: int = 2**19,
        layout: str = "btd",
        exclude_positions: Sequence[int] = (),
        token_mask_fn: Callable[[Any], Tensor] | None = None,
        input_fn: Callable[[Any], tuple[tuple, dict]] = default_input_fn,
        device: str | torch.device = "cpu",
        storage_dtype: torch.dtype | None = None,
        autocast_dtype: torch.dtype | None = None,
        seed: int = 0,
        target_module: nn.Module | None = None,
    ):
        if layout not in LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}, expected one of {LAYOUTS}")
        self.model = model
        self.hook_module = hook_module
        self.input_loader = input_loader
        self.batch_tokens = batch_tokens
        self.buffer_tokens = max(buffer_tokens, 2 * batch_tokens)
        self.layout = layout
        self.exclude_positions = tuple(exclude_positions)
        self.token_mask_fn = token_mask_fn
        self.input_fn = input_fn
        self.device = torch.device(device)
        self.storage_dtype = storage_dtype
        self.autocast_dtype = autocast_dtype
        self.generator = torch.Generator().manual_seed(seed)
        self.target_module = target_module
        if isinstance(hook_module, dict):
            if target_module is not None:
                raise ValueError("pass either a dict of hook modules or target_module, not both")
            self.kind, self.names, self.modules = "dict", list(hook_module), list(hook_module.values())
        elif target_module is not None:
            self.kind, self.names, self.modules = "pair", ["activations", "targets"], [hook_module, target_module]
        else:
            self.kind, self.names, self.modules = "tensor", ["activations"], [hook_module]
        self._widths: list[int] = []

    @torch.no_grad()
    def harvest(self, batch: Any) -> Tensor:
        """Returns the kept tokens of one input batch.

        Args:
            batch: A batch from ``input_loader``.

        Returns:
            Activations of the kept tokens, ``(n, d)``; with several hooks
            (or ``target_module``) their activations concatenated along the
            last dimension.
        """
        self.model.eval()
        batch = _to_device(batch, self.device)
        with torch.autocast(self.device.type, dtype=self.autocast_dtype or torch.float32,
                            enabled=self.autocast_dtype is not None):
            acts = run_capture_many(self.model, self.modules, batch, self.input_fn, stop=True)
        tokens = [to_tokens(a, self.layout) for a in acts]
        keep = token_keep_mask(tokens[0], self.exclude_positions, self.token_mask_fn, batch)
        self._widths = [t.shape[-1] for t in tokens]
        flat = torch.cat([t[keep].to(tokens[0].dtype) for t in tokens], -1)
        return flat.to(self.storage_dtype) if self.storage_dtype else flat

    def _serve(self, batch: Tensor) -> Tensor | tuple[Tensor, Tensor] | dict[str, Tensor]:
        return split_fields(batch, self.kind, self.names, self._widths)

    def __iter__(self) -> Iterator[Tensor]:
        buffer: list[Tensor] = []
        size = 0
        for batch in self.input_loader:
            flat = self.harvest(batch)
            buffer.append(flat)
            size += flat.shape[0]
            if size >= self.buffer_tokens:
                data = torch.cat(buffer)
                data = data[torch.randperm(data.shape[0], generator=self.generator).to(data.device)]
                n_serve = (data.shape[0] // 2) // self.batch_tokens * self.batch_tokens
                for i in range(0, n_serve, self.batch_tokens):
                    yield self._serve(data[i : i + self.batch_tokens])
                buffer, size = [data[n_serve:]], data.shape[0] - n_serve
        if size:
            data = torch.cat(buffer)
            data = data[torch.randperm(data.shape[0], generator=self.generator).to(data.device)]
            for i in range(0, data.shape[0] - self.batch_tokens + 1, self.batch_tokens):
                yield self._serve(data[i : i + self.batch_tokens])
