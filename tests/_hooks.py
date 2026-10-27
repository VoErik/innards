"""Plain PyTorch forward hooks, as references independent of nnsight."""

from contextlib import contextmanager


@contextmanager
def capture_output(module):
    """Yields a dict whose ``"act"`` holds the module's output (first element for tuples)."""
    store = {}

    def hook(_mod, _inp, out):
        store["act"] = out[0] if isinstance(out, (tuple, list)) else out

    handle = module.register_forward_hook(hook)
    try:
        yield store
    finally:
        handle.remove()


@contextmanager
def replace_output(module, fn):
    """Replaces the module's output (first element for tuples) by ``fn(output)``."""

    def hook(_mod, _inp, out):
        if isinstance(out, tuple):
            return (fn(out[0]), *out[1:])
        return fn(out)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
