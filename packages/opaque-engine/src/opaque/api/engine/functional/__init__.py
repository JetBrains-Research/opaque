"""Functional utilities for model and batch-oriented callables."""

import functools
from collections.abc import Callable
from typing import Any

from opaque.api.engine import ops


def _squeeze_output(output):
    """Return a pytree with leading batch dimensions removed from arrays."""
    from opaque.api.engine.pytree import tree_map

    return tree_map(
        lambda value: (
            ops.squeeze(value, axis=0)
            if ops.is_array(value) and ops.shape(value)
            else value
        ),
        output,
    )


def with_batch_dim(
    fn: Callable,
    batch_argnums: int | tuple[int, ...] = (),
    batch_kwargs: tuple[str, ...] | dict[str, int] = (),
    *,
    min_ndim: int | None = None,
) -> Callable:
    """Wrap a function to add a leading batch dimension to specified arguments.

    Use this when a model expects a batch dimension but the function is called
    per-example under ``vmap`` where the batch dim has been stripped.

    Two modes controlled by ``min_ndim``:

    - ``min_ndim=None`` (default): always unsqueeze specified args, never
      touch output. This is the user-facing mode for wrapping loss functions.

    - ``min_ndim=N`` (idempotent mode): only unsqueeze tensors with
      ``ndim < threshold``, and squeeze top-level output tensors when any
      unsqueezing happened. This is the model-forward mode — a no-op for
      already-batched inputs.

    Args:
        fn: Function to wrap.
        batch_argnums: Which positional arguments should get an extra leading
            dimension via ``unsqueeze(0)``. Int or tuple of ints.
        batch_kwargs: Keyword arguments to process. A tuple of names uses the
            global ``min_ndim`` threshold. A dict maps each kwarg name to its
            own ndim threshold (e.g., ``{"input_ids": 2, "inputs_embeds": 3}``).
        min_ndim: Global ndim threshold. ``None`` means always unsqueeze.
            An int means only unsqueeze when ``tensor.ndim < min_ndim``.

    Returns:
        Wrapped function.

    Example — loss function mode (always unsqueeze)::

        >>> def loss_fn(params, tokens):
        ...     output = model(params, tokens)  # model expects (batch, seq)
        ...     return output.loss
        >>> # Under vmap, tokens is (seq,); wrap so model sees (1, seq)
        >>> loss_fn = with_batch_dim(loss_fn, batch_argnums=1)

    Example — model forward mode (idempotent)::

        >>> # Only unsqueeze when inputs lack batch dim; squeeze output
        >>> model.forward = with_batch_dim(
        ...     model.forward,
        ...     batch_kwargs={"input_ids": 2, "inputs_embeds": 3},
        ...     min_ndim=2,
        ... )
    """
    if getattr(fn, "_opaque_batchified", False):
        return fn

    import inspect

    from opaque.api.engine.pytree import tree_map

    if isinstance(batch_argnums, int):
        batch_argnums = (batch_argnums,)

    # Normalize batch_kwargs: tuple → dict using global min_ndim
    if isinstance(batch_kwargs, tuple):
        batch_kwargs_dict: dict[str, int | None] = dict.fromkeys(batch_kwargs, min_ndim)
    else:
        batch_kwargs_dict = dict(batch_kwargs)

    conditional = min_ndim is not None

    # Pre-compute: positional indices of batch_kwargs parameters in fn's
    # signature, so we can catch them even when passed positionally.
    _positional_kwarg_indices: dict[int, str] = {}
    if batch_kwargs_dict:
        try:
            sig = inspect.signature(fn)
            for idx, (name, param) in enumerate(sig.parameters.items()):
                if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                    if name in batch_kwargs_dict:
                        _positional_kwarg_indices[idx] = name
                elif param.kind == param.VAR_POSITIONAL:
                    break
        except (ValueError, TypeError):
            pass

    def _needs_unsqueeze(tensor: Any, threshold: int | None) -> bool:
        if threshold is None:
            return True
        return len(ops.shape(tensor)) < threshold

    # ``functools.wraps`` propagates ``fn.__wrapped__`` (and sets it to
    # ``fn`` itself) so callers can walk the chain back to the original
    # function via ``inspect.signature(wrapper, follow_wrapped=True)`` — the
    # default behavior of :func:`inspect.signature`.  Concretely: when this
    # wraps a transformers ``forward(input_ids=None, attention_mask=None,
    # …)``, ``inspect.signature`` resolves to the real named parameters
    # instead of just ``(*args, **kwargs)``.  HF-style column pruning that
    # introspects ``model.forward`` (and any other downstream consumer of
    # the signature) keeps working after batchifying.
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        unsqueezed = False
        args_list = list(args)

        # Move batch_kwargs that were passed positionally into kwargs.
        for idx in sorted(_positional_kwarg_indices, reverse=True):
            if idx < len(args_list):
                name = _positional_kwarg_indices[idx]
                if name not in kwargs:
                    kwargs[name] = args_list.pop(idx)

        # Process positional args
        for i in batch_argnums:

            def _unsqueeze_arg(x, _threshold=min_ndim):
                nonlocal unsqueezed
                if ops.is_array(x) and _needs_unsqueeze(x, _threshold):
                    unsqueezed = True
                    return ops.expand_dims(x, 0)
                return x

            if i < len(args_list):
                args_list[i] = tree_map(_unsqueeze_arg, args_list[i])

        # Process keyword args
        for name, threshold in batch_kwargs_dict.items():
            if name not in kwargs or kwargs[name] is None:
                continue
            val = kwargs[name]
            if ops.is_array(val) and _needs_unsqueeze(val, threshold):
                unsqueezed = True
                kwargs[name] = ops.expand_dims(val, 0)

        result = fn(*args_list, **kwargs)

        if conditional and unsqueezed:
            result = _squeeze_output(result)

        return result

    wrapper._opaque_batchified = True
    return wrapper


from opaque.api.engine.functional._collate import empty_collate  # noqa: E402

__all__ = ["empty_collate", "with_batch_dim"]
