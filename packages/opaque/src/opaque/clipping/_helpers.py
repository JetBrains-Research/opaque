"""Internal helper utilities for clipping operations."""

from collections.abc import Callable


def normalize_to_tuple(value: int | tuple[int, ...]) -> tuple[int, ...]:
    """Normalize int or tuple of ints to tuple.

    Args:
        value: Single int or tuple of ints

    Returns:
        Tuple of ints
    """
    if isinstance(value, int):
        return (value,)
    return value


def normalize_fun_to_return_aux(fun: Callable, has_aux: bool) -> Callable:
    """Normalize function to always return (value, aux) tuple.

    Args:
        fun: Function to normalize.
        has_aux: Whether fun already returns aux.

    Returns:
        Normalized function that always returns (value, aux).
    """
    if has_aux:
        return fun
    else:
        return lambda *args, **kwargs: (fun(*args, **kwargs), ())


__all__ = ["normalize_to_tuple", "normalize_fun_to_return_aux"]
