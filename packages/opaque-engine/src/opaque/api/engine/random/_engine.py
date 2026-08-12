"""Generic RNG engine with JAX-style key semantics.

This module provides explicit, immutable RNG state transitions inspired by JAX:

- ``key(seed)`` creates an initial key
- ``split(key, n)`` derives independent child keys
- ``fold_in(key, data)`` domain-separates by deterministic metadata

The engine is backend-agnostic and can be bridged to PyTorch via
``generator_from_key`` when APIs require ``torch.Generator``.
"""

from __future__ import annotations

import dataclasses
import hashlib

import torch

from opaque.api.engine.primitive import Primitive

_MAX_TORCH_SEED = 2**63 - 1
_normal = Primitive("opaque.random.normal", tier="core")


def _to_uint64(value: int) -> int:
    return value & ((1 << 64) - 1)


def _stable_hash64(*parts: object) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        if isinstance(part, int):
            h.update(part.to_bytes(16, byteorder="little", signed=True))
        else:
            h.update(str(part).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


@dataclasses.dataclass(frozen=True)
class RngKey:
    """Immutable RNG key.

    Attributes:
        seed: Canonical seed value used as key material.
        impl: Logical implementation identifier.
    """

    seed: int
    impl: str = "opaque_threefry_like"


def key(seed: int) -> RngKey:
    """Create a PRNG key from an integer seed."""
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed)}")
    return RngKey(seed=_to_uint64(seed))


def fold_in(rng_key: RngKey, *data: int | str) -> RngKey:
    """Fold one or more values into a key to deterministically derive a new key.

    Accepts a variable number of int/str arguments. Each value is folded
    sequentially, so ``fold_in(k, a, b)`` equals ``fold_in(fold_in(k, a), b)``.

    Args:
        rng_key: Base key.
        *data: One or more int or str values to fold in sequentially.

    Returns:
        A new RngKey derived from the base key and all folded values.

    Raises:
        TypeError: If any value is not int or str.
        ValueError: If no data values are provided.

    Example:
        >>> k = key(42)
        >>> fold_in(k, 0)           # single value
        >>> fold_in(k, step, rank)  # multiple values (step then rank)
    """
    if not data:
        raise ValueError("fold_in requires at least one data argument")
    result = rng_key
    for d in data:
        if not isinstance(d, (int, str)):
            raise TypeError(f"data must be int or str, got {type(d)}")
        mixed = _stable_hash64(result.seed, d)
        result = RngKey(seed=mixed, impl=result.impl)
    return result


def split(rng_key: RngKey, num: int = 2) -> tuple[RngKey, ...]:
    """Split a key into ``num`` independent child keys."""
    if num < 1:
        raise ValueError(f"num must be >= 1, got {num}")
    return tuple(fold_in(rng_key, i) for i in range(num))


def normal(
    rng_key: RngKey,
    shape: tuple[int, ...] | list[int],
    *,
    dtype: object | None = None,
    like: object | None = None,
) -> object:
    """Draw a normal sample determined solely by immutable ``rng_key``.

    The key is input-only: repeated calls with the same key and arguments
    return the same backend-native sample and do not mutate hidden generator
    state.  ``like`` selects a backend's placement and default dtype.
    """
    return _normal(rng_key, shape, dtype=dtype, like=like)


def generator_from_key(rng_key: RngKey) -> torch.Generator:
    """Create a deterministic ``torch.Generator`` from a key."""
    seed = rng_key.seed % _MAX_TORCH_SEED
    return torch.Generator().manual_seed(seed)


__all__ = ["RngKey", "fold_in", "generator_from_key", "key", "normal", "split"]
