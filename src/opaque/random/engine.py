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

_MAX_TORCH_SEED = 2**63 - 1


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


def fold_in(rng_key: RngKey, data: int | str) -> RngKey:
    """Fold additional data into a key to deterministically derive a new key."""
    if not isinstance(data, (int, str)):
        raise TypeError(f"data must be int or str, got {type(data)}")
    mixed = _stable_hash64(rng_key.seed, data)
    return RngKey(seed=mixed, impl=rng_key.impl)


def split(rng_key: RngKey, num: int = 2) -> tuple[RngKey, ...]:
    """Split a key into ``num`` independent child keys."""
    if num < 1:
        raise ValueError(f"num must be >= 1, got {num}")
    return tuple(fold_in(rng_key, i) for i in range(num))


def generator_from_key(rng_key: RngKey) -> torch.Generator:
    """Create a deterministic ``torch.Generator`` from a key."""
    seed = rng_key.seed % _MAX_TORCH_SEED
    return torch.Generator().manual_seed(seed)


__all__ = ["RngKey", "key", "split", "fold_in", "generator_from_key"]
