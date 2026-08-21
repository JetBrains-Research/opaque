"""Generic RNG engine with JAX-style key semantics.

This module provides explicit, immutable RNG state transitions inspired by JAX:

- ``key(seed)`` creates an initial key
- ``split(key, n)`` derives independent child keys
- ``fold_in(key, data)`` domain-separates by deterministic metadata

Integer and string folds occupy disjoint spaces, and that split is load-bearing:
integers are the caller's (steps, ranks, indices, everything ``split`` returns)
while a string tag roots a mechanism's own key space.  A mechanism that derives
straight from ``fold_in(key, step)`` collides with every other mechanism handed
the same key.  ``docs/reference/rng.md`` states the convention.

Seeds are canonical unsigned 64-bit integers.  ``fold_in`` accepts only
integers and strings; it preserves the existing derivation encoding for signed
128-bit integers and supports larger integers without truncation.  Provider
wheels may bridge keys to framework-native generator objects, but a key's
derivation is independent of their global RNG state.
"""

from __future__ import annotations

import dataclasses
import hashlib

from opaque.api.engine.primitive import PrimitiveTier, primitive


def _to_uint64(value: int) -> int:
    return value & ((1 << 64) - 1)


def _int_to_bytes(value: int) -> bytes:
    """Encode an integer while retaining the established 128-bit encoding."""
    try:
        return value.to_bytes(16, byteorder="little", signed=True)
    except OverflowError:
        num_bytes = max(17, (value.bit_length() + 8) // 8)
        return value.to_bytes(num_bytes, byteorder="little", signed=True)


def _stable_hash64(*parts: object) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        if isinstance(part, int):
            h.update(_int_to_bytes(part))
        else:
            h.update(str(part).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


@dataclasses.dataclass(frozen=True)
class RngKey:
    """Immutable RNG key.

    Attributes:
        seed: Canonical unsigned 64-bit seed value used as key material.
        impl: Logical implementation identifier.
    """

    seed: int
    impl: str = "opaque_threefry_like"

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError(f"seed must be int, got {type(self.seed)}")
        object.__setattr__(self, "seed", _to_uint64(self.seed))


def key(seed: int) -> RngKey:
    """Create a PRNG key from an integer seed normalized modulo ``2**64``."""
    return RngKey(seed=seed)


def fold_in(rng_key: RngKey, *data: int | str) -> RngKey:
    """Fold one or more values into a key to deterministically derive a new key.

    Accepts a variable number of int/str arguments. Each value is folded
    sequentially, so ``fold_in(k, a, b)`` equals ``fold_in(fold_in(k, a), b)``.

    Integers and strings are hashed down disjoint paths: ``fold_in(k, 1)`` and
    ``fold_in(k, "1")`` are different keys, and no sequence of integer folds
    can reach a key derived through a string fold.  That separation is what
    makes the domain convention work, so the two kinds are used for different
    jobs:

    - **Integers are the caller's.** Steps, ranks, epochs, leaf and group
      indices, and every key :func:`split` hands back are integer folds of a
      key you already hold.
    - **Strings root a mechanism.** A mechanism that draws randomness folds one
      unique string tag into the key it was given, once, and derives everything
      else beneath that tag.

    Skipping the string root is the failure this convention exists to prevent:
    ``fold_in(key, step)`` is the derivation *every* mechanism writes, so two
    mechanisms handed the same base key draw byte-identical noise, and nothing
    — not a test, an error, or an accountant — reports it.  See
    :doc:`the RNG reference </reference/rng>` for the convention and the tags
    Opaque's own mechanisms already occupy.

    Args:
        rng_key: Base key.
        *data: One or more int or str values to fold in sequentially.

    Returns:
        A new RngKey derived from the base key and all folded values.

    Raises:
        TypeError: If ``rng_key`` is not an :class:`RngKey` or any value is not
            an int (excluding ``bool``) or str.
        ValueError: If no data values are provided.

    Example:
        >>> k = key(42)
        >>> stream = fold_in(k, "mylab.rare_events")  # root your mechanism
        >>> fold_in(stream, step)                     # then step, rank, leaf...
        >>> fold_in(stream, step, rank)               # equals fold_in twice
    """
    if not isinstance(rng_key, RngKey):
        raise TypeError(f"rng_key must be RngKey, got {type(rng_key)}")
    if not data:
        raise ValueError("fold_in requires at least one data argument")
    result = rng_key
    for d in data:
        if not isinstance(d, (int, str)) or isinstance(d, bool):
            raise TypeError(f"data must be int or str, got {type(d)}")
        mixed = _stable_hash64(result.seed, d)
        result = RngKey(seed=mixed, impl=result.impl)
    return result


def split(rng_key: RngKey, num: int = 2) -> tuple[RngKey, ...]:
    """Split a key into ``num`` independent child keys.

    Defined as ``fold_in(rng_key, i) for i in range(num)``, so the children are
    exactly the integer folds of ``rng_key`` — which is why a mechanism roots
    itself with a string tag rather than a small integer: an integer root would
    hand out the same keys ``split`` does.  See :func:`fold_in`.
    """
    if not isinstance(num, int) or isinstance(num, bool):
        raise TypeError(f"num must be int, got {type(num)}")
    if num < 1:
        raise ValueError(f"num must be >= 1, got {num}")
    return tuple(fold_in(rng_key, i) for i in range(num))


@primitive(tier=PrimitiveTier.CORE)
def normal(
    rng_key: RngKey,
    shape: tuple[int, ...] | list[int],
    *,
    dtype: object | None = None,
    like: object | None = None,
) -> object:
    """Draw a standard normal sample determined solely by immutable ``rng_key``.

    The key is input-only: repeated calls with the same key and arguments
    return the same backend-native sample and do not mutate hidden generator
    state.  ``like`` selects a backend's placement and default dtype.

    That determinism is the whole contract, which makes deriving keys the
    caller's obligation rather than a convenience:

    - **Every distinct draw needs a distinct key.** Calling twice with the same
      key replays the same values; it does not continue a stream.  Advance with
      :func:`fold_in` (or :func:`split`) between draws.
    - **Root your mechanism before you derive.** Fold a unique string tag in
      once, then derive per-step and per-leaf keys beneath it, so a base key
      shared with another mechanism still yields independent noise.  See
      :func:`fold_in`.
    - **One key, two shapes are not independent.** ``normal(k, (4,))`` and
      ``normal(k, (8,))`` share a prefix rather than being separate samples.
      Fold before changing shape.

    ``dtype`` (or ``like``'s dtype) is the dtype of the returned sample, not a
    licence to compute in it.  A provider draws at no less than ``float32``
    internally and returns in the requested dtype, so the draw itself is not
    coarsened — but arithmetic that follows it in a low-precision dtype is.
    Mechanisms that add noise to a low-precision leaf upcast the whole
    expression and downcast only the result; see
    :doc:`/user-guide/precision`.

    Args:
        rng_key: Key determining the sample. Must differ for every draw.
        shape: Shape of the sample; ``()`` draws a scalar.
        dtype: Dtype of the returned sample. Defaults to ``like``'s dtype when
            ``like`` is given, otherwise the provider's default float dtype.
        like: Array whose device (and, absent ``dtype``, dtype) the sample
            adopts.

    Returns:
        A backend-native array of shape ``shape`` holding standard normal
        samples — mean 0, standard deviation 1. Scale and shift it yourself to
        obtain other Gaussians.
    """
    raise NotImplementedError


__all__ = ["RngKey", "fold_in", "key", "normal", "split"]
