"""Convenience helpers for common RNG patterns.

This module provides a non-deterministic key helper for prototyping.
"""

from __future__ import annotations

import secrets

from ._engine import RngKey, key


def random_key() -> RngKey:
    """Create a non-deterministic key using system entropy.

    Useful for prototyping when reproducibility is not critical. For production
    training, prefer ``key()`` with an explicit seed and ``fold_in()`` for
    per-step / per-rank derivation. This is an explicit convenience boundary:
    deterministic engine APIs never call it implicitly.

    Returns:
        A randomly initialized RngKey.

    Example:
        >>> from opaque.api.engine.random import random_key
        >>> from opaque.dpsgd.noise import gaussian_noise
        >>> k = random_key()
        >>> noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=k)
    """
    random_seed = secrets.randbits(64)
    return key(random_seed)


__all__ = ["random_key"]
