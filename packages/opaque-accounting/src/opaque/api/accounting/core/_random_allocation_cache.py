"""Memoised wrappers for the random-allocation native PLD primitives.

Contributor-internal: not re-exported from the ``opaque.accounting`` façade.

The random-allocation transform is O(G²) in the convolution grid and is
rebuilt on every distinct step count by its horizon callers (k-out-of-t
prefixes, identity balls-in-bins). The transform is deterministic, so the
epoch and prefix primitives are memoised here on
``(noise_multiplier, t / total_steps, k / released_steps, resolved
DiscretizationConfig)`` — every input that affects the output. The caches are
bounded LRUs over immutable Python :class:`Pld` objects (no native handles),
so they are self-limiting and need no byte budget or destructor.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from . import _native

if TYPE_CHECKING:
    from ._base import Pld
    from .discretization import DiscretizationConfig

#: Cached entries are one PLD with at most ``max_conv_grid`` bins per PMF
#: (~0.5 MB per entry at the default config), so eight entries per cache are
#: a few MB — comparable to one :func:`horizon_pld_cache` slice.
_MAXSIZE = 8


@functools.lru_cache(maxsize=_MAXSIZE)
def epoch_pld(
    noise_multiplier: float,
    t: int,
    k: int,
    config: DiscretizationConfig,
) -> Pld:
    """Cached ``_native.random_allocation_gaussian_pld``.

    ``config`` must be the resolved Python-side :class:`DiscretizationConfig`
    (frozen and hashable); the native conversion runs only on a cache miss.
    """
    return _native.random_allocation_gaussian_pld(
        noise_multiplier,
        t,
        k,
        config.to_native(),
    )


@functools.lru_cache(maxsize=_MAXSIZE)
def prefix_pld(
    noise_multiplier: float,
    total_steps: int,
    released_steps: int,
    config: DiscretizationConfig,
) -> Pld:
    """Cached ``_native.random_allocation_gaussian_prefix_pld``."""
    return _native.random_allocation_gaussian_prefix_pld(
        noise_multiplier,
        total_steps,
        released_steps,
        config.to_native(),
    )


def clear_random_allocation_caches() -> None:
    """Drop all memoised random-allocation PLDs."""
    epoch_pld.cache_clear()
    prefix_pld.cache_clear()
