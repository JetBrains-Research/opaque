"""Opaque core: shared primitives for DP training.

This package provides algorithm-agnostic building blocks used by both
DP-SGD (``opaque.dpsgd``) and DP-FTRL (``opaque.dpftrl``) mechanisms,
plus the auditing and HuggingFace integration layers.

Contents
--------
- ``opaque.core.random``: JAX-style RNG keys and a PyTorch generator bridge.
- ``opaque.core.pytree``: pytree helpers (``tree_map``, ``global_norm``, ...).
- ``opaque.core.clipping``: per-example gradient clipping primitives,
  including the :class:`~opaque.core.clipping.per_group.PerGroup` type.
- ``opaque.core.sampling``: collation helpers for variable-size / empty batches.
- ``opaque.core.noise.types``: the generic ``NoiseMechanism`` base class.

User-facing primitives live at the namespace root (not inside ``opaque.core``):
``opaque.distributed`` (collectives, gradients, sync), ``opaque.functional``
(``make_functional``, ``with_batch_dim``). Performance/profiling tools ship
in ``opaque.performance``.

Partition policy
----------------
Only truly algorithm-agnostic code lives in ``opaque.core``. Mechanisms
specific to DP-SGD (Gaussian noise, adaptive clipping, truncated Poisson,
per-group noise allocation) ship in ``opaque.dpsgd``. Mechanisms specific
to DP-FTRL (BLT/Toeplitz/BSR noise, b-min-sep sampling, cyclic Poisson)
ship in ``opaque.dpftrl``.
"""

from opaque.core import clipping, noise, pytree, random, sampling
from opaque.core.clipping import (
    ClipState,
    FixedClipState,
    auto_scale_pytree,
    clip_pytree,
    clipped_fun,
    clipped_grad,
)
from opaque.core.clipping.per_group import PerGroup, per_group
from opaque.core.pytree import (
    global_norm,
    merge,
    partition,
    tree_leaves,
    tree_map,
)
from opaque.core.random import RngKey, fold_in, generator_from_key, key, split

__version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    # Subpackages
    "clipping",
    "noise",
    "pytree",
    "random",
    "sampling",
    # RNG
    "RngKey",
    "key",
    "split",
    "fold_in",
    "generator_from_key",
    # Clipping
    "ClipState",
    "FixedClipState",
    "clip_pytree",
    "auto_scale_pytree",
    "clipped_fun",
    "clipped_grad",
    # Per-group
    "PerGroup",
    "per_group",
    # Pytree
    "tree_map",
    "tree_leaves",
    "global_norm",
    "partition",
    "merge",
]
