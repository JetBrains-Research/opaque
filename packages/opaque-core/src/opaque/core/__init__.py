"""Opaque core: shared primitives for DP training.

This package provides algorithm-agnostic building blocks used by both
DP-SGD (``opaque.dpsgd``) and matrix-factorization (``opaque.mf``)
mechanisms, plus the auditing and HuggingFace integration layers.

Contents
--------
- ``opaque.core.random``: JAX-style RNG keys and a PyTorch generator bridge.
- ``opaque.core.clipping``: per-example gradient clipping primitives.
- ``opaque.core.sampling``: Poisson sampling, collation, distributed shards.
- ``opaque.core.noise.types``: the generic ``NoiseMechanism`` base class.
- ``opaque.core.distributed``: distributed training helpers (collectives).
- ``opaque.core.profiling``: memory / timing profiler utilities.
- ``opaque.core.utils``: pytree, per-group, and functional helpers.

Partition policy
----------------
Only truly algorithm-agnostic code lives in ``opaque.core``. Mechanisms
specific to DP-SGD (Gaussian noise, adaptive clipping, truncated Poisson)
ship in ``opaque.dpsgd``. Mechanisms specific to matrix factorization
(BLT/Toeplitz/BSR noise, b-min-sep sampling, cyclic Poisson) ship in
``opaque.mf``.
"""

from opaque.core import clipping, distributed, noise, profiling, random, sampling, utils
from opaque.core.clipping import (
    ClipState,
    FixedClipState,
    auto_scale_pytree,
    clip_pytree,
    clipped_fun,
    clipped_grad,
)
from opaque.core.random import RngKey, fold_in, generator_from_key, key, split
from opaque.core.sampling import PoissonSampler, poisson_collate
from opaque.core.utils import (
    PerGroup,
    global_norm,
    merge,
    partition,
    per_group,
    tree_leaves,
    tree_map,
)

__version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    # Subpackages
    "clipping",
    "distributed",
    "noise",
    "profiling",
    "random",
    "sampling",
    "utils",
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
    # Sampling
    "PoissonSampler",
    "poisson_collate",
    # Utils / pytree
    "PerGroup",
    "per_group",
    "tree_map",
    "tree_leaves",
    "global_norm",
    "partition",
    "merge",
]
