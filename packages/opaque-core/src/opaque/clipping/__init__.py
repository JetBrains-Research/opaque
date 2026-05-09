"""Per-example gradient clipping primitives (algorithm-agnostic).

Headline entry points shared by DP-SGD and DP-FTRL:

- :func:`clipped_grad` — differentiate + fixed-threshold clip + sum the
  gradients
- :func:`auto_clipped_grad` — differentiate + AUTO-S smooth-scale + sum
  the gradients (Bu et al., NeurIPS 2023)
- :func:`per_group` — factory for :class:`opaque.types.PerGroup` groupings

Both fixed clipping and AUTO-S produce a constant per-record sensitivity
bound that is independent of the input data, so they compose with any
mechanism in :mod:`opaque.dpsgd.noise` (Gaussian / truncated Gaussian)
or :mod:`opaque.dpftrl.noise` (identity / band-MF / BLT / BiSR / BSR /
λ-CGD).  The DP-SGD-specific :func:`adaptive_clipped_grad` lives in
:mod:`opaque.dpsgd.clipping` because its threshold drifts across steps
and therefore violates the constant-sensitivity assumption that
matrix-factorization privacy proofs rely on.

Power-user building blocks (``clipped_fun``, ``auto_clipped_fun``,
``clip_pytree``, ``auto_scale_pytree``) live in :mod:`opaque.clipping.fun`.
State and auxiliary dataclasses (``FixedClipState``, ``AutoClipState``,
``ClippedGradAux``, ``AutoClippedGradAux``, ``ClippedFunAux``,
``AutoClippedFunAux``, ``ClipPytreeAux``) live in
:mod:`opaque.clipping.types`. The cross-cutting DP types (``ClipState``
base, ``ClippedPytree``, ``PerGroup``, ``MaxNorm``, ``clipped()`` factory)
live in :mod:`opaque.types`.

To synchronize clipping state across distributed ranks, use
:func:`opaque.distributed.sync`; it auto-dispatches by type to the right
handler without you having to import it by name.
"""

from opaque.clipping._auto import auto_clipped_grad
from opaque.clipping._clipped_grad import clipped_grad
from opaque.clipping._per_group import per_group

import opaque.clipping._distributed  # noqa: F401  (registers sync handlers)

__all__ = [
    "auto_clipped_grad",
    "clipped_grad",
    "per_group",
]
