"""Per-example gradient clipping primitives (algorithm-agnostic).

Headline entry points shared by DP-SGD and DP-FTRL:

- :func:`clipped_grad` — differentiate + clip + sum the gradients
- :func:`per_group` — factory for :class:`opaque.types.PerGroup` groupings

DP-SGD-specific variants (``adaptive_clipped_grad``, ``auto_clipped_grad``)
live in :mod:`opaque.dpsgd.clipping`.

Power-user building blocks (``clipped_fun``, ``clip_pytree``,
``auto_scale_pytree``) live in :mod:`opaque.clipping.fun`. State and
auxiliary dataclasses (``FixedClipState``, ``ClippedGradAux``,
``ClippedFunAux``, ``ClipPytreeAux``) live in :mod:`opaque.clipping.types`.
The cross-cutting DP types (``ClipState`` base, ``ClippedPytree``,
``PerGroup``, ``MaxNorm``, ``clipped()`` factory) live in
:mod:`opaque.types`.

To synchronize clipping state across distributed ranks, use
:func:`opaque.distributed.sync`; it auto-dispatches by type to the right
handler without you having to import it by name.
"""

from opaque.clipping._clipped_grad import clipped_grad
from opaque.clipping._per_group import per_group

import opaque.clipping._distributed  # noqa: F401  (registers sync handlers)

__all__ = [
    "clipped_grad",
    "per_group",
]
