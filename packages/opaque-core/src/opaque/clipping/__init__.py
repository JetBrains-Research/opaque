"""Per-example gradient clipping primitives (algorithm-agnostic).

Exports the fixed-clipping entry points shared by DP-SGD and DP-FTRL:

- :func:`clipped_grad` — differentiate + clip + sum the gradients
- :func:`per_group` — factory for :class:`PerGroup` groupings

DP-SGD-specific variants (``adaptive_clipped_grad``, ``auto_clipped_grad``)
live in :mod:`opaque.dpsgd.clipping`.

Less-common building blocks are reachable via the submodules:

- :mod:`opaque.clipping.clipped_fun` — clip + sum arbitrary functions
- :mod:`opaque.clipping.pytree` — ``clip_pytree``, ``auto_scale_pytree``

Data classes (``ClipState``, ``FixedClipState``, ``ClippedGradAux``,
``PerGroup``) are importable from this module for type annotations but are
intentionally not part of ``__all__`` — the public surface is functional.

To synchronize clipping state across distributed ranks, use
:func:`opaque.distributed.sync`; it auto-dispatches by type to the right
handler without you having to import it by name.
"""

from opaque.clipping.clipped_grad import ClippedGradAux as ClippedGradAux
from opaque.clipping.clipped_grad import clipped_grad
from opaque.clipping.per_group import PerGroup as PerGroup
from opaque.clipping.per_group import per_group
from opaque.clipping.types import ClipState as ClipState
from opaque.clipping.types import FixedClipState as FixedClipState

# Side-effect import: registers FixedClipState / ClippedGradAux sync handlers
# with opaque.distributed.sync().
from opaque.clipping import distributed as _distributed  # noqa: F401

__all__ = [
    "clipped_grad",
    "per_group",
]
