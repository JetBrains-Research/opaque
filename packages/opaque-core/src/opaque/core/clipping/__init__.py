"""Per-example gradient clipping primitives (algorithm-agnostic).

Core exposes the generic clipping building blocks (pytree clip, fixed
``clipped_fun`` / ``clipped_grad``, the base :class:`ClipState`, and the
:class:`~opaque.core.clipping.per_group.PerGroup` type). DP-SGD-specific
variants (``adaptive_*``, ``auto_*``) live in :mod:`opaque.dpsgd.clipping`.

To synchronize a clipping state or aux object across distributed ranks,
call :func:`opaque.distributed.sync` with the object — it dispatches on
type to the right handler without you having to import it by name.
"""

from opaque.core.clipping.clipped_fun import ClippedFunAux, clipped_fun
from opaque.core.clipping.clipped_grad import ClippedGradAux, clipped_grad
from opaque.core.clipping.per_group import PerGroup, per_group
from opaque.core.clipping.pytree import auto_scale_pytree, clip_pytree
from opaque.core.clipping.types import ClipState, FixedClipState

__all__ = [
    "clip_pytree",
    "auto_scale_pytree",
    "clipped_fun",
    "clipped_grad",
    "ClipState",
    "FixedClipState",
    "ClippedFunAux",
    "ClippedGradAux",
    "PerGroup",
    "per_group",
]
