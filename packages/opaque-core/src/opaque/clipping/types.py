"""Public type definitions for :mod:`opaque.clipping`.

Re-exports the clipping-specific state and auxiliary dataclasses for
type annotations. The cross-cutting DP types (``ClipState`` base,
``ClippedPytree``, ``PerGroup``, ``MaxNorm``, ``clipped()`` factory)
live in :mod:`opaque.types`.
"""

from __future__ import annotations

from opaque.clipping._clipped_fun import ClippedFunAux
from opaque.clipping._clipped_fun import FixedClipState
from opaque.clipping._clipped_grad import ClippedGradAux
from opaque.clipping._pytree import ClipPytreeAux
from opaque.types import PerGroup


def _norm_state_dict(norm: float | PerGroup) -> dict | float:
    """Encode a clipping-norm field (scalar or PerGroup) as a tagged value.

    Used by adaptive clip / sampler state-dict serialisation in
    ``opaque.dpsgd``.  Lives here because it shapes the on-disk
    representation of ``ClippedPytree.max_norm`` payloads.
    """
    if isinstance(norm, PerGroup):
        return {"__type__": "PerGroup", **norm.state_dict()}
    return float(norm)


def _norm_from_state(data: dict | float | int) -> float | PerGroup:
    """Decode the inverse of :func:`_norm_state_dict`."""
    if isinstance(data, dict) and data.get("__type__") == "PerGroup":
        return PerGroup.from_state_dict(data)
    return float(data)


__all__ = [
    "FixedClipState",
    "ClippedGradAux",
    "ClippedFunAux",
    "ClipPytreeAux",
]
