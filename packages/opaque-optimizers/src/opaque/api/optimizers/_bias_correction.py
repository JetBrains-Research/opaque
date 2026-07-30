"""DP noise-variance bias correction (φ-EMA).

Shared primitive for any second-moment optimizer (Adam, AdEMAMix,
Adafactor, …) that wants to subtract the noise variance from its
``v̂_t`` estimate under DP-SGD.

Reference:
    Chooi et al., "DP-AdamW: Investigating Decoupled Weight Decay and
    Bias Correction in Private Deep Learning", arXiv:2511.07843.

Math.  Let ``Φ_t = σ_t²`` be the per-step noise *variance*.  The
``φ`` EMA tracks the same β₂-weighted average as the second-moment
EMA::

    φ_t = β₂ φ_{t-1} + (1 − β₂) Φ_t

After bias correction by ``1 − β₂^t``, ``φ̂_t = φ_t / (1 − β₂^t)`` is
an unbiased estimate of the noise-variance contribution to ``v̂_t``::

    v̂_corrected = max(v̂_t − φ̂_t, floor)

The floor is a small positive constant so the corrected denominator
``√v̂_corrected + ε`` cannot collapse to zero when noise dominates
the gradient signal.

This module is mechanism-agnostic.  ``noise_stddev`` is a number (or a
``PerGroup`` of numbers) supplied by the caller; the noise *generation*
lives elsewhere (``opaque.dpsgd.noise``, ``opaque.dpftrl.noise``).
"""

from __future__ import annotations

from typing import Any

from opaque.types import PerGroup


def walk_dict_leaves(tree: Any, prefix: str = "") -> Any:
    """Yield ``(dotted_path, leaf)`` pairs for a ``dict``-tree.

    Mirrors the path convention of
    :func:`opaque._clipping._per_group._extract_keys`: walks ``dict``-valued
    nodes recursively and treats anything that is not a ``dict`` as a
    leaf.  The yielded dotted-key paths match the keys
    :class:`PerGroup` expects when looking up per-leaf values, so this
    is the right walker to use whenever a per-leaf operation needs
    PerGroup parity (BC's φ-EMA, per-leaf noise allocation, …).
    """
    if isinstance(tree, dict):
        for k, v in tree.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            yield from walk_dict_leaves(v, sub)
    else:
        yield prefix, tree


def init_per_group_phi(params: Any) -> dict[str, float]:
    """Initial φ-EMA dict: ``0.0`` per dotted leaf path of ``params``.

    Used by per-group BC at ``init`` time so the state shape covers
    every leaf the per-step BC walk will see, including for nested
    param pytrees.
    """
    return {path: 0.0 for path, _ in walk_dict_leaves(params)}


def resolve_noise_variance(
    noise_stddev: float | PerGroup,
    key: str | None = None,
) -> float:
    """Square a (possibly per-group) noise stddev to get its variance.

    When ``noise_stddev`` is a :class:`PerGroup`, ``key`` selects the
    parameter's group; the per-key value is squared.  When it's a plain
    float, ``key`` is ignored.
    """
    if isinstance(noise_stddev, PerGroup):
        if key is None:
            raise ValueError(
                "resolve_noise_variance requires `key` for PerGroup noise_stddev"
            )
        return float(noise_stddev.for_key(key)) ** 2
    return float(noise_stddev) ** 2


def is_per_group(noise_stddev: float | PerGroup) -> bool:
    """Return ``True`` iff ``noise_stddev`` is a :class:`PerGroup`."""
    return isinstance(noise_stddev, PerGroup)


def update_phi_ema(
    phi: float | dict[str, float],
    new_variance: float | dict[str, float],
    b2: float,
) -> float | dict[str, float]:
    """Advance the noise-variance EMA by one step::

        φ_t = β₂ φ_{t-1} + (1 − β₂) Φ_t

    Both arguments are scalars in the homogeneous case; in the per-group
    case both are ``dict[str, float]`` keyed identically.  Mixed shapes
    raise.
    """
    if isinstance(phi, dict):
        if not isinstance(new_variance, dict):
            raise TypeError(
                "phi is per-group dict but new_variance is scalar; "
                "either both must be per-group or both must be scalar."
            )
        return {k: b2 * phi[k] + (1 - b2) * new_variance[k] for k in phi}
    return b2 * phi + (1 - b2) * float(new_variance)


__all__ = [
    "init_per_group_phi",
    "is_per_group",
    "resolve_noise_variance",
    "update_phi_ema",
    "walk_dict_leaves",
]
