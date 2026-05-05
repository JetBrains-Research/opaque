"""Type definitions for clipping: state markers and the clipped pytree wrapper.

``ClippedPytree`` carries a pytree together with a public maximum L2 norm
on one private record's contribution to that pytree.  The wrapper is
mechanism-agnostic: anything that establishes a per-record L2 cap on the
contained value can construct one, but the typical producer is
per-example clipping (``clipped_grad`` and ``clipped_fun``).

The post-mechanism counterpart ``NoisedPytree`` (extends ``ClippedPytree``
with a ``noise_stddev`` field) lives in :mod:`opaque.core.noise` —
colocated with the noise math helpers.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any

import torch

from opaque.core.pytree import tree_map


# ---------------------------------------------------------------------------
# Clipping state markers
# ---------------------------------------------------------------------------


class ClipState(ABC):
    """Base class for clipping state.

    Clipping state is the explicit state token returned by clipping transforms.
    Fixed clipping uses it as an immutable marker; adaptive schemes may carry
    the threshold and counters needed for the next step. Privacy calibration
    metadata lives on the returned :class:`ClippedPytree`, not on the state
    object.

    Example:
        >>> from opaque.clipping import clipped_grad
        >>> loss_fn = lambda params, x, y: ((x @ params - y) ** 2).mean()
        >>>
        >>> # Fixed clipping returns (callable, ClipState)
        >>> grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=1.0, batch_argnums=(1, 2))
        >>> grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        >>>
        >>> # Noise calibration reads ``max_norm`` from the clipped output
        >>> from opaque.dpsgd.noise import gaussian_noise
        >>> noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(0))
        >>> noisy_grads, noise_state = noise_fn(grads, noise_state)
    """


@dataclass(frozen=True)
class FixedClipState(ClipState):
    """Marker state for fixed (non-adaptive) clipping."""


# ---------------------------------------------------------------------------
# ClippedPytree
# ---------------------------------------------------------------------------


MaxNorm = Any


def _validate_public_scalar(scalar: Any, *, op: str) -> float:
    if isinstance(scalar, bool) or not isinstance(scalar, Real):
        raise TypeError(
            f"{op} only supports public real-number scalars. "
            "Operate on `.pytree` and reconstruct the clipped value with an "
            "explicit max_norm when the clipped interpretation is unclear."
        )
    return float(scalar)


def _scale_tensor_leaves(pytree: Any, scalar: float) -> Any:
    return tree_map(
        lambda leaf: leaf * scalar if isinstance(leaf, torch.Tensor) else leaf,
        pytree,
    )


def _apply_tensor_method(pytree: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    def _apply(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            return getattr(leaf, method)(*args, **kwargs)
        return leaf

    return tree_map(_apply, pytree)


def _scale_max_norm(max_norm: MaxNorm, factor: float) -> MaxNorm:
    return abs(factor) * max_norm


def _unsupported_message(op: str) -> str:
    return (
        f"ClippedPytree {op} does not preserve DP max_norm semantics "
        "automatically. Operate on `.pytree` and reconstruct the clipped "
        "value with an explicit max_norm."
    )


@dataclass(frozen=True)
class ClippedPytree:
    """A pytree with a public maximum L2 norm on one record's contribution.

    Arithmetic is intentionally narrow.  Public scalar multiplication,
    division, and negation preserve the clipped-query interpretation.
    Other operations should be applied to ``.pytree`` directly, followed
    by explicit reconstruction with the correct ``max_norm``.
    """

    pytree: Any
    max_norm: MaxNorm

    @property
    def sensitivity(self) -> float:
        """Scalar effective L2 sensitivity implied by ``max_norm``."""
        effective = getattr(self.max_norm, "effective", None)
        if effective is not None:
            return float(effective)
        return float(self.max_norm)

    def _scaled(self, scalar: float) -> ClippedPytree:
        return replace(
            self,
            pytree=_scale_tensor_leaves(self.pytree, scalar),
            max_norm=_scale_max_norm(self.max_norm, scalar),
        )

    def __mul__(self, scalar: Any) -> ClippedPytree:
        return self._scaled(_validate_public_scalar(scalar, op="ClippedPytree *"))

    def __rmul__(self, scalar: Any) -> ClippedPytree:
        return self._scaled(_validate_public_scalar(scalar, op="* ClippedPytree"))

    def __truediv__(self, scalar: Any) -> ClippedPytree:
        factor = _validate_public_scalar(scalar, op="ClippedPytree /")
        if factor == 0.0:
            raise ZeroDivisionError("ClippedPytree division by zero")
        return self._scaled(1.0 / factor)

    def __rtruediv__(self, scalar: Any) -> ClippedPytree:  # noqa: ARG002
        raise TypeError(_unsupported_message("reverse division"))

    def __neg__(self) -> ClippedPytree:
        return self._scaled(-1.0)

    def __add__(self, other: Any) -> ClippedPytree:  # noqa: ARG002
        raise TypeError(_unsupported_message("addition"))

    def __radd__(self, other: Any) -> ClippedPytree:  # noqa: ARG002
        raise TypeError(_unsupported_message("addition"))

    def __sub__(self, other: Any) -> ClippedPytree:  # noqa: ARG002
        raise TypeError(_unsupported_message("subtraction"))

    def __rsub__(self, other: Any) -> ClippedPytree:  # noqa: ARG002
        raise TypeError(_unsupported_message("subtraction"))

    def __pow__(self, exponent: Any) -> ClippedPytree:  # noqa: ARG002
        raise TypeError(_unsupported_message("power"))

    def clone(self) -> ClippedPytree:
        """Clone tensor leaves while preserving metadata."""
        return replace(self, pytree=_apply_tensor_method(self.pytree, "clone"))

    def detach(self) -> ClippedPytree:
        """Detach tensor leaves while preserving metadata."""
        return replace(self, pytree=_apply_tensor_method(self.pytree, "detach"))

    def to(self, *args: Any, **kwargs: Any) -> ClippedPytree:
        """Call ``Tensor.to`` on tensor leaves while preserving metadata."""
        return replace(
            self,
            pytree=_apply_tensor_method(self.pytree, "to", *args, **kwargs),
        )


def clipped(pytree: Any, *, max_norm: MaxNorm) -> ClippedPytree:
    """Manually wrap a pytree with public DP max-norm metadata."""
    return ClippedPytree(pytree=pytree, max_norm=max_norm)


__all__ = [
    "ClipState",
    "ClippedPytree",
    "FixedClipState",
    "MaxNorm",
    "clipped",
]
