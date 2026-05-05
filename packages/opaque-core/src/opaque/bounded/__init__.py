"""Bounded and noisy pytree containers for DP query values.

``BoundedPytree`` carries a pytree together with a public bound on one
private record's contribution to that pytree.  ``NoisyPytree`` is the
post-mechanism form: the pytree contains privatized values while ``bound``
still describes the original record-impact bound, not a bound on the noisy
output values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real
from typing import Any

import torch

from opaque.core.pytree import tree_map


Bound = Any
NoiseStddev = Any


def _validate_public_scalar(scalar: Any, *, op: str) -> float:
    if isinstance(scalar, bool) or not isinstance(scalar, Real):
        raise TypeError(
            f"{op} only supports public real-number scalars. "
            "Operate on `.pytree` and reconstruct the bounded value with an "
            "explicit bound when the bounded interpretation is unclear."
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


def _scale_bound(bound: Bound, factor: float) -> Bound:
    return abs(factor) * bound


def _scale_stddev(stddev: NoiseStddev, factor: float) -> NoiseStddev:
    if stddev is None:
        return None
    return abs(factor) * stddev


@dataclass(frozen=True)
class BoundedPytree:
    """A pytree with a public bound on one private record's contribution.

    Arithmetic is intentionally narrow.  Public scalar multiplication,
    division, and negation preserve the bounded-query interpretation.  Other
    operations should be applied to ``.pytree`` directly, followed by explicit
    reconstruction with the correct bound.
    """

    pytree: Any
    bound: Bound

    @property
    def sensitivity(self) -> float:
        """Scalar effective L2 sensitivity implied by ``bound``."""
        effective = getattr(self.bound, "effective", None)
        if effective is not None:
            return float(effective)
        return float(self.bound)

    def _scaled(self, scalar: float) -> BoundedPytree:
        return replace(
            self,
            pytree=_scale_tensor_leaves(self.pytree, scalar),
            bound=_scale_bound(self.bound, scalar),
        )

    def __mul__(self, scalar: Any) -> BoundedPytree:
        return self._scaled(_validate_public_scalar(scalar, op="BoundedPytree *"))

    def __rmul__(self, scalar: Any) -> BoundedPytree:
        return self._scaled(_validate_public_scalar(scalar, op="* BoundedPytree"))

    def __truediv__(self, scalar: Any) -> BoundedPytree:
        factor = _validate_public_scalar(scalar, op="BoundedPytree /")
        if factor == 0.0:
            raise ZeroDivisionError("BoundedPytree division by zero")
        return self._scaled(1.0 / factor)

    def __rtruediv__(self, scalar: Any) -> BoundedPytree:  # noqa: ARG002
        raise TypeError(_unsupported_message("reverse division"))

    def __neg__(self) -> BoundedPytree:
        return self._scaled(-1.0)

    def __add__(self, other: Any) -> BoundedPytree:  # noqa: ARG002
        raise TypeError(_unsupported_message("addition"))

    def __radd__(self, other: Any) -> BoundedPytree:  # noqa: ARG002
        raise TypeError(_unsupported_message("addition"))

    def __sub__(self, other: Any) -> BoundedPytree:  # noqa: ARG002
        raise TypeError(_unsupported_message("subtraction"))

    def __rsub__(self, other: Any) -> BoundedPytree:  # noqa: ARG002
        raise TypeError(_unsupported_message("subtraction"))

    def __pow__(self, exponent: Any) -> BoundedPytree:  # noqa: ARG002
        raise TypeError(_unsupported_message("power"))

    def clone(self) -> BoundedPytree:
        """Clone tensor leaves while preserving metadata."""
        return replace(self, pytree=_apply_tensor_method(self.pytree, "clone"))

    def detach(self) -> BoundedPytree:
        """Detach tensor leaves while preserving metadata."""
        return replace(self, pytree=_apply_tensor_method(self.pytree, "detach"))

    def to(self, *args: Any, **kwargs: Any) -> BoundedPytree:
        """Call ``Tensor.to`` on tensor leaves while preserving metadata."""
        return replace(
            self,
            pytree=_apply_tensor_method(self.pytree, "to", *args, **kwargs),
        )


@dataclass(frozen=True)
class NoisyPytree(BoundedPytree):
    """A privatized pytree carrying original bound and noise metadata."""

    noise_stddev: NoiseStddev = None

    def _scaled(self, scalar: float) -> NoisyPytree:
        return replace(
            self,
            pytree=_scale_tensor_leaves(self.pytree, scalar),
            bound=_scale_bound(self.bound, scalar),
            noise_stddev=_scale_stddev(self.noise_stddev, scalar),
        )


def bounded(pytree: Any, *, bound: Bound) -> BoundedPytree:
    """Manually wrap a pytree with public DP contribution-bound metadata."""
    return BoundedPytree(pytree=pytree, bound=bound)


def noisy(pytree: Any, *, bound: Bound, noise_stddev: NoiseStddev) -> NoisyPytree:
    """Manually wrap an already privatized pytree with noise metadata."""
    return NoisyPytree(pytree=pytree, bound=bound, noise_stddev=noise_stddev)


def _unsupported_message(op: str) -> str:
    return (
        f"BoundedPytree {op} does not preserve DP bound semantics "
        "automatically. Operate on `.pytree` and reconstruct the bounded value "
        "with an explicit bound."
    )


__all__ = ["BoundedPytree", "NoisyPytree", "bounded", "noisy"]
