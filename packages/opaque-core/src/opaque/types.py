"""Cross-package DP-flow types.

Single canonical home for the data types that connect clipping →
noise → optimizer:

- **Pytree wrappers**: ``ClippedPytree`` (post-clipping), ``NoisedPytree``
  (post-noise), and the paired-stream outputs
  ``SecondMomentClippingOutput`` / ``SecondMomentNoiseOutput``.
- **Per-group container**: ``PerGroup`` — a dict-like that flows through
  the entire pipeline carrying per-parameter-group scalar values.
- **Abstract state bases**: ``ClipState`` and ``NoiseState`` — markers
  shared across DP-SGD and DP-FTRL implementations.
- **Aliases**: ``MaxNorm``, ``NoiseStddev`` — opaque-typed unions used
  in wrapper metadata fields.
- **Factories**: ``clipped()`` and ``noised()`` — manual wrapper
  constructors for callers that already produced privatised values.

Concrete state classes live with the factories that produce them:
``FixedClipState`` in :mod:`opaque.clipping.types`,
``AdaptiveClipState`` in :mod:`opaque.dpsgd.clipping.adaptive`,
``GaussianNoiseState`` in :mod:`opaque.dpsgd.noise.gaussian`,
``MFNoiseState`` in :mod:`opaque.dpftrl.noise._engine`.
"""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, NamedTuple, Union

import torch

from opaque.core.pytree import tree_map
from opaque.random import RngKey


# ===========================================================================
# Pytree alias
# ===========================================================================


# A pytree whose leaves are ``torch.Tensor``.  Restricted to the
# concrete container types the rest of the library actually rebuilds
# and serialises (``dict``, ``list``, ``tuple``); custom ``Mapping`` /
# ``Sequence`` subclasses are intentionally excluded.
#
# Real ``Union`` (not a string) so ``typing.get_type_hints()`` and
# ``typing.get_args()`` see the alias correctly.  Recursion is encoded
# with forward references; static checkers expand the alias to ``Any``,
# which is fine — the alias is documentation-grade typing.
TensorPytree = Union[
    torch.Tensor,
    dict[str, "TensorPytree"],
    list["TensorPytree"],
    tuple["TensorPytree", ...],
]


# ===========================================================================
# Per-group container
# ===========================================================================


@dataclass(frozen=True)
class PerGroup:
    """Per-group values with pre-resolved parameter-to-group assignment.

    Flows through the DP pipeline as clipping norms, output bounds,
    noise multipliers, and noise stddevs.  Supports arithmetic so
    training loop code reads naturally::

        noise_multiplier * clipping_norm  # PerGroup when clipping_norm is PerGroup

    For MSE-optimal per-group noise allocation, use
    :func:`opaque.dpsgd.noise.per_group_noise_stddev` instead of
    a plain product.

    Attributes:
        groups: Mapping from parameter key to group name (pre-resolved).
        values: Mapping from group name to the per-group value.
    """

    groups: dict[str, str]
    values: dict[str, float]

    @property
    def effective(self) -> float:
        """``sqrt(sum v²)`` — effective global value for accounting."""
        return math.sqrt(sum(v**2 for v in self.values.values()))

    def __rmul__(self, scalar: float) -> PerGroup:
        return PerGroup(self.groups, {k: scalar * v for k, v in self.values.items()})

    def __mul__(self, scalar: float) -> PerGroup:
        return PerGroup(self.groups, {k: v * scalar for k, v in self.values.items()})

    def __truediv__(self, scalar: float) -> PerGroup:
        return PerGroup(self.groups, {k: v / scalar for k, v in self.values.items()})

    def for_key(self, key: str) -> float:
        """Look up the per-group value for a parameter key."""
        return self.values[self.groups[key]]

    def state_dict(self) -> dict[str, dict[str, float] | dict[str, str]]:
        """Serialize to a JSON-compatible dict for checkpointing."""
        return {
            "groups": dict(self.groups),
            "values": {k: float(v) for k, v in self.values.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> PerGroup:
        """Reconstruct a :class:`PerGroup` from :meth:`state_dict` output."""
        return cls(
            groups=dict(state["groups"]),
            values={k: float(v) for k, v in state["values"].items()},
        )


# ===========================================================================
# Abstract state bases (cross-family markers)
# ===========================================================================


class ClipState(ABC):
    """Base class for clipping state.

    The explicit state token returned by clipping transforms.  Fixed
    clipping uses an empty marker; adaptive schemes carry the
    threshold and counters needed for the next step.  Privacy
    calibration metadata lives on the returned :class:`ClippedPytree`,
    not on the state object.
    """


class NoiseState(ABC):
    """Base class for noise state.

    All noise functions (Gaussian, truncated Gaussian, MF) return a
    state object that inherits from this class, providing a unified
    interface for step tracking and RNG key management.

    Attributes:
        _step_counter: Number of ``noise_fn`` calls made.
        _rng_key: Immutable RNG key for deterministic per-step derivation.
    """

    _step_counter: int
    """Number of ``noise_fn`` calls made."""

    _rng_key: RngKey
    """Immutable RNG key for deterministic per-step derivation."""


# ===========================================================================
# DP wrapper types
# ===========================================================================


MaxNorm = Any
"""Type of the ``ClippedPytree.max_norm`` field — ``float`` or ``PerGroup``."""

NoiseStddev = Any
"""Type of the ``NoisedPytree.noise_stddev`` field — ``float | PerGroup | None``."""


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


def _scale_stddev(stddev: NoiseStddev, factor: float) -> NoiseStddev:
    if stddev is None:
        return None
    return abs(factor) * stddev


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


@dataclass(frozen=True)
class NoisedPytree(ClippedPytree):
    """A privatised pytree carrying max-norm and realised noise metadata.

    Extends :class:`ClippedPytree` with the per-step ``noise_stddev``
    recorded by the noise mechanism.  ``max_norm`` still describes the
    original record-impact bound, not a bound on the noised output values.
    """

    noise_stddev: NoiseStddev = None

    def _scaled(self, scalar: float) -> NoisedPytree:
        return replace(
            self,
            pytree=_scale_tensor_leaves(self.pytree, scalar),
            max_norm=_scale_max_norm(self.max_norm, scalar),
            noise_stddev=_scale_stddev(self.noise_stddev, scalar),
        )


# ===========================================================================
# Paired-stream outputs (private second moment)
# ===========================================================================


class SecondMomentClippingOutput(NamedTuple):
    """Pre-noise paired-stream input to a noise mechanism.

    Symmetric with :class:`SecondMomentNoiseOutput` but on the
    *pre-noise* side: where the output pairs two ``NoisedPytree``s
    (post-noise), this pairs two ``ClippedPytree``s (pre-noise).
    Each carries its own ``max_norm``.

    Constructed by clipping when the user requests paired-stream
    output (per-example squaring inside the vmap loop).  The presence
    of this type at a noise mechanism's input switches the mechanism
    into paired-stream mode without an explicit constructor flag.

    Attributes:
        grads: Clipped per-example summed gradients (``Σᵢ gᵢ``).
        squared_grads: Clipped per-example summed squared gradients
            (``Σᵢ gᵢ²``).  The squaring happens per-example inside the
            clipping loop so the second-stream sensitivity is ``C²``
            (per record) and the streams are jointly DP-accountable.
    """

    grads: ClippedPytree
    squared_grads: ClippedPytree


class SecondMomentNoiseOutput(NamedTuple):
    """Noise output with private first and second moment streams.

    Attributes:
        noisy_grads: Noised clipped gradients for the optimizer's first
            moment / update direction.
        noisy_squared_grads: Noised element-wise squared clipped gradients
            for optimizers with a second moment accumulator.
    """

    noisy_grads: NoisedPytree
    noisy_squared_grads: NoisedPytree


# ===========================================================================
# Manual wrapper factories
# ===========================================================================


def clipped(pytree: Any, *, max_norm: MaxNorm) -> ClippedPytree:
    """Manually wrap a pytree with public DP max-norm metadata."""
    return ClippedPytree(pytree=pytree, max_norm=max_norm)


def noised(
    pytree: Any,
    *,
    max_norm: MaxNorm,
    noise_stddev: NoiseStddev,
) -> NoisedPytree:
    """Manually wrap an already-privatised pytree with noise metadata."""
    return NoisedPytree(pytree=pytree, max_norm=max_norm, noise_stddev=noise_stddev)


__all__ = [
    "ClipState",
    "ClippedPytree",
    "MaxNorm",
    "NoiseState",
    "NoiseStddev",
    "NoisedPytree",
    "PerGroup",
    "SecondMomentClippingOutput",
    "SecondMomentNoiseOutput",
    "TensorPytree",
    "clipped",
    "noised",
]
