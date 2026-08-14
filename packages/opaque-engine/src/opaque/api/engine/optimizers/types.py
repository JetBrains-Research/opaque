"""State containers for backend-neutral optimizers."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opaque.api.engine.pytree import ParamPath
    from opaque.types import TensorPytree


@dataclasses.dataclass(frozen=True)
class AdamState:
    """Immutable state for Adam-family moment scaling.

    Carries the noise-variance EMA ``phi`` regardless of whether DP
    bias correction is actively in use — this keeps the state shape
    constant across calls so checkpoints don't depend on call history.
    With ``noise_bias_correction=True``, ``phi`` is a path-keyed dict
    from init (stable for ``state_dict`` / ``from_state_dict``).

    Attributes:
        mu: First-moment EMA (pytree matching params).
        nu: Second-moment EMA (pytree matching params).
        phi: Noise-variance EMA (scalar, or ``dict[ParamPath, float]``
            when BC is enabled).  Stays at zero unless ``NoisedPytree``
            updates supply realized σ metadata.
        step: Number of completed updates.
    """

    mu: TensorPytree
    nu: TensorPytree
    phi: float | dict[ParamPath, float]
    step: int


@dataclasses.dataclass(frozen=True)
class SGDState:
    """Immutable state for SGD momentum.

    Attributes:
        momentum: Momentum buffer pytree matching params, or ``None``
            when momentum is disabled.
        step: Number of completed updates.
    """

    momentum: TensorPytree | None
    step: int


@dataclasses.dataclass(frozen=True)
class LionState:
    """State for Lion's first-moment buffer."""

    m: TensorPytree
    step: int


@dataclasses.dataclass(frozen=True)
class RAdamState:
    """State for RAdam's moments and optional noise correction."""

    mu: TensorPytree
    nu: TensorPytree
    phi: float | dict[ParamPath, float]
    step: int


@dataclasses.dataclass(frozen=True)
class RMSpropState:
    """State for RMSprop's second moment and noise correction."""

    nu: TensorPytree
    phi: float | dict[ParamPath, float]
    step: int


@dataclasses.dataclass(frozen=True)
class AdagradState:
    """State for Adagrad's cumulative moment and variance."""

    v_acc: TensorPytree
    phi_acc: float | dict[ParamPath, float]
    step: int


@dataclasses.dataclass(frozen=True)
class AdadeltaState:
    """State for Adadelta's gradient and update moment EMAs."""

    v_g: TensorPytree
    v_dx: TensorPytree
    phi_g: float | dict[ParamPath, float] | None
    phi_dx: TensorPytree | None
    step: int


@dataclasses.dataclass(frozen=True)
class AdafactorState:
    """State for Adafactor's factored second moments."""

    m: TensorPytree | None
    v_flat: tuple[tuple[object, ...], ...]
    phi_flat: tuple[float, ...]
    treespec: object
    paths: tuple[ParamPath, ...]
    step: int


@dataclasses.dataclass(frozen=True)
class AdEMAMixState:
    """State for AdEMAMix's fast, slow, and second moments."""

    m_fast: TensorPytree
    m_slow: TensorPytree
    nu: TensorPytree
    phi: float | dict[ParamPath, float]
    step: int


@dataclasses.dataclass(frozen=True)
class ScheduleFreeState:
    """State for Schedule-Free raw and published parameter sequences."""

    z: TensorPytree
    x: TensorPytree
    inner: object
    step: int
    beta: float


__all__ = [
    "AdamState",
    "AdEMAMixState",
    "AdadeltaState",
    "AdafactorState",
    "AdagradState",
    "LionState",
    "RAdamState",
    "RMSpropState",
    "SGDState",
    "ScheduleFreeState",
]
