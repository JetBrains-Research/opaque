"""Immutable host-side execution plans for matrix-factorization noise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

MfExecutionMode = Literal["identity", "toeplitz", "blt", "lambda_replay"]


def _float_tuple(values: object) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(
            f"execution-plan vectors must be one-dimensional, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("execution-plan vectors must contain only finite values")
    return tuple(float(value) for value in array)


def _band_count(values: tuple[float, ...]) -> int:
    """Return the length of ``values`` without its exactly-zero tail."""
    for index in range(len(values) - 1, -1, -1):
        if values[index] != 0.0:
            return index + 1
    return 0


@dataclass(frozen=True, slots=True)
class MfExecutionPlan:
    """Provider-independent description of an MF mechanism at one horizon.

    ``strategy_bands`` / ``inverse_bands`` record how many leading
    coefficients of the strategy (respectively its Toeplitz inverse) are
    structurally non-zero. The executor streams whichever window is
    banded: a banded strategy drives the O(bands)-state recurrence
    ``x_t = (z_t - sum_j c_j x_{t-j}) / c_0``; a banded inverse (BISR)
    drives the direct convolution over the last ``inverse_bands`` noise
    draws. Dense-dense plans degrade to O(n_steps) state, never worse.
    """

    mode: MfExecutionMode
    n_steps: int
    strategy_coefficients: tuple[float, ...]
    inverse_coefficients: tuple[float, ...]
    row_l2: tuple[float, ...]
    column_scales: tuple[float, ...]
    buffer_decay: tuple[float, ...] = ()
    output_scale: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        for name in (
            "strategy_coefficients",
            "inverse_coefficients",
            "row_l2",
            "column_scales",
            "buffer_decay",
            "output_scale",
        ):
            object.__setattr__(self, name, _float_tuple(getattr(self, name)))
        if len(self.strategy_coefficients) != self.n_steps:
            raise ValueError("strategy_coefficients must have length n_steps")
        if len(self.row_l2) != self.n_steps:
            raise ValueError("row_l2 must have length n_steps")
        if len(self.column_scales) != self.n_steps:
            raise ValueError("column_scales must have length n_steps")
        if not self.inverse_coefficients:
            raise ValueError("inverse_coefficients must not be empty")
        if len(self.buffer_decay) != len(self.output_scale):
            raise ValueError(
                "buffer_decay and output_scale must have equal length, got "
                f"{len(self.buffer_decay)} and {len(self.output_scale)}"
            )

    @property
    def strategy_bands(self) -> int:
        """Number of structurally non-zero leading strategy coefficients."""
        return max(1, _band_count(self.strategy_coefficients))

    @property
    def inverse_bands(self) -> int:
        """Number of structurally non-zero leading inverse coefficients."""
        return max(1, _band_count(self.inverse_coefficients))

    def coefficients(self) -> np.ndarray:
        """Return a mutable host copy of the strategy coefficients."""
        return np.asarray(self.strategy_coefficients, dtype=np.float64)

    def row_l2_norms(self) -> np.ndarray:
        """Return the effective inverse-matrix row norms."""
        return np.asarray(self.row_l2, dtype=np.float64)


def _inverse_toeplitz_coefficients(coefficients: np.ndarray) -> np.ndarray:
    inverse = np.zeros_like(coefficients)
    inverse[0] = 1.0 / coefficients[0]
    for step in range(1, len(coefficients)):
        inverse[step] = (
            -np.dot(coefficients[1 : step + 1], inverse[step - 1 :: -1])
            / coefficients[0]
        )
    return inverse


def toeplitz_execution_plan(
    coefficients: object,
    *,
    mode: Literal["toeplitz", "blt"] = "toeplitz",
    column_normalized: bool = False,
    buffer_decay: object = (),
    output_scale: object = (),
    inverse_coefficients: object | None = None,
) -> MfExecutionPlan:
    """Build a finite-horizon plan from the first column of a Toeplitz strategy.

    ``inverse_coefficients`` optionally supplies the known banded Toeplitz
    coefficients of the un-normalized ``C^{-1}`` (e.g. BISR's banded
    inverse). It is validated against ``coefficients`` — the Toeplitz
    product of the two must be the identity — and replaces the O(n^2)
    inversion recurrence, keeping the inverse *exactly* banded so the
    executor can stream over an O(bands) window of past noise draws.
    """
    strategy = np.asarray(coefficients, dtype=np.float64)
    if strategy.ndim != 1 or len(strategy) < 1:
        raise ValueError("coefficients must be a non-empty one-dimensional array")
    if not np.isfinite(strategy[0]) or strategy[0] == 0:
        raise ValueError("the leading strategy coefficient must be finite and non-zero")
    if inverse_coefficients is not None:
        hint = np.asarray(inverse_coefficients, dtype=np.float64)
        if hint.ndim != 1 or len(hint) < 1:
            raise ValueError(
                "inverse_coefficients must be a non-empty one-dimensional array"
            )
        # A finite-horizon plan is the n x n truncation, so only the first
        # n_steps == len(strategy) product terms must equal the identity:
        # lower-triangular Toeplitz inverses truncate consistently, and a
        # hint longer than the horizon (bandwidth > n_steps) is valid with
        # its out-of-horizon taps ignored.
        hint = hint[: len(strategy)]
        window = len(strategy)
        product = np.convolve(strategy, hint)[:window]
        identity = np.zeros_like(product)
        identity[0] = 1.0
        if not np.allclose(product, identity, atol=1e-8):
            raise ValueError(
                "inverse_coefficients is not the Toeplitz inverse of coefficients: "
                "max |toeplitz(coefficients) @ toeplitz(inverse_coefficients) - I| "
                f"= {np.abs(product - identity).max():.3e}"
            )
        inverse = np.zeros(len(strategy), dtype=np.float64)
        inverse[: len(hint)] = hint
    else:
        inverse = _inverse_toeplitz_coefficients(strategy)
    if column_normalized:
        cumulative = np.cumsum(np.square(strategy, dtype=np.float64))
        scales = np.sqrt(cumulative)[::-1]
    else:
        scales = np.ones(len(strategy), dtype=np.float64)
    row_l2 = scales * np.sqrt(np.cumsum(np.square(inverse, dtype=np.float64)))
    return MfExecutionPlan(
        mode=mode,
        n_steps=len(strategy),
        strategy_coefficients=_float_tuple(strategy),
        inverse_coefficients=_float_tuple(inverse),
        row_l2=_float_tuple(row_l2),
        column_scales=_float_tuple(scales),
        buffer_decay=_float_tuple(buffer_decay),
        output_scale=_float_tuple(output_scale),
    )


def identity_execution_plan(n_steps: int) -> MfExecutionPlan:
    """Build an identity-plan record."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    coefficients = np.zeros(n_steps, dtype=np.float64)
    coefficients[0] = 1.0
    plan = toeplitz_execution_plan(coefficients)
    return MfExecutionPlan(
        mode="identity",
        n_steps=plan.n_steps,
        strategy_coefficients=plan.strategy_coefficients,
        inverse_coefficients=plan.inverse_coefficients,
        row_l2=plan.row_l2,
        column_scales=plan.column_scales,
    )


def lambda_replay_execution_plan(
    lambda_: float, n_steps: int, *, normalized: bool
) -> MfExecutionPlan:
    """Build the bidiagonal inverse plan used by lambda-CGD replay."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    strategy = np.power(lambda_, np.arange(n_steps, dtype=np.float64))
    base = toeplitz_execution_plan(strategy, column_normalized=normalized)
    inverse = np.zeros(n_steps, dtype=np.float64)
    inverse[0] = 1.0
    if n_steps > 1:
        inverse[1] = -lambda_
    row_l2 = np.asarray(base.column_scales) * np.sqrt(
        np.cumsum(np.square(inverse, dtype=np.float64))
    )
    return MfExecutionPlan(
        mode="lambda_replay",
        n_steps=n_steps,
        strategy_coefficients=base.strategy_coefficients,
        inverse_coefficients=_float_tuple(inverse),
        row_l2=_float_tuple(row_l2),
        column_scales=base.column_scales,
    )


__all__ = [
    "MfExecutionMode",
    "MfExecutionPlan",
    "identity_execution_plan",
    "lambda_replay_execution_plan",
    "toeplitz_execution_plan",
]
