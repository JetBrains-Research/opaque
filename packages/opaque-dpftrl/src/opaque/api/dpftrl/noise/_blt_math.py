"""Buffered Linear Toeplitz (BLT) strategy matrices.

BLT matrices provide a memory-efficient parameterization of Toeplitz
matrices via a small number of buffers. This enables:
- O(num_buffers) memory for noise generation (vs O(n) for general Toeplitz)
- Closed-form sensitivity and error computation
- Efficient optimization via L-BFGS

References:
    - BLT: https://arxiv.org/abs/2404.16706
    - Multi-epoch BLT: https://arxiv.org/abs/2408.08868
    - Inversion theorem: https://arxiv.org/abs/2504.21413
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from opaque.api.engine import ops

from . import (
    _sensitivity as sensitivity,
)
from . import (
    _streaming_matrix as streaming_matrix,
)
from . import (
    _toeplitz as toeplitz,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "BufferedToeplitz",
    "inverse",
    "inverse_as_streaming_matrix",
    "iteration_error",
    "materialize",
    "max_error",
    "optimize",
    "sensitivity_squared",
]


# ── Internal builder ────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class _StreamingBufferState:
    zero: Any
    buffers: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class _StreamingMatrixBuilder:
    """Builder to convert a BLT to a StreamingMatrix.

    Attributes:
        buf_decay: Decay factors for each buffer, shape (num_buffers,).
        output_scale: Scale factors for output, shape (num_buffers,).
    """

    buf_decay: np.ndarray
    output_scale: np.ndarray

    @property
    def dtype(self) -> np.dtype:
        assert self.output_scale.dtype == self.buf_decay.dtype
        return self.output_scale.dtype

    def _init(self, abstract_value):
        num_buffers = self.buf_decay.shape[0]
        if isinstance(abstract_value, np.ndarray):
            dtype = (
                np.dtype(np.float32)
                if abstract_value.dtype == np.dtype(np.float16)
                else abstract_value.dtype
            )
        else:
            value_dtype = ops.dtype(abstract_value)
            dtype = ops.float32() if ops.is_low_precision(value_dtype) else value_dtype
        zero = streaming_matrix._zeros_like(abstract_value, dtype=dtype)
        return _StreamingBufferState(zero, (zero,) * num_buffers)

    def _read(self, state: _StreamingBufferState):
        result = state.zero
        for scale, buffer in zip(self.output_scale, state.buffers, strict=True):
            result = streaming_matrix._add(
                result,
                streaming_matrix._multiply(
                    streaming_matrix._scalar_like(scale, buffer), buffer
                ),
            )
        return result

    def _update(self, state: _StreamingBufferState, next_rhs_value):
        if not state.buffers:
            return state
        buffers = tuple(
            streaming_matrix._add(
                streaming_matrix._multiply(
                    streaming_matrix._scalar_like(decay, buffer), buffer
                ),
                next_rhs_value,
            )
            for decay, buffer in zip(self.buf_decay, state.buffers, strict=True)
        )
        return _StreamingBufferState(state.zero, buffers)

    def build(self) -> streaming_matrix.StreamingMatrix:
        """Returns a StreamingMatrix representing C.

        Implements Algorithm 2 of https://arxiv.org/abs/2408.08868
        """

        def multiply_next(xi, state):
            yi = xi + self._read(state)
            state = self._update(state, xi)
            return yi, state

        return streaming_matrix.StreamingMatrix.from_array_implementation(
            self._init, multiply_next
        )

    def build_inverse(self) -> streaming_matrix.StreamingMatrix:
        """Returns a StreamingMatrix representing C^{-1}.

        Implements Algorithm 3 of https://arxiv.org/abs/2408.08868
        """

        def inv_multiply_next(yi, state):
            xi = yi - self._read(state)
            state = self._update(state, xi)
            return xi, state

        return streaming_matrix.StreamingMatrix.from_array_implementation(
            self._init, inv_multiply_next
        )


# ── Data container ──────────────────────────────────────────────────────


@dataclasses.dataclass
class BufferedToeplitz:
    """A lower-triangular Toeplitz C parameterized as a BLT.

    If buf_decay = [d1, d2] and output_scale = [s1, s2], for n = 5::

        C = [[1    0   0   0   0 ]     t0 = 1
             [t1   1   0   0   0 ]     t1 = s1       + s2
             [t2   t1  1   0   0 ]     t2 = s1*d1    + s2*d2
             [t3   t2  t1  1   0 ]     t3 = s1*d1**2 + s2*d2**2
             [t4   t3  t2  t1  1 ]]    t4 = s1*d1**3 + s2*d2**3

    Attributes:
        buf_decay: Shape (nbuf,), diagonal elements.
        output_scale: Shape (nbuf,), output scaling factors.
    """

    buf_decay: NDArray[np.float64]
    output_scale: NDArray[np.float64]

    def validate(self) -> None:
        """Validate basic properties of the BLT parameters."""
        if not (self.buf_decay.ndim <= 1 and self.output_scale.ndim <= 1):
            raise ValueError(
                f"buf_decay and output_scale must be 0D or 1D, but: "
                f"buf_decay.shape={self.buf_decay.shape}, "
                f"output_scale.shape={self.output_scale.shape}"
            )
        if self.buf_decay.shape != self.output_scale.shape:
            raise ValueError(
                f"buf_decay and output_scale must have same shape: "
                f"{self.buf_decay.shape} != {self.output_scale.shape}"
            )

    @classmethod
    def build(
        cls,
        buf_decay,
        output_scale,
        dtype=np.float64,
    ) -> BufferedToeplitz:
        """Construct and canonicalize a BLT.

        Args:
            buf_decay: Decay parameters.
            output_scale: Scale parameters.
            dtype: Data type (float64 recommended for numerical stability).

        Returns:
            A canonicalized BufferedToeplitz.
        """
        del dtype
        blt = cls(
            buf_decay=np.asarray(buf_decay, dtype=np.float64),
            output_scale=np.asarray(output_scale, dtype=np.float64),
        )
        return canonicalize(blt)

    @classmethod
    def from_rational_approx_to_sqrt_x(
        cls,
        num_buffers: int,
        *,
        max_buf_decay: float = 1.0,
        max_pillutla_score: float | None = None,
        buf_decay_scale: float = 1.6,
        buf_decay_shift: int = -1,
    ) -> BufferedToeplitz:
        """Returns a BLT from a rational approximation of 1/sqrt(1-x).

        From Proposition 4.5 of https://arxiv.org/abs/2404.16706v2.

        Args:
            num_buffers: Number of buffers (degree of rational function).
            max_buf_decay: Maximum value for buf_decay.
            max_pillutla_score: If set, enforce this bound.
            buf_decay_scale: Factor scaling the dynamic range.
            buf_decay_shift: Shift for the counter k.

        Returns:
            A BufferedToeplitz initialization.
        """
        if num_buffers < 1:
            raise ValueError("num_buffers must be >= 1.")

        degree = num_buffers
        d1 = (degree + 1) // 2
        h = buf_decay_scale * np.pi / np.sqrt(2 * (d1 + 1))

        ks = np.arange(-d1 + 1, degree - d1 + 1, 1) + buf_decay_shift
        buf_decay = 1 / (1 + np.exp(2 * h * ks))
        output_scale = -(buf_decay**2) * np.exp(3 * h * ks)
        constant_term = np.sum(np.exp(h * ks) * buf_decay)
        output_scale /= constant_term

        # Build C^{-1}, then invert to get C
        inv_blt = cls.build(buf_decay=buf_decay, output_scale=output_scale)
        blt = inverse(inv_blt)
        buf_decay_t = blt.buf_decay.copy()
        output_scale_t = blt.output_scale.copy()

        # Scale buf_decay if needed
        largest = buf_decay_t[0]
        scale = min(1.0, max_buf_decay / float(largest))
        buf_decay_t = buf_decay_t * scale

        if max_pillutla_score is not None:
            score = float(np.sum(output_scale_t / buf_decay_t))
            score_scale = min(1.0, max_pillutla_score / score)
            output_scale_t = output_scale_t * score_scale

        return cls.build(buf_decay=buf_decay_t, output_scale=output_scale_t)

    @property
    def dtype(self):
        return self.buf_decay.dtype

    @property
    def _num_buffers(self) -> int:
        return self.buf_decay.shape[0]

    def pillutla_score(self) -> float:
        """Returns the Pillutla Score of the BLT.

        From Theorem 1 of https://arxiv.org/abs/2504.21413.
        Score < 1 ensures well-behaved inverse.

        Returns:
            sum(output_scale / buf_decay).
        """
        return float(np.sum(self.output_scale / self.buf_decay))

    def __repr__(self) -> str:
        return (
            f"BufferedToeplitz(buf_decay={self.buf_decay.tolist()}, "
            f"output_scale={self.output_scale.tolist()})"
        )


# ── Standalone functions (extracted from BufferedToeplitz methods) ───────


def canonicalize(blt: BufferedToeplitz) -> BufferedToeplitz:
    """Return a BLT with buf_decay in decreasing order.

    Args:
        blt: The BLT to canonicalize.

    Returns:
        A new BufferedToeplitz with sorted buf_decay.
    """
    blt.validate()
    idx = np.argsort(blt.buf_decay)[::-1]
    return BufferedToeplitz(
        buf_decay=blt.buf_decay[idx],
        output_scale=blt.output_scale[idx],
    )


def toeplitz_coefs(blt: BufferedToeplitz, n: int) -> NDArray[np.float64]:
    """Returns the Toeplitz coefficients for C.

    Args:
        blt: The BLT.
        n: Number of coefficients.

    Returns:
        Tensor of n Toeplitz coefficients.
    """
    if blt._num_buffers == 0:
        result = np.zeros(n, dtype=np.float64)
        result[0] = 1.0
        return result
    powers = np.arange(n - 1, dtype=np.float64)
    tmp = blt.buf_decay[np.newaxis, :] ** powers[:, np.newaxis] * blt.output_scale
    return np.concatenate([np.ones(1, dtype=np.float64), tmp.sum(axis=1)])


def materialize(blt: BufferedToeplitz, n: int) -> NDArray[np.float64]:
    """Convert to dense n x n matrix.

    Args:
        blt: The BLT.
        n: Matrix dimension.

    Returns:
        Dense lower-triangular Toeplitz matrix.
    """
    return toeplitz.materialize_lower_triangular(toeplitz_coefs(blt, n))


def inverse(blt: BufferedToeplitz, skip_checks: bool = False) -> BufferedToeplitz:
    """Compute the BLT parameterization of C^{-1}.

    Implements the inverse computation from Lemma 5.2 of
    https://arxiv.org/abs/2404.16706.

    Args:
        blt: The BLT to invert.
        skip_checks: Skip validation checks.

    Returns:
        A BufferedToeplitz representing C^{-1}.
    """
    if blt._num_buffers == 0:
        return BufferedToeplitz.build(buf_decay=[], output_scale=[])

    if not skip_checks and len(blt.buf_decay) > 1:
        gap = min_buf_decay_gap(blt.buf_decay)
        if gap < 1e-9:
            raise ValueError(
                "Input BLT has buf_decay values too close: "
                f"gap={float(gap)}, buf_decay={blt.buf_decay}"
            )

    nbuf = len(blt.buf_decay)
    Theta = np.diag(blt.buf_decay)
    omega = blt.output_scale
    alpha = np.ones(nbuf, dtype=np.float64)

    Theta2 = Theta - np.outer(omega, alpha)
    omega2 = -omega

    # Diagonalize Theta2
    evals = np.linalg.eigvals(Theta2).real

    # Closed-form eigenvectors
    evecs = omega[:, np.newaxis] / (evals[np.newaxis, :] - blt.buf_decay[:, np.newaxis])
    einv = np.linalg.inv(evecs)

    if not skip_checks:
        Theta2_diag = evecs @ np.diag(evals) @ einv
        if not np.allclose(Theta2_diag, Theta2, atol=1e-7):
            raise RuntimeError(
                f"Error computing inverse: Theta2 mismatch.\n"
                f"blt={blt}\nevecs={evecs}\nevals={evals}"
            )

    omega3 = (einv @ omega2) * (evecs.T @ alpha)
    return BufferedToeplitz.build(
        buf_decay=evals,
        output_scale=omega3,
        dtype=blt.dtype,
    )


def _streaming_matrix_builder(blt: BufferedToeplitz) -> _StreamingMatrixBuilder:
    """Create a _StreamingMatrixBuilder from a BLT."""
    dtype = np.float64
    return _StreamingMatrixBuilder(
        output_scale=blt.output_scale.astype(dtype, copy=False),
        buf_decay=blt.buf_decay.astype(dtype, copy=False),
    )


def as_streaming_matrix(blt: BufferedToeplitz) -> streaming_matrix.StreamingMatrix:
    """Returns a StreamingMatrix representing C.

    Args:
        blt: The BLT.

    Returns:
        StreamingMatrix for C.
    """
    return _streaming_matrix_builder(blt).build()


def inverse_as_streaming_matrix(
    blt: BufferedToeplitz,
) -> streaming_matrix.StreamingMatrix:
    """Returns a StreamingMatrix representing C^{-1}.

    Args:
        blt: The BLT.

    Returns:
        StreamingMatrix for C^{-1}.
    """
    return _streaming_matrix_builder(blt).build_inverse()


# ── Helper functions (unchanged) ────────────────────────────────────────


def min_buf_decay_gap(buf_decay: ArrayLike) -> np.float64:
    """Returns the minimum gap between buf_decay parameters.

    Args:
        buf_decay: The buf_decay parameters.

    Returns:
        min_{i!=j} |theta[i] - theta[j]|
    """
    theta = np.asarray(buf_decay, dtype=np.float64)
    A = theta[:, np.newaxis] - theta[np.newaxis, :]
    np.fill_diagonal(A, np.inf)
    return np.min(np.abs(A))


def _gt_zero_penalty(x: ArrayLike) -> np.float64:
    """Penalize values to enforce x > 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log(x).sum()


def _lt_one_penalty(x: ArrayLike) -> np.float64:
    """Penalize values to enforce x < 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log(1 - np.asarray(x)).sum()


def _lt_penalty(x: ArrayLike, upper_bound: float) -> np.float64:
    """Penalize values to enforce x < upper_bound."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log(upper_bound - np.asarray(x)).sum()


def _lt_zero_penalty(x: ArrayLike) -> np.float64:
    """Penalize values to enforce x < 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log(-np.asarray(x)).sum()


def geometric_sum(a: ArrayLike, r: ArrayLike, num: float = float("inf")):
    """Compute a + a*r + a*r^2 + ... + a*r^(num-1).

    Uses a quadratic Taylor approximation near r=1 for numerical stability.
    The threshold for switching to the Taylor series is chosen to minimise
    the error in the *gradient* w.r.t. ``r`` (not just the value), following
    the JAX-Privacy implementation.

    Args:
        a: Scale factor(s).
        r: Common ratio(s), requires |r| < 1.
        num: Number of terms (inf for infinite series).

    Returns:
        The sum.
    """
    if math.isinf(num):
        return a / (1 - r)

    n_val = np.asarray(num, dtype=np.float64)
    n = num

    # Adaptive threshold: calibrated to minimise gradient error.
    # Constants from regression on numerical experiments (see JAX-Privacy).
    _SLOPE = 0.53018965
    _INTERCEPT = 3.33503185
    pow_threshold = _INTERCEPT + _SLOPE * np.log(n_val)
    threshold = 1 - 10 ** (-pow_threshold)

    use_direct = r < threshold

    # Direct computation (safe when r is not near 1)
    safe_r = np.where(use_direct, r, np.zeros_like(r))
    direct = a * (1 - safe_r**n) / (1 - safe_r)

    # Quadratic Taylor polynomial at r = 1 (from sympy)
    x0 = n - 1
    x1 = r - 1
    series = (1 / 6) * a * n * (x0 * x1**2 * (n - 2) + 3 * x0 * x1 + 6)

    return np.where(use_direct, direct, series)


# ── BLT error and sensitivity (already standalone) ──────────────────────


def sensitivity_squared(blt: BufferedToeplitz, n: float) -> np.float64:
    """Compute sensitivity^2 for a BLT strategy matrix C.

    See Lemma 5.3 of https://arxiv.org/abs/2404.16706.

    Args:
        blt: The BLT representing C.
        n: Number of iterations (can be inf).

    Returns:
        Maximum column norm squared of C.
    """
    if blt._num_buffers == 0:
        return np.float64(1.0)

    if np.any(blt.buf_decay > 1):
        return np.float64(np.inf)

    omega = blt.output_scale
    theta = blt.buf_decay
    num = n - 1

    omega_pairs = omega[np.newaxis, :] * omega[:, np.newaxis]
    theta_pairs = theta[np.newaxis, :] * theta[:, np.newaxis]
    geo_pairs = geometric_sum(omega_pairs, theta_pairs, num=num)
    return 1.0 + geo_pairs.sum()


def max_error(inv_blt: BufferedToeplitz, n: float) -> np.float64:
    """Max squared error for any iteration 0, ..., n-1.

    Args:
        inv_blt: BLT representing C^{-1}.
        n: Number of iterations.

    Returns:
        Max squared error.
    """
    return iteration_error(inv_blt, n - 1)


def iteration_error(inv_blt: BufferedToeplitz, i: float) -> np.float64:
    """Compute the squared error on iteration i.

    For BLT matrices, the max error through iteration i is achieved at
    iteration i itself.

    Implements Lemma 5.4 of https://arxiv.org/abs/2404.16706.

    Args:
        inv_blt: BLT representing C^{-1}.
        i: Iteration index (0-based).

    Returns:
        Squared error at iteration i.
    """
    if inv_blt._num_buffers == 0:
        return np.float64(i + 1)

    n = i + 1
    omega = inv_blt.output_scale
    theta = inv_blt.buf_decay

    # Vectorised over buffers using broadcasting
    s1 = robust_max_error_Gamma_j(omega, theta, n).sum()
    s2 = robust_max_error_Gamma_jk(
        omega,
        theta,
        omega[:, np.newaxis],
        theta[:, np.newaxis],
        n,
    ).sum()

    return n * (1 + 2 * s1 + s2)


def limit_max_error(inv_blt: BufferedToeplitz) -> np.float64:
    """Closed-form max squared error per iteration as n -> infinity.

    This is the limit of ``max_error(inv_blt, n) / n`` as n grows.
    Uses direct arithmetic on the BLT parameters (no loop over n),
    making it O(num_buffers^2).

    Args:
        inv_blt: BLT representing C^{-1}.

    Returns:
        The asymptotic per-iteration max squared error.
    """
    if inv_blt._num_buffers == 0:
        return np.float64(1.0)

    omega = inv_blt.output_scale
    theta = inv_blt.buf_decay

    omega_pairs = omega[np.newaxis, :] * omega[:, np.newaxis]
    theta_complement_pairs = (1 - theta)[np.newaxis, :] * (1 - theta)[:, np.newaxis]
    cross_term_sum = (omega_pairs / theta_complement_pairs).sum()

    return 1 + 2 * (omega / (1 - theta)).sum() + cross_term_sum


def limit_max_loss(blt: BufferedToeplitz) -> np.float64:
    """Closed-form loss (error * sensitivity^2) as n -> infinity.

    Composes ``limit_max_error`` with ``sensitivity_squared(blt, inf)``.

    Args:
        blt: BLT representing the strategy C.

    Returns:
        The asymptotic loss.
    """
    sens_sq = sensitivity_squared(blt, n=float("inf"))
    inv_blt = inverse(blt, skip_checks=True)
    return limit_max_error(inv_blt) * sens_sq


def _max_error_Gamma_j(omega, theta, n):
    """Direct computation of Gamma_j for max error."""
    return (omega / (1.0 - theta)) * (
        1 - geometric_sum(np.ones_like(theta), theta, num=n) / n
    )


def _max_error_Gamma_j_series(omega, theta, n):
    """Taylor series approximation to _max_error_Gamma_j near theta=1.

    Auto-generated via sympy (see robust_max_error_for_blts.ipynb in
    JAX-Privacy).
    """
    x0 = theta - 1
    x1 = omega * (n - 2) * (n - 1)
    return (
        -omega * (1 / 2 - n / 2) + (1 / 24) * x0**2 * x1 * (n - 3) + (1 / 6) * x0 * x1
    )


def robust_max_error_Gamma_j(omega, theta, n):
    """Robustly compute Gamma_j, dispatching to Taylor series near theta=1.

    Uses empirically-calibrated thresholds from JAX-Privacy to decide
    when to switch from the direct formula to a Taylor series.
    """
    n_t = np.asarray(n, dtype=np.float64)
    _J_SLOPE = 0.43877484
    _J_INTERCEPT = 2.91215085
    power = _J_INTERCEPT + _J_SLOPE * np.log(n_t)
    threshold = 1 - 10 ** (-power)

    use_direct = theta < threshold
    safe_theta = np.where(use_direct, theta, np.zeros_like(theta))
    v0 = _max_error_Gamma_j(omega, safe_theta, n)
    v1 = _max_error_Gamma_j_series(omega, theta, n)
    return np.where(use_direct, v0, v1)


def _max_error_Gamma_jk(omega1, theta1, omega2, theta2, n):
    """Direct computation of cross term Gamma_jk for max error."""
    temp1 = omega1 * omega2 / ((1 - theta1) * (1 - theta2))
    gs1 = geometric_sum(np.ones_like(theta1), theta1, num=n)
    gs2 = geometric_sum(np.ones_like(theta2), theta2, num=n)
    gs12 = geometric_sum(np.ones_like(theta1), theta1 * theta2, num=n)
    temp2 = (n - gs1 - gs2 + gs12) / n
    return temp1 * temp2


def _max_error_Gamma_jk_series_j(omega1, theta1, omega2, theta2, n):
    """Taylor series in theta1 near 1, theta2 handled directly.

    Auto-generated via sympy (see robust_max_error_for_blts.ipynb in
    JAX-Privacy).
    """
    # fmt: off
    x0 = theta2 - 1
    x1 = theta2 ** (n + 1)
    x2 = -x1
    x3 = 6 * x0
    x4 = theta2 ** n
    x5 = n - 1
    x6 = theta1 - 1
    x7 = theta2 ** (n + 2)
    return (-1 / 6 * omega1 * omega2 * (
        n * x0 ** 3 * (3 * n + x5 * x6 * (n - 2) - 3) + n * x3 * (x2 + x4) - x3 * (theta2 + x2)
        + 3 * x6 * (n * x0 * (-x1 * x5 + x4 * x5) - 2 * n * (x1 - x7)
                     + 2 * theta2 ** 2 - 2 * x7)) / (n * x0 ** 4))
    # fmt: on


def _max_error_Gamma_jk_series_jk(omega1, theta1, omega2, theta2, n):
    """Taylor series in both theta1 and theta2 near 1.

    Auto-generated via sympy (see robust_max_error_for_blts.ipynb in
    JAX-Privacy).
    """
    # fmt: off
    x0 = n ** 2
    x1 = 3 * n ** 3 + 9 * n - 10 * x0 - 2
    return ((1 / 24) * omega1 * omega2 * (-12 * n + 8 * x0 + x1 * (theta1 - 1)
                                           + x1 * (theta2 - 1) + 4))
    # fmt: on


def robust_max_error_Gamma_jk(omega1, theta1, omega2, theta2, n):
    """Robustly compute Gamma_jk, dispatching to Taylor series.

    Uses three-way dispatch based on how close theta1/theta2 are to 1:
    - Both far from 1: direct formula
    - theta1 near 1, theta2 far: Taylor series in theta1
    - Both near 1: Taylor series in both

    The series_j approximation requires theta1 >= theta2, so we sort.
    """
    n_t = np.asarray(n, dtype=np.float64)
    # Ensure theta1 >= theta2 (required by series_j)
    theta1, theta2 = np.maximum(theta1, theta2), np.minimum(theta1, theta2)

    _JK_SLOPE = 0.35321577
    _JK_INTERCEPT = 2.81518052
    power = _JK_INTERCEPT + _JK_SLOPE * np.log(n_t)
    threshold = 1 - 10 ** (-power)

    v0_predicate = theta1 < threshold
    v1_predicate = theta2 < threshold

    # Avoid inf/nan in untaken branches
    safe_theta1 = np.where(v0_predicate, theta1, np.zeros_like(theta1))
    safe_theta2_v0 = np.where(v0_predicate, theta2, np.zeros_like(theta2))
    v0 = _max_error_Gamma_jk(omega1, safe_theta1, omega2, safe_theta2_v0, n)

    safe_theta2_v1 = np.where(v1_predicate, theta2, np.zeros_like(theta2))
    v1 = _max_error_Gamma_jk_series_j(omega1, theta1, omega2, safe_theta2_v1, n)
    v2 = _max_error_Gamma_jk_series_jk(omega1, theta1, omega2, theta2, n)

    return np.where(
        v0_predicate,
        v0,
        np.where(v1_predicate, v1, v2),
    )


# ── Loss function and optimization ─────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class LossFn:
    """Encapsulates the loss to be optimized for a BLT.

    Attributes:
        error_for_inv: Error function taking C^{-1} BLT.
        sensitivity_squared_fn: Sensitivity function taking C BLT.
        n: Number of iterations.
        min_sep: Minimum separation of participations.
        max_participations: Maximum participations.
        penalty_strength: Multiplier for constraint penalties.
    """

    error_for_inv: Callable[[BufferedToeplitz], float]
    sensitivity_squared_fn: Callable[[BufferedToeplitz], float]
    n: int
    min_sep: int
    max_participations: int
    penalty_strength: float = 1e-8
    max_second_coef: float = 1.0
    min_theta_gap: float = 1e-12
    error: str | None = None
    workload_coef: NDArray[np.float64] | None = None
    query_weights: NDArray[np.float64] | None = None

    @classmethod
    def build_closed_form_single_participation(cls, n: int, **kwargs) -> LossFn:
        """Construct LossFn for single participation max-error."""
        return cls(
            error_for_inv=functools.partial(max_error, n=n),
            sensitivity_squared_fn=functools.partial(sensitivity_squared, n=n),
            n=n,
            min_sep=1,
            max_participations=1,
            **kwargs,
        )

    @classmethod
    def build_min_sep(
        cls,
        n: int,
        error: str = "max",
        min_sep: int = 1,
        max_participations: int | None = None,
        workload_coef: ArrayLike | None = None,
        query_weights: ArrayLike | None = None,
        **kwargs,
    ) -> LossFn:
        """Construct LossFn for min-sep participation."""

        def mean_error_fn(inv_blt):
            return toeplitz.mean_error(
                noising_coef=toeplitz_coefs(inv_blt, n),
                workload_coef=workload_coef,
                query_weights=query_weights,
            )

        def max_error_fn(inv_blt):
            return toeplitz.max_error(
                noising_coef=toeplitz_coefs(inv_blt, n),
                workload_coef=workload_coef,
                query_weights=query_weights,
            )

        if error == "mean":
            error_fn = mean_error_fn
        elif error == "max":
            error_fn = max_error_fn
        else:
            raise ValueError(f"Unknown error={error}")

        def minsep_sens_sq(blt):
            return toeplitz.minsep_sensitivity_squared(
                strategy_coef=toeplitz_coefs(blt, n),
                min_sep=min_sep,
                max_participations=max_participations,
                skip_checks=True,
            )

        return cls(
            n=n,
            error_for_inv=error_fn,
            sensitivity_squared_fn=minsep_sens_sq,
            min_sep=min_sep,
            max_participations=sensitivity.minsep_true_max_participations(
                n=n,
                min_sep=min_sep,
                max_participations=max_participations,
            ),
            error=error,
            workload_coef=(
                None
                if workload_coef is None
                else np.asarray(workload_coef, dtype=np.float64)
            ),
            query_weights=(
                None
                if query_weights is None
                else np.asarray(query_weights, dtype=np.float64)
            ),
            **kwargs,
        )

    def compute_penalties(
        self, blt: BufferedToeplitz, inv_blt: BufferedToeplitz
    ) -> np.float64:
        """Compute log-barrier penalties enforcing the BLT feasible set.

        Penalties enforce:
        - buf_decay in (0, 1)
        - output_scale > 0
        - inv_buf_decay in (0, 1)
        - inv_output_scale < 0
        - buf_decay gap >= min_theta_gap
        - second Toeplitz coefficient <= max_second_coef
        - Pillutla score < 1

        Args:
            blt: The strategy BLT (C).
            inv_blt: The inverse BLT (C^{-1}).

        Returns:
            Scalar penalty value.
        """
        penalty = np.float64(0.0)

        if blt._num_buffers == 0:
            return penalty

        # buf_decay in (0, 1)
        penalty = penalty + _gt_zero_penalty(blt.buf_decay)
        penalty = penalty + _lt_one_penalty(blt.buf_decay)

        # output_scale > 0
        penalty = penalty + _gt_zero_penalty(blt.output_scale)

        # inv_buf_decay in (0, 1)
        penalty = penalty + _gt_zero_penalty(inv_blt.buf_decay)
        penalty = penalty + _lt_one_penalty(inv_blt.buf_decay)

        # inv_output_scale < 0
        penalty = penalty + _lt_zero_penalty(inv_blt.output_scale)

        # buf_decay gap separation
        if blt._num_buffers > 1:
            gap = min_buf_decay_gap(blt.buf_decay)
            penalty = penalty - np.log(gap - self.min_theta_gap)

        # Second coefficient: sum(output_scale) <= max_second_coef
        second_coef = np.sum(blt.output_scale)
        penalty = penalty + _lt_penalty(second_coef, self.max_second_coef)

        # Pillutla score < 1
        pillutla = np.sum(blt.output_scale / blt.buf_decay)
        penalty = penalty + _lt_one_penalty(np.atleast_1d(pillutla))

        return penalty


def loss(
    loss_fn: LossFn, blt: BufferedToeplitz, skip_checks: bool = False
) -> np.float64:
    """Returns the loss (not including penalties).

    Args:
        loss_fn: The loss function configuration.
        blt: The BLT to evaluate.
        skip_checks: Skip validation.

    Returns:
        error * sensitivity_squared.
    """
    try:
        inv_blt = inverse(blt, skip_checks=skip_checks)
    except (RuntimeError, ValueError):
        return np.float64(np.inf)
    error = loss_fn.error_for_inv(inv_blt)
    sens_sq = loss_fn.sensitivity_squared_fn(blt)
    return error * sens_sq


def penalized_loss(
    loss_fn: LossFn, blt: BufferedToeplitz, inv_blt: BufferedToeplitz
) -> np.float64:
    """Returns loss + penalty_strength * penalties.

    This is the objective function used during L-BFGS optimization.

    Args:
        loss_fn: The loss function configuration.
        blt: The strategy BLT (C).
        inv_blt: The inverse BLT (C^{-1}).

    Returns:
        loss + penalties.
    """
    error = loss_fn.error_for_inv(inv_blt)
    sens_sq = loss_fn.sensitivity_squared_fn(blt)
    loss_val = error * sens_sq
    penalties = loss_fn.compute_penalties(blt, inv_blt)
    result = loss_val + loss_fn.penalty_strength * penalties
    if not np.isfinite(result):
        return np.float64(1e100)
    return result


def blt_pair_from_theta_pair(
    theta: ArrayLike, theta_hat: ArrayLike
) -> tuple[BufferedToeplitz, BufferedToeplitz]:
    """Compute BLTs (C, C_inv) from theta and theta_hat.

    Implements Lemma 5.2 of https://arxiv.org/abs/2404.16706.

    Args:
        theta: Array of thetas for C.
        theta_hat: Array of thetas for C^{-1}.

    Returns:
        Tuple (C_blt, C_inv_blt).
    """
    theta = np.asarray(theta, dtype=np.float64)
    theta_hat = np.asarray(theta_hat, dtype=np.float64)

    def get_omega(th, th_hat):
        numerators = np.prod(th[:, np.newaxis] - th_hat[np.newaxis, :], axis=1)
        A = th[:, np.newaxis] - th[np.newaxis, :]
        np.fill_diagonal(A, 1.0)
        denominators = np.prod(A, axis=1)
        return numerators / denominators

    return (
        BufferedToeplitz.build(
            output_scale=get_omega(theta, theta_hat),
            buf_decay=theta,
        ),
        BufferedToeplitz.build(
            output_scale=get_omega(theta_hat, theta),
            buf_decay=theta_hat,
        ),
    )


def get_init_blt(
    num_buffers: int = 3,
    init_blt: BufferedToeplitz | None = None,
) -> BufferedToeplitz:
    """Returns an initial BLT for optimization.

    Args:
        num_buffers: Number of buffers.
        init_blt: Optional explicit initialization.

    Returns:
        A BufferedToeplitz for initialization.
    """
    if init_blt is None:
        if num_buffers == 0:
            return BufferedToeplitz.build(buf_decay=[], output_scale=[])
        return BufferedToeplitz.from_rational_approx_to_sqrt_x(
            num_buffers=num_buffers,
            max_buf_decay=1 - 1e-6,
            max_pillutla_score=1 - 1e-6,
        )

    if len(init_blt.buf_decay) != num_buffers:
        raise ValueError(
            f"num_buffers={num_buffers} does not match "
            f"len(init_blt.buf_decay)={len(init_blt.buf_decay)}"
        )
    return init_blt


@dataclasses.dataclass(frozen=True)
class Parameterization:
    """A reparameterization of a BufferedToeplitz for L-BFGS optimization.

    Provides the mapping between optimizable parameters and the BLT pair
    (strategy, noising) used for loss computation.

    Attributes:
        params_from_blt: Extracts optimizable parameters from a BLT.
        blt_and_inverse_from_params: Reconstructs (C, C^{-1}) from parameters.
    """

    params_from_blt: Callable[[BufferedToeplitz], NDArray[np.float64]]
    blt_and_inverse_from_params: Callable[
        ..., tuple[BufferedToeplitz, BufferedToeplitz]
    ]
    supports_analytic_gradient: bool = False

    @classmethod
    def buf_decay_pair(cls) -> Parameterization:
        """Parameterization over the pair (theta, theta_hat).

        Optimises the buf_decay arrays of C and C^{-1} jointly, which is
        more numerically stable than optimising the full BLT and avoids
        the SVD inside ``inverse()``.

        Returns:
            A Parameterization.
        """

        def params_from_blt(blt: BufferedToeplitz) -> NDArray[np.float64]:
            inv_blt = inverse(blt)
            return np.concatenate([blt.buf_decay, inv_blt.buf_decay])

        def blt_and_inverse_from_params(
            params: ArrayLike,
        ) -> tuple[BufferedToeplitz, BufferedToeplitz]:
            half = len(params) // 2
            theta = params[:half]
            theta_hat = params[half:]
            return blt_pair_from_theta_pair(theta, theta_hat)

        return cls(
            params_from_blt=params_from_blt,
            blt_and_inverse_from_params=blt_and_inverse_from_params,
            supports_analytic_gradient=True,
        )


def get_parameterized_loss(
    param: Parameterization, loss_fn: LossFn
) -> Callable[[NDArray[np.float64]], float]:
    """Returns a scalar loss function over the flat parameter vector.

    Args:
        param: The parameterization.
        loss_fn: The loss function.

    Returns:
        A callable mapping flat parameters to scalar loss.
    """

    def parameterized_loss(params: NDArray[np.float64]) -> float:
        # Numerical differentiation probes points near active constraints.
        # Those points are intentionally converted to a large finite penalty.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return float(
                penalized_loss(loss_fn, *param.blt_and_inverse_from_params(params))
            )

    return parameterized_loss


def _rational_weights_and_jacobian(
    theta: NDArray[np.float64], theta_hat: NDArray[np.float64]
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Return BLT rational weights and their derivatives for a theta pair."""
    nbuf = len(theta)
    difference = theta[:, np.newaxis] - theta[np.newaxis, :]
    np.fill_diagonal(difference, 1.0)
    cross_difference = theta[:, np.newaxis] - theta_hat[np.newaxis, :]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        omega = np.prod(cross_difference, axis=1) / np.prod(difference, axis=1)

    d_theta = np.empty((nbuf, nbuf), dtype=np.float64)
    for i in range(nbuf):
        other = np.arange(nbuf) != i
        d_theta[i] = 0.0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            d_theta[i, other] = omega[i] / (theta[i] - theta[other])
            d_theta[i, i] = omega[i] * (
                np.sum(1.0 / cross_difference[i]) - np.sum(1.0 / difference[i, other])
            )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d_theta_hat = -omega[:, np.newaxis] / cross_difference
    return omega, d_theta, d_theta_hat, cross_difference


def _blt_coefs_and_jacobian(
    theta: NDArray[np.float64], theta_hat: NDArray[np.float64], n: int
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Return C/C⁻¹ Toeplitz coefficients and exact derivatives by parameters."""
    omega, d_omega_theta, d_omega_theta_hat, _ = _rational_weights_and_jacobian(
        theta, theta_hat
    )
    inv_omega, d_inv_omega_hat, d_inv_omega_theta, _ = _rational_weights_and_jacobian(
        theta_hat, theta
    )
    nbuf = len(theta)
    powers = np.arange(n - 1, dtype=np.float64)[:, np.newaxis]

    def coefficients_and_jacobian(
        decay: NDArray[np.float64],
        weights: NDArray[np.float64],
        d_weights_first: NDArray[np.float64],
        d_weights_second: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        decay_powers = decay[np.newaxis, :] ** powers
        coef = np.empty(n, dtype=np.float64)
        coef[0] = 1.0
        coef[1:] = (decay_powers * weights).sum(axis=1)
        jac = np.zeros((n, 2 * nbuf), dtype=np.float64)
        jac[1:, :nbuf] = decay_powers @ d_weights_first
        jac[1:, nbuf:] = decay_powers @ d_weights_second
        if n > 1:
            decay_derivative = np.where(
                powers == 0,
                0.0,
                powers * decay[np.newaxis, :] ** (powers - 1),
            )
            jac[1:, :nbuf] += decay_derivative * weights
        return coef, jac

    coefs, coefs_jac = coefficients_and_jacobian(
        theta, omega, d_omega_theta, d_omega_theta_hat
    )
    inv_coefs, inv_coefs_jac_swapped = coefficients_and_jacobian(
        theta_hat, inv_omega, d_inv_omega_hat, d_inv_omega_theta
    )
    inv_coefs_jac = np.concatenate(
        [inv_coefs_jac_swapped[:, nbuf:], inv_coefs_jac_swapped[:, :nbuf]], axis=1
    )
    return coefs, coefs_jac, inv_coefs, inv_coefs_jac


def _minsep_sensitivity_and_jacobian(
    coefs: NDArray[np.float64], jacobian: NDArray[np.float64], loss_fn: LossFn
) -> tuple[float, NDArray[np.float64]]:
    """Differentiate the min-separation sensitivity's linear prefix transform."""
    n = loss_fn.n
    padding = (loss_fn.min_sep - n) % loss_fn.min_sep
    full = np.pad(coefs, (0, padding))
    full_jacobian = np.pad(jacobian, ((0, padding), (0, 0)))
    vector = full.reshape(-1, loss_fn.min_sep).cumsum(axis=0).reshape(-1)
    vector_jacobian = (
        full_jacobian.reshape(-1, loss_fn.min_sep, jacobian.shape[1])
        .cumsum(axis=0)
        .reshape(-1, jacobian.shape[1])
    )
    active = loss_fn.min_sep * loss_fn.max_participations
    if active < len(vector):
        vector[active:] -= vector[: len(vector) - active]
        vector_jacobian[active:] -= vector_jacobian[: len(vector) - active]
    vector = vector[:n]
    vector_jacobian = vector_jacobian[:n]
    return float(vector @ vector), 2.0 * vector @ vector_jacobian


def _error_and_jacobian(
    noising_coefs: NDArray[np.float64],
    jacobian: NDArray[np.float64],
    loss_fn: LossFn,
) -> tuple[float, NDArray[np.float64]]:
    """Differentiate the host Toeplitz workload error for a BLT inverse."""
    n = loss_fn.n
    if loss_fn.workload_coef is None:
        workload = np.ones(n, dtype=np.float64)
    else:
        workload = toeplitz.pad_coefs_to_n(loss_fn.workload_coef, n)
    noising = noising_coefs[:n]
    b_coef = np.convolve(workload, noising)[:n]
    b_jacobian = np.stack(
        [np.convolve(workload, jacobian[:, i])[:n] for i in range(jacobian.shape[1])],
        axis=1,
    )
    per_query = np.cumsum(b_coef**2)
    per_query_jacobian = np.cumsum(2.0 * b_coef[:, np.newaxis] * b_jacobian, axis=0)
    if loss_fn.query_weights is not None:
        weights_squared = np.square(loss_fn.query_weights)
        per_query *= weights_squared
        per_query_jacobian *= weights_squared[:, np.newaxis]
    if loss_fn.error == "mean":
        return float(per_query.mean()), per_query_jacobian.mean(axis=0)
    index = int(np.argmax(per_query))
    return float(per_query[index]), per_query_jacobian[index]


def get_parameterized_loss_and_gradient(
    param: Parameterization, loss_fn: LossFn
) -> Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]] | None:
    """Build an exact host-side BLT loss/Jacobian callable when supported.

    The min-separation objective is differentiated through the rational BLT
    parameterization and all Toeplitz operations.  The small feasibility
    barrier is intentionally differentiated separately with a centred host
    difference because it is only active close to a constraint boundary.
    """
    if not param.supports_analytic_gradient or loss_fn.error not in {"max", "mean"}:
        return None

    def barrier_is_inactive(blt: BufferedToeplitz, inv_blt: BufferedToeplitz) -> bool:
        """Whether every log-barrier argument is safely away from zero."""
        slack = [
            blt.buf_decay,
            1.0 - blt.buf_decay,
            blt.output_scale,
            inv_blt.buf_decay,
            1.0 - inv_blt.buf_decay,
            -inv_blt.output_scale,
            np.atleast_1d(loss_fn.max_second_coef - np.sum(blt.output_scale)),
            np.atleast_1d(1.0 - np.sum(blt.output_scale / blt.buf_decay)),
        ]
        if blt._num_buffers > 1:
            slack.append(
                np.atleast_1d(min_buf_decay_gap(blt.buf_decay) - loss_fn.min_theta_gap)
            )
        return all(
            np.all(np.isfinite(values)) and np.all(values >= 1e-3) for values in slack
        )

    def value_and_gradient(
        params: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        params = np.asarray(params, dtype=np.float64)
        half = len(params) // 2
        theta, theta_hat = params[:half], params[half:]
        try:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                coefs, coefs_jac, inv_coefs, inv_coefs_jac = _blt_coefs_and_jacobian(
                    theta, theta_hat, loss_fn.n
                )
                error, error_jac = _error_and_jacobian(
                    inv_coefs, inv_coefs_jac, loss_fn
                )
                sensitivity_squared, sensitivity_jac = _minsep_sensitivity_and_jacobian(
                    coefs, coefs_jac, loss_fn
                )
                blt, inv_blt = param.blt_and_inverse_from_params(params)
                penalty = float(loss_fn.compute_penalties(blt, inv_blt))
                value = error * sensitivity_squared + loss_fn.penalty_strength * penalty
                gradient = error_jac * sensitivity_squared + error * sensitivity_jac
        except (FloatingPointError, ValueError, ZeroDivisionError):
            return 1e100, np.zeros_like(params)

        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return 1e100, np.zeros_like(params)

        # The feasibility barrier has a deliberately tiny multiplier.  Avoid
        # its expensive centred derivative while every constraint is safely
        # inactive; near a boundary retain the exact existing correction.
        if barrier_is_inactive(blt, inv_blt):
            return float(value), np.asarray(gradient, dtype=np.float64)

        for index in range(len(params)):
            step = min(
                1e-6 * (1.0 + abs(params[index])),
                0.25 * min(params[index], 1.0 - params[index]),
            )
            if step <= 0:
                continue
            plus = params.copy()
            minus = params.copy()
            plus[index] += step
            minus[index] -= step
            try:
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    plus_penalty = loss_fn.compute_penalties(
                        *param.blt_and_inverse_from_params(plus)
                    )
                    minus_penalty = loss_fn.compute_penalties(
                        *param.blt_and_inverse_from_params(minus)
                    )
                gradient[index] += (
                    loss_fn.penalty_strength
                    * (plus_penalty - minus_penalty)
                    / (2.0 * step)
                )
            except (FloatingPointError, ValueError, ZeroDivisionError):
                pass
        return float(value), np.asarray(gradient, dtype=np.float64)

    return value_and_gradient


def optimize_loss(
    loss_fn: LossFn,
    num_buffers: int = 1,
    init_blt: BufferedToeplitz | None = None,
    parameterization: Parameterization | None = None,
    max_optimizer_steps: int = 600,
    **kwargs,
) -> tuple[BufferedToeplitz, np.float64]:
    """Optimise a BLT for a fixed number of buffers using L-BFGS.

    This is the main workhorse: it initialises a BLT, converts it to an
    optimisable parameter vector via ``parameterization``, runs L-BFGS,
    and returns the resulting BLT together with its (unpenalised) loss.

    Args:
        loss_fn: The loss function to minimise.
        num_buffers: Number of buffers for the BLT.
        init_blt: Optional explicit initial BLT.
        parameterization: Reparameterization to use.  Defaults to
            ``Parameterization.buf_decay_pair()``.
        max_optimizer_steps: Maximum L-BFGS iterations.
        **kwargs: Additional keyword arguments forwarded to
            ``_lbfgs_optimize``.

    Returns:
        Tuple ``(blt, loss_val)`` where *blt* is the optimised BLT and
        *loss_val* is the unpenalised loss value.

    Raises:
        RuntimeError: If the optimiser produces a BLT with non-finite loss.
    """
    from ._toeplitz import _lbfgs_optimize

    if num_buffers == 0:
        blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
        return blt, loss(loss_fn, blt)

    if parameterization is None:
        parameterization = Parameterization.buf_decay_pair()

    blt = get_init_blt(num_buffers=num_buffers, init_blt=init_blt)
    params = parameterization.params_from_blt(blt)

    loss_fn_to_optimize = get_parameterized_loss(parameterization, loss_fn)
    loss_and_grad_fn = get_parameterized_loss_and_gradient(parameterization, loss_fn)

    # For buf_decay_pair parameterization, all parameters are buf_decay
    # values (theta and theta_hat) which must be in (0, 1).
    if "bounds" not in kwargs:
        eps = 1e-9
        kwargs["bounds"] = [(eps, 1.0 - eps)] * len(params)
    kwargs.setdefault(
        "restart_stalled_analytic_optimization", loss_and_grad_fn is not None
    )

    params = _lbfgs_optimize(
        loss_and_grad_fn if loss_and_grad_fn is not None else loss_fn_to_optimize,
        params,
        max_optimizer_steps=max_optimizer_steps,
        grad=loss_and_grad_fn is not None,
        **kwargs,
    )

    blt, _ = parameterization.blt_and_inverse_from_params(params)
    blt = canonicalize(blt)

    loss_val = loss(loss_fn, blt)
    if not np.isfinite(loss_val):
        raise RuntimeError(
            f"Optimization produced BLT with non-finite loss {loss_val}:\n{blt}"
        )

    if np.any(np.abs(blt.output_scale) < 1e-8):
        logger.warning(
            "BLT has near-zero output_scale parameters, which "
            "means some buffers are ignored. Consider re-optimizing "
            "with a smaller number of buffers.\n%s",
            blt,
        )
    return blt, loss_val


def _optimize_increasing_nbuf(
    opt_blt_and_loss_fn: Callable[[int], tuple[BufferedToeplitz, float]],
    min_buffers: int = 0,
    max_buffers: int = 10,
    rtol: float = 1.02,
) -> BufferedToeplitz:
    """Increase num_buffers until improvement < rtol."""
    prev_blt, prev_loss = opt_blt_and_loss_fn(min_buffers)
    for nbuf in range(min_buffers + 1, max_buffers + 1):
        try:
            blt, loss_val = opt_blt_and_loss_fn(nbuf)
        except RuntimeError as err:
            logger.warning("Optimization failed for %d buffers: %s", nbuf, err)
            blt, loss_val = None, float("inf")

        if rtol * loss_val < prev_loss:
            prev_blt, prev_loss = blt, loss_val
        else:
            return prev_blt
    return prev_blt


def optimize(
    *,
    n: int,
    min_sep: int = 1,
    max_participations: int | None = 1,
    error: str = "max",
    min_buffers: int = 0,
    max_buffers: int = 10,
    rtol: float = 1.01,
    workload_coef: ArrayLike | None = None,
    query_weights: ArrayLike | None = None,
    **kwargs,
) -> BufferedToeplitz:
    """Compute an optimised BLT with dynamically-chosen num_buffers.

    Uses L-BFGS to refine BLT parameters at each buffer count, then
    increases the buffer count until improvement drops below ``rtol``.

    For single-participation max-error, uses closed-form sensitivity and
    error from https://arxiv.org/abs/2404.16706, so optimisation time is
    essentially independent of ``n``.  For multi-participation or mean
    error, materialises Toeplitz coefficients (benefits from GPU for
    large ``n``).

    Args:
        n: Number of iterations the mechanism is optimised for.
        min_sep: Minimum separation of participations.
        max_participations: Maximum participations.
        error: ``'max'`` or ``'mean'``.
        min_buffers: Minimum buffers to try (inclusive).
        max_buffers: Maximum buffers to try (inclusive).
        rtol: Relative tolerance for loss improvement.
        workload_coef: Toeplitz coefficients of the workload matrix.
            Defaults to ``None`` (prefix-sum workload).  For momentum-SGD
            with coefficient β, pass ``[1, β, β², ...]``.
        query_weights: Per-query workload row weights.
        **kwargs: Additional keyword arguments forwarded to
            ``optimize_loss`` / ``_lbfgs_optimize``.

    Returns:
        An optimised BLT.
    """
    if max_buffers > 15:
        raise ValueError("max_buffers > 15 will likely cause numerical issues.")

    k = sensitivity.minsep_true_max_participations(
        n=n, min_sep=min_sep, max_participations=max_participations
    )

    # Closed-form path only works for an unweighted prefix-sum workload.
    if k == 1 and error == "max" and workload_coef is None and query_weights is None:
        loss_fn = LossFn.build_closed_form_single_participation(n=n)
    else:
        loss_fn = LossFn.build_min_sep(
            n=n,
            error=error,
            min_sep=min_sep,
            max_participations=max_participations,
            workload_coef=workload_coef,
            query_weights=query_weights,
        )

    return _optimize_increasing_nbuf(
        opt_blt_and_loss_fn=lambda nbuf: optimize_loss(
            loss_fn=loss_fn,
            num_buffers=nbuf,
            parameterization=Parameterization.buf_decay_pair(),
            **kwargs,
        ),
        min_buffers=min_buffers,
        max_buffers=max_buffers,
        rtol=rtol,
    )
