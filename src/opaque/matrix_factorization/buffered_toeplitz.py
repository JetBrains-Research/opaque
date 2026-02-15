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
from collections.abc import Callable

import numpy as np
import torch

from . import sensitivity, streaming_matrix, toeplitz

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StreamingMatrixBuilder:
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

    def _init(self, abstract_value: torch.Tensor) -> torch.Tensor:
        num_buffers = self.buf_decay.shape[0]
        dtype = torch.float64
        zero = torch.zeros_like(abstract_value, dtype=dtype)
        return zero.unsqueeze(0).expand(num_buffers, *zero.shape).clone()

    def _read(self, state: torch.Tensor) -> torch.Tensor:
        output_scale = torch.tensor(self.output_scale, dtype=torch.float64)
        return torch.tensordot(output_scale, state, dims=([0], [0]))

    def _update(
        self, state: torch.Tensor, next_rhs_value: torch.Tensor
    ) -> torch.Tensor:
        buf_decay = torch.tensor(self.buf_decay, dtype=torch.float64)
        bufs = torch.diag(buf_decay) @ state.reshape(len(buf_decay), -1)
        bufs = bufs.reshape(state.shape) + next_rhs_value
        return bufs

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

    buf_decay: torch.Tensor
    output_scale: torch.Tensor

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
        dtype=torch.float64,
    ) -> BufferedToeplitz:
        """Construct and canonicalize a BLT.

        Args:
            buf_decay: Decay parameters.
            output_scale: Scale parameters.
            dtype: Data type (float64 recommended for numerical stability).

        Returns:
            A canonicalized BufferedToeplitz.
        """
        blt = cls(
            buf_decay=torch.as_tensor(buf_decay, dtype=dtype),
            output_scale=torch.as_tensor(output_scale, dtype=dtype),
        )
        return blt.canonicalize()

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
        blt = inv_blt.inverse()
        buf_decay_t = blt.buf_decay.clone()
        output_scale_t = blt.output_scale.clone()

        # Scale buf_decay if needed
        largest = buf_decay_t[0]
        scale = min(1.0, max_buf_decay / float(largest))
        buf_decay_t = buf_decay_t * scale

        if max_pillutla_score is not None:
            score = float(torch.sum(output_scale_t / buf_decay_t))
            score_scale = min(1.0, max_pillutla_score / score)
            output_scale_t = output_scale_t * score_scale

        return cls.build(buf_decay=buf_decay_t, output_scale=output_scale_t)

    def canonicalize(self) -> BufferedToeplitz:
        """Return a BLT with buf_decay in decreasing order."""
        self.validate()
        idx = torch.argsort(self.buf_decay, descending=True)
        return BufferedToeplitz(
            buf_decay=self.buf_decay[idx],
            output_scale=self.output_scale[idx],
        )

    @property
    def dtype(self):
        return self.buf_decay.dtype

    @property
    def _num_buffers(self) -> int:
        return self.buf_decay.shape[0]

    def toeplitz_coefs(self, n: int) -> torch.Tensor:
        """Returns the Toeplitz coefficients for C.

        Args:
            n: Number of coefficients.

        Returns:
            Tensor of n Toeplitz coefficients.
        """
        if self._num_buffers == 0:
            result = torch.zeros(n, dtype=self.dtype)
            result[0] = 1.0
            return result
        powers = torch.arange(n - 1, dtype=self.dtype)
        tmp = self.buf_decay.unsqueeze(0) ** powers.unsqueeze(1) * self.output_scale
        return torch.cat([torch.ones(1, dtype=self.dtype), tmp.sum(dim=1)])

    def materialize(self, n: int) -> torch.Tensor:
        """Convert to dense n x n matrix."""
        return toeplitz.materialize_lower_triangular(self.toeplitz_coefs(n))

    def inverse(self, skip_checks: bool = False) -> BufferedToeplitz:
        """Compute the BLT parameterization of C^{-1}.

        Implements the inverse computation from Lemma 5.2 of
        https://arxiv.org/abs/2404.16706.

        Args:
            skip_checks: Skip validation checks.

        Returns:
            A BufferedToeplitz representing C^{-1}.
        """
        if self._num_buffers == 0:
            return BufferedToeplitz.build(buf_decay=[], output_scale=[])

        blt = self
        if not skip_checks and len(blt.buf_decay) > 1:
            gap = min_buf_decay_gap(blt.buf_decay)
            if gap < 1e-9:
                raise ValueError(
                    "Input BLT has buf_decay values too close: "
                    f"gap={float(gap)}, buf_decay={blt.buf_decay}"
                )

        nbuf = len(blt.buf_decay)
        Theta = torch.diag(blt.buf_decay)
        omega = blt.output_scale
        alpha = torch.ones(nbuf, dtype=blt.dtype)

        Theta2 = Theta - torch.outer(omega, alpha)
        omega2 = -omega

        # Diagonalize Theta2
        evals = torch.linalg.eigvals(Theta2).real

        # Closed-form eigenvectors
        evecs = omega.unsqueeze(1) / (evals.unsqueeze(0) - blt.buf_decay.unsqueeze(1))
        einv = torch.linalg.inv(evecs)

        if not skip_checks:
            Theta2_diag = evecs @ torch.diag(evals) @ einv
            if not torch.allclose(Theta2_diag, Theta2, atol=1e-7):
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

    def pillutla_score(self) -> float:
        """Returns the Pillutla Score of the BLT.

        From Theorem 1 of https://arxiv.org/abs/2504.21413.
        Score < 1 ensures well-behaved inverse.

        Returns:
            sum(output_scale / buf_decay).
        """
        return float(torch.sum(self.output_scale / self.buf_decay))

    def _streaming_matrix_builder(self) -> StreamingMatrixBuilder:
        dtype = np.float64
        return StreamingMatrixBuilder(
            output_scale=self.output_scale.detach().numpy().astype(dtype),
            buf_decay=self.buf_decay.detach().numpy().astype(dtype),
        )

    def as_streaming_matrix(self) -> streaming_matrix.StreamingMatrix:
        """Returns a StreamingMatrix representing C."""
        return self._streaming_matrix_builder().build()

    def inverse_as_streaming_matrix(
        self,
    ) -> streaming_matrix.StreamingMatrix:
        """Returns a StreamingMatrix representing C^{-1}."""
        return self._streaming_matrix_builder().build_inverse()

    def __repr__(self) -> str:
        return (
            f"BufferedToeplitz(buf_decay={self.buf_decay.tolist()}, "
            f"output_scale={self.output_scale.tolist()})"
        )


def min_buf_decay_gap(buf_decay: torch.Tensor) -> torch.Tensor:
    """Returns the minimum gap between buf_decay parameters.

    Args:
        buf_decay: The buf_decay parameters.

    Returns:
        min_{i!=j} |theta[i] - theta[j]|
    """
    theta = torch.as_tensor(buf_decay)
    A = theta.unsqueeze(1) - theta.unsqueeze(0)
    A = A + torch.diag(torch.full_like(theta, float("inf")))
    return torch.min(torch.abs(A))


def _gt_zero_penalty(x: torch.Tensor) -> torch.Tensor:
    """Penalize values to enforce x > 0."""
    return -torch.log(x).sum()


def _lt_one_penalty(x: torch.Tensor) -> torch.Tensor:
    """Penalize values to enforce x < 1."""
    return -torch.log(1 - x).sum()


def _lt_penalty(x: torch.Tensor, upper_bound: float) -> torch.Tensor:
    """Penalize values to enforce x < upper_bound."""
    return -torch.log(upper_bound - x).sum()


def _lt_zero_penalty(x: torch.Tensor) -> torch.Tensor:
    """Penalize values to enforce x < 0."""
    return -torch.log(-x).sum()


def geometric_sum(
    a: torch.Tensor, r: torch.Tensor, num: float = float("inf")
) -> torch.Tensor:
    """Compute a + a*r + a*r^2 + ... + a*r^(num-1).

    Args:
        a: Scale factor(s).
        r: Common ratio(s), requires |r| < 1.
        num: Number of terms (inf for infinite series).

    Returns:
        The sum.
    """
    if math.isinf(num):
        return a / (1 - r)

    n = num
    # Choose between direct computation and Taylor approximation
    threshold = 1 - 1e-3  # Simplified threshold
    use_direct = r < threshold

    # Direct computation (safe when r is not near 1)
    safe_r = torch.where(use_direct, r, torch.zeros_like(r))
    direct = a * (1 - safe_r**n) / (1 - safe_r)

    # Taylor series near r = 1
    x0 = n - 1
    x1 = r - 1
    series = (1 / 6) * a * n * (x0 * x1**2 * (n - 2) + 3 * x0 * x1 + 6)

    return torch.where(use_direct, direct, series)


def sensitivity_squared(blt: BufferedToeplitz, n: float) -> torch.Tensor:
    """Compute sensitivity^2 for a BLT strategy matrix C.

    See Lemma 5.3 of https://arxiv.org/abs/2404.16706.

    Args:
        blt: The BLT representing C.
        n: Number of iterations (can be inf).

    Returns:
        Maximum column norm squared of C.
    """
    if blt._num_buffers == 0:
        return torch.tensor(1.0, dtype=torch.float64)

    if torch.any(blt.buf_decay > 1):
        return torch.tensor(float("inf"), dtype=torch.float64)

    omega = blt.output_scale
    theta = blt.buf_decay
    num = n - 1

    omega_pairs = omega.unsqueeze(0) * omega.unsqueeze(1)
    theta_pairs = theta.unsqueeze(0) * theta.unsqueeze(1)
    geo_pairs = geometric_sum(omega_pairs, theta_pairs, num=num)
    return 1.0 + geo_pairs.sum()


def max_error(inv_blt: BufferedToeplitz, n: float) -> torch.Tensor:
    """Max squared error for any iteration 0, ..., n-1.

    Args:
        inv_blt: BLT representing C^{-1}.
        n: Number of iterations.

    Returns:
        Max squared error.
    """
    return iteration_error(inv_blt, n - 1)


def iteration_error(inv_blt: BufferedToeplitz, i: float) -> torch.Tensor:
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
        return torch.tensor(float(i + 1), dtype=torch.float64)

    n = i + 1
    omega = inv_blt.output_scale
    theta = inv_blt.buf_decay

    # s1: sum of Gamma_j terms
    s1 = torch.tensor(0.0, dtype=torch.float64)
    for j in range(len(omega)):
        gj = _max_error_Gamma_j(omega[j], theta[j], n)
        s1 = s1 + gj

    # s2: sum of Gamma_jk cross terms
    s2 = torch.tensor(0.0, dtype=torch.float64)
    for j in range(len(omega)):
        for k in range(len(omega)):
            gjk = _max_error_Gamma_jk(omega[j], theta[j], omega[k], theta[k], n)
            s2 = s2 + gjk

    return n * (1 + 2 * s1 + s2)


def _max_error_Gamma_j(omega, theta, n):
    """Compute Gamma_j for max error."""
    if abs(float(theta) - 1.0) < 1e-6:
        # Taylor series near theta=1
        return omega * (n - 1) / 2
    return (omega / (1 - theta)) * (
        1 - geometric_sum(torch.ones_like(theta), theta, num=n) / n
    )


def _max_error_Gamma_jk(omega1, theta1, omega2, theta2, n):
    """Compute cross term Gamma_jk for max error."""
    if abs(float(theta1) - 1.0) < 1e-6 and abs(float(theta2) - 1.0) < 1e-6:
        # Both near 1
        return omega1 * omega2 * (2 * n**2 / 3 - n + 1 / 3) / n

    if abs(float(theta1) - 1.0) < 1e-6 or abs(float(theta2) - 1.0) < 1e-6:
        # One near 1 - use formula with geometric sum
        pass

    temp1 = omega1 * omega2 / ((1 - theta1) * (1 - theta2))
    gs1 = geometric_sum(torch.ones_like(theta1), theta1, num=n)
    gs2 = geometric_sum(torch.ones_like(theta2), theta2, num=n)
    gs12 = geometric_sum(torch.ones_like(theta1), theta1 * theta2, num=n)
    temp2 = (n - gs1 - gs2 + gs12) / n
    return temp1 * temp2


@dataclasses.dataclass(frozen=True)
class LossFn:
    """Encapsulates the loss to be optimized for a BLT.

    Attributes:
        error_for_inv: Error function taking C^{-1} BLT.
        sensitivity_squared: Sensitivity function taking C BLT.
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
        **kwargs,
    ) -> LossFn:
        """Construct LossFn for min-sep participation."""
        def mean_error_fn(inv_blt):
            return toeplitz.mean_error(noising_coef=inv_blt.toeplitz_coefs(n))

        def max_error_fn(inv_blt):
            return toeplitz.max_error(noising_coef=inv_blt.toeplitz_coefs(n))

        if error == "mean":
            error_fn = mean_error_fn
        elif error == "max":
            error_fn = max_error_fn
        else:
            raise ValueError(f"Unknown error={error}")

        def minsep_sens_sq(blt):
            return toeplitz.minsep_sensitivity_squared(
                strategy_coef=blt.toeplitz_coefs(n),
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
            **kwargs,
        )

    def loss(self, blt: BufferedToeplitz, skip_checks: bool = False) -> torch.Tensor:
        """Returns the loss (not including penalties).

        Args:
            blt: The BLT to evaluate.
            skip_checks: Skip validation.

        Returns:
            error * sensitivity_squared.
        """
        try:
            inv_blt = blt.inverse(skip_checks=skip_checks)
        except (RuntimeError, ValueError):
            return torch.tensor(float("inf"), dtype=torch.float64)
        error = self.error_for_inv(inv_blt)
        sens_sq = self.sensitivity_squared_fn(blt)
        return error * sens_sq


def blt_pair_from_theta_pair(
    theta: torch.Tensor, theta_hat: torch.Tensor
) -> tuple[BufferedToeplitz, BufferedToeplitz]:
    """Compute BLTs (C, C_inv) from theta and theta_hat.

    Implements Lemma 5.2 of https://arxiv.org/abs/2404.16706.

    Args:
        theta: Array of thetas for C.
        theta_hat: Array of thetas for C^{-1}.

    Returns:
        Tuple (C_blt, C_inv_blt).
    """
    theta = torch.as_tensor(theta, dtype=torch.float64)
    theta_hat = torch.as_tensor(theta_hat, dtype=torch.float64)

    def get_omega(th, th_hat):
        numerators = torch.prod(th.unsqueeze(1) - th_hat.unsqueeze(0), dim=1)
        A = th.unsqueeze(1) - th.unsqueeze(0)
        A = A + torch.diag(torch.ones_like(th))
        A[range(len(th)), range(len(th))] = 1.0
        denominators = torch.prod(A, dim=1)
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


def optimize(
    *,
    n: int,
    min_sep: int = 1,
    max_participations: int | None = 1,
    error: str = "max",
    min_buffers: int = 0,
    max_buffers: int = 10,
    rtol: float = 1.01,
) -> BufferedToeplitz:
    """Compute a good BLT with dynamically-chosen num_buffers.

    Internally increases num_buffers until improvement < rtol.

    Args:
        n: Number of iterations.
        min_sep: Minimum separation of participations.
        max_participations: Maximum participations.
        error: 'max' or 'mean'.
        min_buffers: Minimum buffers to try.
        max_buffers: Maximum buffers to try.
        rtol: Relative tolerance for improvement.

    Returns:
        An optimized BLT.
    """
    if max_buffers > 15:
        raise ValueError("max_buffers > 15 will likely cause numerical issues.")

    k = sensitivity.minsep_true_max_participations(
        n=n, min_sep=min_sep, max_participations=max_participations
    )

    if k == 1 and error == "max":
        loss_fn = LossFn.build_closed_form_single_participation(n=n)
    else:
        loss_fn = LossFn.build_min_sep(
            n=n,
            error=error,
            min_sep=min_sep,
            max_participations=max_participations,
        )

    prev_blt = get_init_blt(num_buffers=min_buffers)
    prev_loss = float(loss_fn.loss(prev_blt))

    for nbuf in range(min_buffers + 1, max_buffers + 1):
        try:
            blt = get_init_blt(num_buffers=nbuf)
            curr_loss = float(loss_fn.loss(blt))
        except (RuntimeError, ValueError) as err:
            logger.warning("Optimization failed for %d buffers: %s", nbuf, err)
            curr_loss = float("inf")
            blt = None

        if blt is not None and rtol * curr_loss < prev_loss:
            prev_blt, prev_loss = blt, curr_loss
        else:
            return prev_blt

    return prev_blt
