"""Joint Gaussian noise calibration for paired first + second-moment release.

Used by :func:`opaque.dpftrl.noise.dispatcher.mf_noise` when a
``SecondMomentClippingOutput`` flows in.  See Kalinin, Upadhyay,
Lampert, "Continual Release Moment Estimation with Differential
Privacy", arXiv:2502.06597.
"""

from __future__ import annotations

import math


DEFAULT_SECOND_MOMENT_OVERHEAD = math.sqrt(3.0 / 2.0)
"""Default first-stream noise overhead for private second moments.

The add/remove-DP d >= 2 value from the joint first+second moment
analysis.  Multiplies the first-moment sensitivity when the noise
mechanism also releases a private element-wise squared-gradient stream.
"""


def resolve_second_moment_overhead(second_moment: bool | float) -> float:
    """Resolve ``second_moment`` to a first-moment noise overhead.

    ``True`` selects the default paper value ``sqrt(3/2)`` for d >= 2.
    A float supplies the overhead directly and must be greater than 1.
    """
    if isinstance(second_moment, bool):
        if not second_moment:
            raise ValueError("second_moment=False has no overhead to resolve")
        return DEFAULT_SECOND_MOMENT_OVERHEAD

    overhead = float(second_moment)
    if overhead <= 1.0:
        raise ValueError(
            f"second_moment overhead must be greater than 1.0, got {second_moment}"
        )
    return overhead


def second_moment_joint_sensitivity(
    c1_max_column_norm: float,
    sensitivity: float,
    *,
    first_moment_overhead: float = DEFAULT_SECOND_MOMENT_OVERHEAD,
) -> float:
    """Sensitivity of the first stream when releasing a second moment.

    ``sensitivity`` is the clipped-gradient sensitivity before applying
    the mechanism strategy (for averaged gradients this is typically
    ``clipping_norm / batch_size``).  ``c1_max_column_norm`` is the
    first strategy's max column norm.
    """
    if c1_max_column_norm <= 0:
        raise ValueError(
            f"c1_max_column_norm must be positive, got {c1_max_column_norm}"
        )
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be positive, got {sensitivity}")
    if first_moment_overhead <= 1.0:
        raise ValueError(
            "first_moment_overhead must be greater than 1.0, "
            f"got {first_moment_overhead}"
        )
    return sensitivity * c1_max_column_norm * first_moment_overhead


def second_moment_noise_scale(
    c1_max_column_norm: float,
    c2_max_column_norm: float,
    first_max_norm: float,
    squared_max_norm: float,
    *,
    first_moment_overhead: float = DEFAULT_SECOND_MOMENT_OVERHEAD,
) -> float:
    """Scale from first-stream stddev to second-stream stddev.

    For overhead ``rho`` and per-record sensitivities ``Δ_first`` /
    ``Δ_second`` on the two streams, the joint Gaussian allocation gives::

        σ_second = σ_first · (c2 · Δ_second) / (c1 · Δ_first · sqrt(rho² − 1))

    so the scale returned here is the right-hand factor.  Pass
    ``squared_max_norm`` as the per-record bound on the squared stream:
    for averaged ``Σᵢ gᵢ² / n`` it is ``C² / n``.
    """
    if c1_max_column_norm <= 0:
        raise ValueError(
            f"c1_max_column_norm must be positive, got {c1_max_column_norm}"
        )
    if c2_max_column_norm <= 0:
        raise ValueError(
            f"c2_max_column_norm must be positive, got {c2_max_column_norm}"
        )
    if first_max_norm <= 0:
        raise ValueError(f"first_max_norm must be positive, got {first_max_norm}")
    if squared_max_norm <= 0:
        raise ValueError(f"squared_max_norm must be positive, got {squared_max_norm}")
    if first_moment_overhead <= 1.0:
        raise ValueError(
            "first_moment_overhead must be greater than 1.0, "
            f"got {first_moment_overhead}"
        )
    return (
        squared_max_norm
        * c2_max_column_norm
        / (
            first_max_norm
            * c1_max_column_norm
            * math.sqrt(first_moment_overhead**2 - 1.0)
        )
    )


def second_moment_stddevs(
    noise_multiplier: float,
    first_max_norm: float,
    squared_max_norm: float,
    *,
    c1_max_column_norm: float = 1.0,
    c2_max_column_norm: float = 1.0,
    first_moment_overhead: float = DEFAULT_SECOND_MOMENT_OVERHEAD,
) -> tuple[float, float]:
    """Return ``(first_stddev, second_stddev)`` for private moments.

    ``first_max_norm`` is the per-record contribution bound on the
    first stream (``C / n`` for averaged clipped grads).
    ``squared_max_norm`` is the analogous bound on the second stream;
    for averaged ``Σᵢ gᵢ² / n`` it is ``C² / n``.
    """
    if noise_multiplier < 0:
        raise ValueError(
            f"noise_multiplier must be non-negative, got {noise_multiplier}"
        )
    first_sensitivity = second_moment_joint_sensitivity(
        c1_max_column_norm,
        first_max_norm,
        first_moment_overhead=first_moment_overhead,
    )
    first_stddev = noise_multiplier * first_sensitivity
    second_stddev = first_stddev * second_moment_noise_scale(
        c1_max_column_norm,
        c2_max_column_norm,
        first_max_norm,
        squared_max_norm,
        first_moment_overhead=first_moment_overhead,
    )
    return first_stddev, second_stddev
