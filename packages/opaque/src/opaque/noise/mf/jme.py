"""JME (Joint Moment Estimation) strategy for DP-Adam/AdaGrad.

Implements the JME algorithm from Kalinin, Upadhyay, Lampert (2025)
"Continual Release Moment Estimation with Differential Privacy"
(arXiv:2502.06597, NeurIPS 2025).

JME wraps two inner MF strategies to produce correlated noise for both
the first moment (gradients) and second moment (squared gradients)
simultaneously, with the second moment coming at zero additional privacy
cost via joint sensitivity analysis.

This enables DP-Adam and DP-AdaGrad with MF correlated noise for the
first time — previously only SGD with Polyak momentum was compatible
with MF noise.

Use ``jme_noise(jme_strategy(...), ...)`` to create the noise functions.

References:
    - JME: https://arxiv.org/abs/2502.06597
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from opaque.noise.types import NoiseState
from opaque.random import RngKey
from opaque.random import fold_in as rng_fold_in

from ._engine import (
    MFNoiseState,
    _matrix_factorization_noise,
)
from ._streaming_matrix import identity
from .band_mf import BandMfStrategy
from .bisr import BisrStrategy
from .blt import BltStrategy
from .identity import IdentityStrategy
from .lambda_cgd import LambdaCgdStrategy, _make_lambda_cgd_noise

__all__ = ["JmeStrategy", "jme_strategy", "jme_noise"]


# ---------------------------------------------------------------------------
# Constants from the paper (Section 3, Algorithm 1)
# ---------------------------------------------------------------------------

C_D_1 = 8.0 / (11.0 + 5.0 * math.sqrt(5.0))  # c_d for d=1
C_D_GE2 = 2.0  # c_d for d >= 2


def _c_d(d: int) -> float:
    """JME dimension-dependent constant c_d (Algorithm 1 of the paper)."""
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    return C_D_1 if d == 1 else C_D_GE2


def _max_column_norm(strategy: Any) -> float:
    """Extract max column norm ‖C‖_{1→2} from a strategy.

    For single-participation, this equals the strategy's `sensitivity`.
    For strategies built with max_participations > 1, the max column norm
    is the single-participation sensitivity (max column L2 norm).
    """
    return strategy.sensitivity


def _jme_lambda(
    c1_norm: float,
    c2_norm: float,
    zeta: float,
    d: int,
) -> float:
    """Compute optimal JME scaling parameter λ.

    λ = ‖C₁‖²_{1→2} / (c_d · ζ² · ‖C₂‖²_{1→2})

    With λ set to this value, estimating the second moment is "free":
    the joint sensitivity equals the first-moment-only sensitivity.
    """
    cd = _c_d(d)
    denom = cd * zeta * zeta * c2_norm * c2_norm
    if denom <= 0:
        raise ValueError(
            f"Invalid parameters: c_d={cd}, zeta={zeta}, c2_norm={c2_norm}"
        )
    return (c1_norm * c1_norm) / denom


def _jme_joint_sensitivity(c1_norm: float, zeta: float) -> float:
    """Joint sensitivity for JME: s = 2ζ · ‖C₁‖_{1→2} (Theorem 3.2)."""
    return 2.0 * zeta * c1_norm


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JmeNoiseState(NoiseState):
    """State for JME dual-stream noise (first + second moment).

    Attributes:
        _first_moment_state: Noise state for the gradient (first moment) stream.
        _second_moment_state: Noise state for the squared-gradient (second moment) stream.
    """

    _first_moment_state: MFNoiseState
    _second_moment_state: MFNoiseState


# ---------------------------------------------------------------------------
# Strategy dataclass and factory
# ---------------------------------------------------------------------------

_InnerStrategy = (
    BandMfStrategy | BltStrategy | LambdaCgdStrategy | BisrStrategy | IdentityStrategy
)


@dataclass(frozen=True, slots=True)
class JmeStrategy:
    """Joint Moment Estimation strategy for DP-Adam/AdaGrad.

    Wraps two inner MF strategies (for first and second moments) and
    computes the joint sensitivity that makes second moment estimation
    "free" from a privacy perspective (Theorem 3.2 of arXiv:2502.06597).

    The ``sensitivity`` field holds the **joint** sensitivity
    ``s = 2ζ · ‖C₁‖_{1→2}``, which is used for noise calibration.
    Privacy accounting uses the same mechanism as the first-moment-only
    strategy (the whole point of JME).
    """

    sensitivity: float
    coefficients: tuple[float, ...]
    gram_matrix: tuple[float, ...] | None = None

    _first_moment_strategy: _InnerStrategy | None = None
    _second_moment_strategy: _InnerStrategy | None = None
    _lambda_jme: float = 1.0
    _zeta: float = 1.0
    _d: int = 2


def jme_strategy(
    first_moment_strategy: _InnerStrategy,
    second_moment_strategy: _InnerStrategy | None = None,
    *,
    zeta: float,
    d: int = 2,
) -> JmeStrategy:
    """Create a JME strategy for DP-Adam/AdaGrad with MF correlated noise.

    JME (Joint Moment Estimation) wraps any existing MF strategy to produce
    two correlated noise streams: one for gradients (first moment) and one
    for squared gradients (second moment). The second moment comes at
    **zero additional privacy cost** via joint sensitivity analysis.

    Args:
        first_moment_strategy: MF strategy for the gradient stream. Any
            existing strategy works: ``band_mf_strategy()``,
            ``blt_strategy()``, ``lambda_cgd_strategy()``,
            ``bisr_strategy()``, or ``identity_strategy()``.
        second_moment_strategy: MF strategy for the squared-gradient stream.
            If ``None`` (default), reuses ``first_moment_strategy`` for both.
        zeta: Per-sample sensitivity bound, typically
            ``clip_state.sensitivity`` (= clipping_norm / normalize_by).
        d: Parameter dimension for the c_d constant. For neural networks,
            ``d >= 2`` (default), so ``c_d = 2``. Only set ``d=1`` for
            scalar optimization.

    Returns:
        A :class:`JmeStrategy` that can be passed to :func:`jme_noise`.

    Example::

        from opaque.noise.mf import band_mf_strategy, jme_strategy, jme_noise

        inner = band_mf_strategy(n_steps=1000, bands=8, momentum=0.0)
        strategy = jme_strategy(inner, zeta=clip_state.sensitivity)
        noise_fn, state = jme_noise(grad_template, strategy, stddev=stddev, key=rng_key)

    References:
        - Kalinin, Upadhyay, Lampert (2025) "Continual Release Moment
          Estimation with Differential Privacy" https://arxiv.org/abs/2502.06597
    """
    if zeta <= 0:
        raise ValueError(f"zeta must be positive, got {zeta}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")

    if second_moment_strategy is None:
        second_moment_strategy = first_moment_strategy

    c1_norm = _max_column_norm(first_moment_strategy)
    c2_norm = _max_column_norm(second_moment_strategy)

    lambda_jme = _jme_lambda(c1_norm, c2_norm, zeta, d)
    joint_sensitivity = _jme_joint_sensitivity(c1_norm, zeta)

    return JmeStrategy(
        sensitivity=joint_sensitivity,
        coefficients=first_moment_strategy.coefficients,
        gram_matrix=first_moment_strategy.gram_matrix,
        _first_moment_strategy=first_moment_strategy,
        _second_moment_strategy=second_moment_strategy,
        _lambda_jme=lambda_jme,
        _zeta=zeta,
        _d=d,
    )


# ---------------------------------------------------------------------------
# Noise builder
# ---------------------------------------------------------------------------


def _dispatch_inner_noise(
    grad_template: Any,
    strategy: _InnerStrategy,
    *,
    stddev: float,
    key: RngKey,
    dtype: torch.dtype | None = None,
) -> tuple[Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]], MFNoiseState]:
    """Dispatch to the correct noise builder for an inner strategy."""
    match strategy:
        case IdentityStrategy():
            return _matrix_factorization_noise(
                grad_template, identity(), stddev=stddev, key=key, dtype=dtype
            )
        case LambdaCgdStrategy():
            return _make_lambda_cgd_noise(
                grad_template, strategy, stddev=stddev, key=key, dtype=dtype
            )
        case BandMfStrategy() | BltStrategy() | BisrStrategy():
            if strategy._streaming_matrix is None:
                raise ValueError(
                    "Inner strategy must have a _streaming_matrix for noise generation."
                )
            return _matrix_factorization_noise(
                grad_template,
                strategy._streaming_matrix,
                stddev=stddev,
                key=key,
                dtype=dtype,
            )
        case _:
            raise TypeError(
                f"Unknown inner strategy type: {type(strategy).__name__}"
            )


def jme_noise(
    grad_template: Any,
    strategy: JmeStrategy,
    *,
    stddev: float,
    key: RngKey,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[
        [Any, Any, JmeNoiseState],
        tuple[tuple[Any, Any], JmeNoiseState],
    ],
    JmeNoiseState,
]:
    """Create dual noise functions for JME (first + second moment).

    Returns a ``(noise_fn, state)`` pair where ``noise_fn`` adds
    correlated noise to **both** the gradient and squared-gradient
    streams simultaneously.

    Args:
        grad_template: Pytree with same structure/shapes as gradients.
        strategy: A :class:`JmeStrategy` from :func:`jme_strategy`.
        stddev: Standard deviation for the base noise. Typically
            ``noise_multiplier * clip_state.sensitivity``.
        key: Explicit RNG key for deterministic randomness.
        dtype: Optional dtype for intermediate noise computation.

    Returns:
        A tuple ``(noise_fn, state)`` where::

            noise_fn(clipped_grads, squared_grads, state)
                -> ((noisy_grads, noisy_sq_grads), new_state)

        ``noisy_grads`` feeds Adam's first moment, ``noisy_sq_grads``
        feeds Adam's second moment.

    Example::

        noise_fn, noise_state = jme_noise(grad_template, strategy,
                                          stddev=noise_stddev, key=rng_key)

        # In training loop:
        sq_grads = tree_map(lambda g: g * g, clipped_grads)
        (noisy_grads, noisy_sq_grads), noise_state = noise_fn(
            clipped_grads, sq_grads, noise_state
        )
    """
    if not isinstance(strategy, JmeStrategy):
        raise TypeError(
            f"jme_noise() requires a JmeStrategy, got {type(strategy).__name__}. "
            f"Use mf_noise() for single-stream strategies."
        )

    first_strat = strategy._first_moment_strategy
    second_strat = strategy._second_moment_strategy
    if first_strat is None or second_strat is None:
        raise ValueError("JmeStrategy must have both inner strategies set.")

    lambda_jme = strategy._lambda_jme

    # Second moment noise is scaled by λ^{-1/2}
    second_moment_scale = 1.0 / math.sqrt(lambda_jme)
    stddev_second = stddev * second_moment_scale

    # Independent RNG keys for the two streams
    key_first = rng_fold_in(key, 0)
    key_second = rng_fold_in(key, 1)

    # Build inner noise functions
    first_noise_fn, first_state = _dispatch_inner_noise(
        grad_template, first_strat, stddev=stddev, key=key_first, dtype=dtype
    )
    second_noise_fn, second_state = _dispatch_inner_noise(
        grad_template, second_strat, stddev=stddev_second, key=key_second, dtype=dtype
    )

    state = JmeNoiseState(
        _first_moment_state=first_state,
        _second_moment_state=second_state,
    )

    def noise_fn(
        clipped_grads: Any,
        squared_grads: Any,
        st: JmeNoiseState,
    ) -> tuple[tuple[Any, Any], JmeNoiseState]:
        noisy_grads, new_first_state = first_noise_fn(
            clipped_grads, st._first_moment_state
        )
        noisy_sq_grads, new_second_state = second_noise_fn(
            squared_grads, st._second_moment_state
        )
        new_state = JmeNoiseState(
            _first_moment_state=new_first_state,
            _second_moment_state=new_second_state,
        )
        return (noisy_grads, noisy_sq_grads), new_state

    return noise_fn, state
