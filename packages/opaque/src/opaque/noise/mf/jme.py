"""JME (Joint Moment Estimation) — DP-Adam noise via MF.

Implements JME from Kalinin, Upadhyay, Lampert (NeurIPS 2025,
arXiv:2502.06597).  The key result (Theorem 3.2): two MF noise streams
(gradients and squared gradients) can share a single privacy budget when
noise scales are set correctly.  The second moment is "free".

User-facing API
---------------
:func:`jme_noise` — creates a noise function for DP-Adam.

Setup::

    noise_fn, state = jme_noise(
        grad_template, strategy,
        noise_multiplier=sigma, key=rng_key,
        zeta=clip_state.sensitivity,
        beta2=0.999,
    )

Training loop::

    (noisy_grads, noisy_sq), state = noise_fn(clipped_grads, state)
    updates, opt_state = optimizer.update(
        noisy_grads, opt_state, noisy_squared_grads=noisy_sq,
    )

References:
    - JME: https://arxiv.org/abs/2502.06597
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

from opaque.noise.types import NoiseState
from opaque.random import RngKey
from opaque.random import fold_in as rng_fold_in
from opaque.utils.pytree import tree_map

from ._engine import MFNoiseState
from .dispatcher import MfStrategy, mf_noise

__all__ = [
    "jme_noise",
    "JmeNoiseOutput",
    "JmeNoiseState",
    "jme_lambda",
    "jme_joint_sensitivity",
    "jme_second_moment_stddev",
]


# ---------------------------------------------------------------------------
# Constants (Section 3, Algorithm 1)
# ---------------------------------------------------------------------------

_C_D_1 = 8.0 / (11.0 + 5.0 * math.sqrt(5.0))  # ≈ 0.339, for d=1
_C_D_GE2 = 2.0  # for d ≥ 2


def _c_d(d: int) -> float:
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    return _C_D_1 if d == 1 else _C_D_GE2


# ---------------------------------------------------------------------------
# Pure math helpers (also usable standalone)
# ---------------------------------------------------------------------------


def jme_lambda(
    c1_sensitivity: float,
    c2_sensitivity: float,
    zeta: float,
    d: int = 2,
) -> float:
    """Optimal JME scaling parameter λ (Algorithm 1).

    ``λ = ‖C₁‖²_{1→2} / (c_d · ζ² · ‖C₂‖²_{1→2})``
    """
    if c1_sensitivity <= 0:
        raise ValueError(f"c1_sensitivity must be positive, got {c1_sensitivity}")
    if c2_sensitivity <= 0:
        raise ValueError(f"c2_sensitivity must be positive, got {c2_sensitivity}")
    if zeta <= 0:
        raise ValueError(f"zeta must be positive, got {zeta}")
    cd = _c_d(d)
    return (c1_sensitivity**2) / (cd * zeta**2 * c2_sensitivity**2)


def jme_joint_sensitivity(c1_sensitivity: float, zeta: float) -> float:
    """Joint sensitivity for both moments: ``s = 2ζ · ‖C₁‖_{1→2}`` (Theorem 3.2)."""
    if c1_sensitivity <= 0:
        raise ValueError(f"c1_sensitivity must be positive, got {c1_sensitivity}")
    if zeta <= 0:
        raise ValueError(f"zeta must be positive, got {zeta}")
    return 2.0 * zeta * c1_sensitivity


def jme_second_moment_stddev(
    first_moment_stddev: float,
    lambda_jme: float,
) -> float:
    """Noise stddev for the second moment stream: ``stddev / √λ``."""
    if lambda_jme <= 0:
        raise ValueError(f"lambda_jme must be positive, got {lambda_jme}")
    return first_moment_stddev / math.sqrt(lambda_jme)


# ---------------------------------------------------------------------------
# Output and state
# ---------------------------------------------------------------------------


class JmeNoiseOutput(NamedTuple):
    """Per-step output of a :func:`jme_noise` noise function.

    Attributes:
        noisy_grads: Noisy clipped gradients (first moment).
        noisy_squared_grads: Noisy element-wise squared gradients (second moment).
    """

    noisy_grads: Any
    noisy_squared_grads: Any


@dataclasses.dataclass(frozen=True)
class JmeNoiseState(NoiseState):
    """Internal state for :func:`jme_noise` (two MF streams)."""

    _first_state: MFNoiseState
    _second_state: MFNoiseState

    @property
    def _step_counter(self) -> int:  # type: ignore[override]
        return self._first_state._step_counter

    @property
    def _rng_key(self) -> RngKey:  # type: ignore[override]
        return self._first_state._rng_key


# ---------------------------------------------------------------------------
# Second-strategy auto-derivation
# ---------------------------------------------------------------------------


def _derive_second_strategy(
    strategy: MfStrategy,
    beta2: float,
) -> MfStrategy:
    """Same mechanism, ``momentum=beta2`` workload."""
    from .band_mf import BandMfStrategy, band_mf_strategy
    from .bisr import BisrStrategy, bisr_strategy
    from .blt import BltStrategy, blt_strategy
    from .identity import IdentityStrategy, identity_strategy
    from .lambda_cgd import LambdaCgdStrategy, lambda_cgd_strategy

    match strategy:
        case BandMfStrategy():
            return band_mf_strategy(
                n_steps=strategy._n_steps,
                bands=strategy._bands,
                momentum=beta2,
            )
        case BltStrategy():
            return blt_strategy(
                n_steps=strategy._n_steps,
                min_sep=strategy._min_sep,
                max_participations=strategy._max_participations,
                max_buffers=strategy._max_buffers,
                momentum=beta2,
            )
        case LambdaCgdStrategy():
            return lambda_cgd_strategy(
                lambda_=strategy._lambda,
                n_steps=strategy._n_steps,
                min_sep=strategy._min_sep
                if hasattr(strategy, "_min_sep")
                else strategy._n_steps,
                max_participations=strategy._max_participations
                if hasattr(strategy, "_max_participations")
                else 1,
            )
        case BisrStrategy():
            return bisr_strategy(
                bandwidth=strategy._bandwidth
                if hasattr(strategy, "_bandwidth")
                else 4,
                n_steps=strategy._n_steps
                if hasattr(strategy, "_n_steps")
                else 1,
                min_sep=strategy._min_sep
                if hasattr(strategy, "_min_sep")
                else 1,
                max_participations=strategy._max_participations
                if hasattr(strategy, "_max_participations")
                else 1,
                momentum=beta2,
            )
        case IdentityStrategy():
            return identity_strategy()
        case _:
            return identity_strategy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def jme_noise(
    grad_template: Any,
    strategy: MfStrategy,
    *,
    noise_multiplier: float,
    key: RngKey,
    zeta: float,
    beta2: float = 0.999,
    second_moment_strategy: MfStrategy | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[[Any, JmeNoiseState], tuple[JmeNoiseOutput, JmeNoiseState]],
    JmeNoiseState,
]:
    """Create a JME noise function for DP-Adam.

    Returns ``(noise_fn, state)`` where each call to ``noise_fn``
    produces a :class:`JmeNoiseOutput` containing both noisy gradients
    and noisy squared gradients.

    Internally:

    1. Computes JME joint sensitivity and both noise stddevs.
    2. Auto-derives the second-moment strategy (same mechanism,
       ``momentum=beta2``).
    3. Creates two ``mf_noise`` streams with independent RNG keys.
    4. ``noise_fn`` computes ``g²`` and runs both streams per call.

    Args:
        grad_template: Pytree matching gradient shapes.
        strategy: MF strategy for the first moment (gradient stream).
            Build with ``momentum=beta1`` (Adam's β₁).
        noise_multiplier: The calibrated noise multiplier σ — same value
            used in privacy accounting.  JME joint sensitivity and stddev
            are computed internally.
        key: RNG key.
        zeta: Per-sample clipping bound (``clip_state.sensitivity``).
        beta2: Adam's β₂.  Used to auto-derive the second-moment
            strategy.
        second_moment_strategy: Explicit override for the second-moment
            strategy.  If ``None``, derived from ``strategy``.
        dtype: Optional dtype for intermediate noise.

    Returns:
        ``(noise_fn, state)`` where ``noise_fn`` returns a
        :class:`JmeNoiseOutput` that unpacks as a tuple::

            (noisy_grads, noisy_sq), state = noise_fn(clipped_grads, state)
    """
    strat_v = second_moment_strategy
    if strat_v is None:
        strat_v = _derive_second_strategy(strategy, beta2)

    # JME calibration — uses max column norms ‖C‖_{1→2}, not
    # multi-participation sensitivity (Theorem 3.2, Algorithm 1).
    c1_norm = strategy._max_column_norm
    c2_norm = strat_v._max_column_norm
    joint_sens = jme_joint_sensitivity(c1_norm, zeta)
    stddev = noise_multiplier * joint_sens
    lam = jme_lambda(c1_norm, c2_norm, zeta)
    stddev_v = jme_second_moment_stddev(stddev, lam)

    # Two independent noise streams
    key_m = rng_fold_in(key, 0)
    key_v = rng_fold_in(key, 1)

    first_fn, first_state = mf_noise(
        grad_template, strategy, stddev=stddev, key=key_m, dtype=dtype,
    )
    second_fn, second_state = mf_noise(
        grad_template, strat_v, stddev=stddev_v, key=key_v, dtype=dtype,
    )

    init_state = JmeNoiseState(
        _first_state=first_state,
        _second_state=second_state,
    )

    def noise_fn(
        clipped_grads: Any,
        st: JmeNoiseState,
    ) -> tuple[JmeNoiseOutput, JmeNoiseState]:
        noisy_grads, new_first = first_fn(clipped_grads, st._first_state)

        sq_grads = tree_map(lambda g: g * g, clipped_grads)
        noisy_sq, new_second = second_fn(sq_grads, st._second_state)

        output = JmeNoiseOutput(
            noisy_grads=noisy_grads,
            noisy_squared_grads=noisy_sq,
        )
        new_state = JmeNoiseState(
            _first_state=new_first,
            _second_state=new_second,
        )
        return output, new_state

    return noise_fn, init_state
