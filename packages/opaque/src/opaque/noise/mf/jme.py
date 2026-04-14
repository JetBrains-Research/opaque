"""JME (Joint Moment Estimation) — DP-Adam noise via MF.

Implements JME from Kalinin, Upadhyay, Lampert (NeurIPS 2025,
arXiv:2502.06597).  The key result (Theorem 3.2): two MF noise streams
(gradients and squared gradients) can share a single privacy budget when
noise scales are set correctly.  The second moment is "free".

User-facing API
---------------
:func:`mf_noise_jme` — drop-in replacement for :func:`mf_noise` that
internally runs **two** correlated-noise streams and returns the noisy
squared gradients on the state object.

Setup::

    noise_fn, state = mf_noise_jme(
        grad_template, strategy,
        stddev=noise_stddev, key=rng_key,
        beta2=0.999,           # Adam second-moment workload
        zeta=clip_state.sensitivity,
    )

Training loop (same signature as ``mf_noise``)::

    noisy_grads, state = noise_fn(clipped_grads, state)
    noisy_sq_grads = state.noisy_squared_grads     # ← the extra output

References:
    - JME: https://arxiv.org/abs/2502.06597
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import torch

from opaque.noise.types import NoiseState
from opaque.random import RngKey
from opaque.random import fold_in as rng_fold_in
from opaque.utils.pytree import tree_map

from ._engine import MFNoiseState
from .dispatcher import MfStrategy, mf_noise

__all__ = [
    "mf_noise_jme",
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

    Args:
        c1_sensitivity: Max column norm of C₁ (``strategy.sensitivity``).
        c2_sensitivity: Max column norm of C₂.
        zeta: Per-sample clipping bound (``clip_state.sensitivity``).
        d: Dimension (≥ 2 for neural nets, 1 for scalars).
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
    """Joint sensitivity for both moments (Theorem 3.2).

    ``s = 2ζ · ‖C₁‖_{1→2}``
    """
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
# State
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class JmeNoiseState(NoiseState):
    """Noise state for :func:`mf_noise_jme`.

    Carries the usual MF state plus the noisy squared gradients
    produced by the second noise stream.

    Attributes:
        noisy_squared_grads: Noisy element-wise squared gradients from
            the most recent ``noise_fn`` call.  ``None`` before the first
            call.  Feed this to Adam's second-moment EMA.
    """

    _first_state: MFNoiseState
    _second_state: MFNoiseState
    noisy_squared_grads: Any = None

    @property
    def _step_counter(self) -> int:  # type: ignore[override]
        return self._first_state._step_counter

    @property
    def _rng_key(self) -> RngKey:  # type: ignore[override]
        return self._first_state._rng_key


# ---------------------------------------------------------------------------
# Factory: second-moment strategy derivation
# ---------------------------------------------------------------------------


def _derive_second_strategy(
    strategy: MfStrategy,
    beta2: float,
) -> MfStrategy:
    """Build a second-moment strategy from the first-moment strategy.

    Same mechanism type and participation parameters, different workload
    momentum (β₂ instead of β₁).
    """
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


def mf_noise_jme(
    grad_template: Any,
    strategy: MfStrategy,
    *,
    stddev: float,
    key: RngKey,
    zeta: float,
    beta2: float = 0.999,
    second_moment_strategy: MfStrategy | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[[Any, JmeNoiseState], tuple[Any, JmeNoiseState]],
    JmeNoiseState,
]:
    """Create a JME noise function for DP-Adam.

    Drop-in shape for :func:`mf_noise`: returns ``(noise_fn, state)``
    where ``noise_fn(grads, state) -> (noisy_grads, state)``.  The
    noisy *squared* gradients are available on ``state.noisy_squared_grads``
    after each call.

    Internally creates two ``mf_noise`` streams:

    - **First moment** (gradients): uses ``strategy`` as-is.
    - **Second moment** (squared gradients): auto-derived from
      ``strategy`` with ``momentum=beta2``, or pass
      ``second_moment_strategy`` explicitly.

    Noise scales are set by JME (Theorem 3.2, arXiv:2502.06597) so that
    the second moment comes at zero additional privacy cost.

    Args:
        grad_template: Pytree with same structure/shapes as gradients.
        strategy: MF strategy for the first moment (gradient stream).
            The ``momentum`` used when creating this strategy should match
            Adam's ``beta1``.
        stddev: Noise stddev for the first moment stream, typically
            ``noise_multiplier * jme_joint_sensitivity(strategy.sensitivity, zeta)``.
        key: RNG key for deterministic noise.
        zeta: Per-sample clipping bound (``clip_state.sensitivity``).
        beta2: Adam's second-moment decay.  Used to auto-derive the
            second-moment strategy when ``second_moment_strategy`` is None.
        second_moment_strategy: Explicit strategy for the second moment.
            If None, derived from ``strategy`` with ``momentum=beta2``.
        dtype: Optional dtype for intermediate noise computation.

    Returns:
        ``(noise_fn, state)`` — same shape as :func:`mf_noise`.

    Example::

        strategy = blt_strategy(n_steps=1000, ..., momentum=0.9)
        joint_s = jme_joint_sensitivity(strategy.sensitivity, zeta)
        noise_stddev = noise_multiplier * joint_s

        noise_fn, state = mf_noise_jme(
            grad_template, strategy,
            stddev=noise_stddev, key=rng_key,
            zeta=zeta, beta2=0.999,
        )

        # Training loop — same call signature as mf_noise:
        noisy_grads, state = noise_fn(clipped_grads, state)
        noisy_sq = state.noisy_squared_grads   # for Adam's v_t
    """
    # Second-moment strategy
    strat_v = second_moment_strategy
    if strat_v is None:
        strat_v = _derive_second_strategy(strategy, beta2)

    # JME noise scaling
    lam = jme_lambda(strategy.sensitivity, strat_v.sensitivity, zeta)
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
        noisy_squared_grads=None,
    )

    def noise_fn(
        clipped_grads: Any,
        st: JmeNoiseState,
    ) -> tuple[Any, JmeNoiseState]:
        # First moment: noise on gradients
        noisy_grads, new_first = first_fn(clipped_grads, st._first_state)

        # Second moment: noise on element-wise squared gradients
        sq_grads = tree_map(lambda g: g * g, clipped_grads)
        noisy_sq, new_second = second_fn(sq_grads, st._second_state)

        new_state = JmeNoiseState(
            _first_state=new_first,
            _second_state=new_second,
            noisy_squared_grads=noisy_sq,
        )
        return noisy_grads, new_state

    return noise_fn, init_state
