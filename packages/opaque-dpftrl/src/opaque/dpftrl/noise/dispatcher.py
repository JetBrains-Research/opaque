"""MF strategy types and unified noise dispatcher.

Each mechanism file defines its own strategy dataclass and factory:
- ``identity.py``: :class:`IdentityStrategy`, :func:`identity_strategy`
- ``band_mf.py``: :class:`BandMfStrategy`, :func:`band_mf_strategy`
- ``blt.py``: :class:`BltStrategy`, :func:`blt_strategy`
- ``lambda_cgd.py``: :class:`LambdaCgdStrategy`, :func:`lambda_cgd_strategy`
- ``bisr.py``: :class:`BisrStrategy`, :func:`bisr_strategy`
- ``bsr.py``: :class:`BsrStrategy`, :func:`bsr_strategy`

The :func:`mf_noise` function dispatches on the strategy type to create
the appropriate noise mechanism.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch

from opaque.bounded import BoundedPytree, NoisyPytree
from opaque.clipping.per_group import PerGroup
from opaque.core.noise import (
    DEFAULT_SECOND_MOMENT_OVERHEAD,
    SecondMomentNoiseOutput,
    resolve_second_moment_overhead,
    second_moment_joint_sensitivity,
    second_moment_noise_scale,
    second_moment_stddevs,
)
from opaque.core.pytree import tree_map
from opaque.random import RngKey
from opaque.random import fold_in as rng_fold_in

from .band_mf import BandMfStrategy, band_mf_strategy
from .bisr import BisrStrategy, bisr_strategy
from .bsr import BsrStrategy, bsr_strategy
from .blt import BltStrategy, blt_strategy
from .identity import IdentityStrategy, identity_strategy
from .lambda_cgd import (
    LambdaCgdStrategy,
    _make_lambda_cgd_noise,
    lambda_cgd_strategy,
)
from ._engine import (
    MFNoiseState,
    _matrix_factorization_noise,
)
from ._streaming_matrix import (
    identity,
)
from .second_moment import SecondMomentMFNoiseState

MfStrategy = (
    BandMfStrategy
    | BltStrategy
    | LambdaCgdStrategy
    | BisrStrategy
    | BsrStrategy
    | IdentityStrategy
)


def mf_noise(
    grad_template: Any,
    strategy: MfStrategy,
    *,
    noise_multiplier: float,
    key: RngKey,
    dtype: torch.dtype | None = None,
    second_moment: bool | float = False,
    second_moment_strategy: MfStrategy | None = None,
) -> tuple[
    Callable[..., tuple[Any, MFNoiseState | SecondMomentMFNoiseState]],
    MFNoiseState | SecondMomentMFNoiseState,
]:
    """Create a correlated noise mechanism for the given MF strategy.

    Dispatches on the strategy type:
    - :class:`LambdaCgdStrategy`: PRNG replay noise (zero extra memory).
    - :class:`BandMfStrategy`, :class:`BltStrategy`, :class:`BisrStrategy`,
      :class:`BsrStrategy`:
      StreamingMatrix-based noise.

    Args:
        grad_template: Pytree with same structure/shapes as gradients.
        strategy: MF strategy from one of the factory functions.
        noise_multiplier: Gaussian noise multiplier. The clipped-gradient
            contribution bound is read from each ``BoundedPytree`` input.
        key: Explicit RNG key for deterministic randomness.
        dtype: Optional dtype for intermediate noise computation.
        second_moment: ``False`` for the regular single stream.  ``True``
            enables the default first-stream overhead ``sqrt(3/2)``; a float
            supplies the overhead directly and must be greater than 1.
        second_moment_strategy: Explicit strategy for the squared-gradient
            stream.  Required when ``second_moment`` is enabled.

    Returns:
        A tuple ``(noise_fn, state)`` for the training loop.
    """
    resolved_noise_multiplier = _resolve_noise_multiplier(noise_multiplier)
    if not isinstance(key, RngKey):
        raise TypeError(f"key must be RngKey, got {type(key)}")

    if second_moment is not False:
        if second_moment_strategy is None:
            raise ValueError(
                "second_moment_strategy is required when second_moment is enabled. "
                "Build it explicitly for the squared-gradient workload."
            )
        return _make_second_moment_mf_noise(
            grad_template,
            strategy,
            second_moment_strategy,
            noise_multiplier=resolved_noise_multiplier,
            key=key,
            dtype=dtype,
            second_moment=second_moment,
        )

    raw_noise_fn, raw_state = _make_raw_mf_noise(
        grad_template,
        strategy,
        key=key,
        dtype=dtype,
    )

    def noise_fn(
        clipped_grads: Any,
        st: MFNoiseState,
    ) -> tuple[NoisyPytree, MFNoiseState]:
        bounded_grads = _expect_bounded(clipped_grads, op="mf_noise")
        bound = _validate_constant_bound(
            bounded_grads, st._first_bound, op="mf_noise"
        )
        base_stddev = resolved_noise_multiplier * bound
        noisy_tree, new_state = raw_noise_fn(
            bounded_grads.pytree,
            st,
            stddev=base_stddev,
        )
        return (
            NoisyPytree(
                pytree=noisy_tree,
                bound=bounded_grads.bound,
                noise_stddev=base_stddev,
            ),
            replace(new_state, _first_bound=bound),
        )

    return noise_fn, raw_state


def _make_raw_mf_noise(
    grad_template: Any,
    strategy: MfStrategy,
    *,
    key: RngKey,
    dtype: torch.dtype | None,
) -> tuple[Callable[..., tuple[Any, MFNoiseState]], MFNoiseState]:
    match strategy:
        case IdentityStrategy():
            return _matrix_factorization_noise(
                grad_template,
                identity(),
                key=key,
                dtype=dtype,
            )
        case LambdaCgdStrategy():
            return _make_lambda_cgd_noise(
                grad_template,
                strategy,
                key=key,
                dtype=dtype,
            )
        case BandMfStrategy() | BltStrategy() | BisrStrategy() | BsrStrategy():
            if strategy._streaming_matrix is None:
                raise ValueError(
                    "Strategy must have a _streaming_matrix for noise generation."
                )
            return _matrix_factorization_noise(
                grad_template,
                strategy._streaming_matrix,
                key=key,
                dtype=dtype,
            )
        case _:
            raise TypeError(f"Unknown strategy type: {type(strategy).__name__}")


def _resolve_noise_multiplier(noise_multiplier: float) -> float:
    multiplier = float(noise_multiplier)
    if multiplier < 0:
        raise ValueError(
            f"noise_multiplier must be non-negative, got {noise_multiplier}"
        )
    return multiplier


def _expect_bounded(value: Any, *, op: str) -> BoundedPytree:
    if isinstance(value, NoisyPytree):
        raise TypeError(
            f"{op} expects BoundedPytree inputs, not NoisyPytree values that "
            "have already passed through a noise mechanism."
        )
    if not isinstance(value, BoundedPytree):
        raise TypeError(
            f"{op} expects BoundedPytree inputs. Wrap manual values with "
            "opaque.bounded.bounded(...)."
        )
    if isinstance(value.bound, PerGroup):
        raise TypeError(
            f"{op} does not support PerGroup bounds. Per-group noise allocation "
            "for matrix-factorization mechanisms has not been validated; the "
            "interaction of per-group clipping with correlated noise across "
            "time is an open question. Use a scalar clipping_norm with "
            "MF mechanisms."
        )
    return value


def _validate_constant_bound(
    grads: BoundedPytree,
    first_bound: float | None,
    *,
    op: str,
) -> float:
    """Latch the per-step bound and reject changes across calls.

    MF privacy analyses calibrate noise from a sensitivity that is constant
    across the sequence; varying ``BoundedPytree.bound`` per call (e.g. from
    adaptive clipping) breaks the proof.  The dispatcher latches the
    first-call bound in the state and rejects any subsequent call whose
    bound differs.
    """
    sensitivity = grads.sensitivity
    if sensitivity < 0:
        raise ValueError(f"BoundedPytree bound must be non-negative, got {grads.bound}")
    bound = float(sensitivity)
    if first_bound is not None and bound != first_bound:
        raise ValueError(
            f"{op} saw a varying BoundedPytree.bound across calls "
            f"(first={first_bound}, now={bound}). MF privacy proofs assume a "
            "constant per-step sensitivity; adaptive clipping with MF noise "
            "is not supported."
        )
    return bound


def _make_second_moment_mf_noise(
    grad_template: Any,
    first_strategy: MfStrategy,
    second_strategy: MfStrategy,
    *,
    noise_multiplier: float,
    key: RngKey,
    dtype: torch.dtype | None,
    second_moment: bool | float,
) -> tuple[
    Callable[
        [Any, SecondMomentMFNoiseState],
        tuple[SecondMomentNoiseOutput, SecondMomentMFNoiseState],
    ],
    SecondMomentMFNoiseState,
]:
    overhead = resolve_second_moment_overhead(second_moment)

    first_fn, first_state = _make_raw_mf_noise(
        grad_template,
        first_strategy,
        key=rng_fold_in(key, 0),
        dtype=dtype,
    )
    second_fn, second_state = _make_raw_mf_noise(
        grad_template,
        second_strategy,
        key=rng_fold_in(key, 1),
        dtype=dtype,
    )

    init_state = SecondMomentMFNoiseState(
        _first_state=first_state,
        _second_state=second_state,
    )

    def noise_fn(
        clipped_grads: Any,
        st: SecondMomentMFNoiseState,
    ) -> tuple[SecondMomentNoiseOutput, SecondMomentMFNoiseState]:
        bounded_grads = _expect_bounded(clipped_grads, op="mf_noise")
        bound = _validate_constant_bound(
            bounded_grads, st._first_state._first_bound, op="mf_noise"
        )
        squared_bound = bound * bound
        first_stddev, second_stddev = second_moment_stddevs(
            noise_multiplier,
            bound,
            c1_max_column_norm=first_strategy._max_column_norm,
            c2_max_column_norm=second_strategy._max_column_norm,
            first_moment_overhead=overhead,
        )
        noisy_grads, new_first = first_fn(
            bounded_grads.pytree,
            st._first_state,
            stddev=first_stddev,
        )
        squared_grads = tree_map(lambda grad: grad * grad, bounded_grads.pytree)
        noisy_squared, new_second = second_fn(
            squared_grads,
            st._second_state,
            stddev=second_stddev,
        )
        return (
            SecondMomentNoiseOutput(
                NoisyPytree(
                    pytree=noisy_grads,
                    bound=bounded_grads.bound,
                    noise_stddev=first_stddev,
                ),
                NoisyPytree(
                    pytree=noisy_squared,
                    bound=squared_bound,
                    noise_stddev=second_stddev,
                ),
            ),
            SecondMomentMFNoiseState(
                _first_state=replace(new_first, _first_bound=bound),
                _second_state=replace(new_second, _first_bound=squared_bound),
            ),
        )

    return noise_fn, init_state


__all__ = [
    "BandMfStrategy",
    "BltStrategy",
    "IdentityStrategy",
    "LambdaCgdStrategy",
    "BisrStrategy",
    "BsrStrategy",
    "DEFAULT_SECOND_MOMENT_OVERHEAD",
    "MfStrategy",
    "SecondMomentMFNoiseState",
    "SecondMomentNoiseOutput",
    "band_mf_strategy",
    "blt_strategy",
    "identity_strategy",
    "lambda_cgd_strategy",
    "bisr_strategy",
    "bsr_strategy",
    "second_moment_joint_sensitivity",
    "second_moment_noise_scale",
    "second_moment_stddevs",
    "mf_noise",
]
