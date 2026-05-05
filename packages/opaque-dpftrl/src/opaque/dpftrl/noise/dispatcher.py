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
from typing import Any

import torch

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
    stddev: float | None = None,
    noise_multiplier: float | None = None,
    sensitivity: float | None = None,
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
        stddev: Standard deviation for the base noise.  Mutually exclusive with
            ``noise_multiplier`` + ``sensitivity``.
        noise_multiplier: Gaussian noise multiplier.  Required when
            ``second_moment`` is enabled.
        sensitivity: Clipped-gradient sensitivity before the MF strategy is
            applied.  Required when ``second_moment`` is enabled.
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
    if second_moment is not False:
        if stddev is not None:
            raise ValueError(
                "second_moment mode requires noise_multiplier+sensitivity; stddev "
                "does not contain enough information to calibrate the squared stream."
            )
        if second_moment_strategy is None:
            raise ValueError(
                "second_moment_strategy is required when second_moment is enabled. "
                "Build it explicitly for the squared-gradient workload."
            )
        return _make_second_moment_mf_noise(
            grad_template,
            strategy,
            second_moment_strategy,
            noise_multiplier=noise_multiplier,
            sensitivity=sensitivity,
            key=key,
            dtype=dtype,
            second_moment=second_moment,
        )

    resolved_stddev = _resolve_stddev(
        stddev,
        noise_multiplier=noise_multiplier,
        sensitivity=sensitivity,
    )

    match strategy:
        case IdentityStrategy():
            return _matrix_factorization_noise(
                grad_template,
                identity(),
                stddev=resolved_stddev,
                key=key,
                dtype=dtype,
            )
        case LambdaCgdStrategy():
            return _make_lambda_cgd_noise(
                grad_template,
                strategy,
                stddev=resolved_stddev,
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
                stddev=resolved_stddev,
                key=key,
                dtype=dtype,
            )
        case _:
            raise TypeError(f"Unknown strategy type: {type(strategy).__name__}")


def _resolve_stddev(
    stddev: float | None,
    *,
    noise_multiplier: float | None,
    sensitivity: float | None,
) -> float:
    if stddev is not None:
        if noise_multiplier is not None or sensitivity is not None:
            raise ValueError(
                "Pass either stddev or noise_multiplier+sensitivity, not both."
            )
        if stddev < 0:
            raise ValueError(f"stddev must be non-negative, got {stddev}")
        return float(stddev)
    if noise_multiplier is None or sensitivity is None:
        raise ValueError(
            "mf_noise() requires stddev, or both noise_multiplier and sensitivity."
        )
    if noise_multiplier < 0:
        raise ValueError(
            f"noise_multiplier must be non-negative, got {noise_multiplier}"
        )
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be positive, got {sensitivity}")
    return float(noise_multiplier) * float(sensitivity)


def _make_second_moment_mf_noise(
    grad_template: Any,
    first_strategy: MfStrategy,
    second_strategy: MfStrategy,
    *,
    noise_multiplier: float | None,
    sensitivity: float | None,
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
    if noise_multiplier is None or sensitivity is None:
        raise ValueError(
            "second_moment=True requires noise_multiplier and sensitivity so both "
            "moment streams can be calibrated."
        )
    overhead = resolve_second_moment_overhead(second_moment)
    first_stddev, second_stddev = second_moment_stddevs(
        float(noise_multiplier),
        float(sensitivity),
        c1_max_column_norm=first_strategy._max_column_norm,
        c2_max_column_norm=second_strategy._max_column_norm,
        first_moment_overhead=overhead,
    )

    first_fn, first_state = mf_noise(
        grad_template,
        first_strategy,
        stddev=first_stddev,
        key=rng_fold_in(key, 0),
        dtype=dtype,
    )
    second_fn, second_state = mf_noise(
        grad_template,
        second_strategy,
        stddev=second_stddev,
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
        noisy_grads, new_first = first_fn(clipped_grads, st._first_state)
        squared_grads = tree_map(lambda grad: grad * grad, clipped_grads)
        noisy_squared, new_second = second_fn(squared_grads, st._second_state)
        return (
            SecondMomentNoiseOutput(noisy_grads, noisy_squared),
            SecondMomentMFNoiseState(
                _first_state=new_first,
                _second_state=new_second,
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
