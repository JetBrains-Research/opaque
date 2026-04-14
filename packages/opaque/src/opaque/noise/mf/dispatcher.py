"""MF strategy types and unified noise dispatcher.

Each mechanism file defines its own strategy dataclass and factory:
- ``identity.py``: :class:`IdentityStrategy`, :func:`identity_strategy`
- ``band_mf.py``: :class:`BandMfStrategy`, :func:`band_mf_strategy`
- ``blt.py``: :class:`BltStrategy`, :func:`blt_strategy`
- ``lambda_cgd.py``: :class:`LambdaCgdStrategy`, :func:`lambda_cgd_strategy`
- ``bisr.py``: :class:`BisrStrategy`, :func:`bisr_strategy`

The :func:`mf_noise` function dispatches on the strategy type to create
the appropriate noise mechanism.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .band_mf import BandMfStrategy, band_mf_strategy
from .bisr import BisrStrategy, bisr_strategy
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
from opaque.random import RngKey

MfStrategy = (
    BandMfStrategy | BltStrategy | LambdaCgdStrategy | BisrStrategy | IdentityStrategy
)


def mf_noise(
    grad_template: Any,
    strategy: MfStrategy,
    *,
    stddev: float,
    key: RngKey,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Create a correlated noise mechanism for the given MF strategy.

    Dispatches on the strategy type:
    - :class:`LambdaCgdStrategy`: PRNG replay noise (zero extra memory).
    - :class:`BandMfStrategy`, :class:`BltStrategy`, :class:`BisrStrategy`:
      StreamingMatrix-based noise.

    Args:
        grad_template: Pytree with same structure/shapes as gradients.
        strategy: MF strategy from one of the factory functions.
        stddev: Standard deviation for the base noise.
        key: Explicit RNG key for deterministic randomness.
        dtype: Optional dtype for intermediate noise computation.

    Returns:
        A tuple ``(noise_fn, state)`` for the training loop.
    """
    match strategy:
        case IdentityStrategy():
            return _matrix_factorization_noise(
                grad_template,
                identity(),
                stddev=stddev,
                key=key,
                dtype=dtype,
            )
        case LambdaCgdStrategy():
            return _make_lambda_cgd_noise(
                grad_template,
                strategy,
                stddev=stddev,
                key=key,
                dtype=dtype,
            )
        case BandMfStrategy() | BltStrategy() | BisrStrategy():
            if strategy._streaming_matrix is None:
                raise ValueError(
                    "Strategy must have a _streaming_matrix for noise generation."
                )
            return _matrix_factorization_noise(
                grad_template,
                strategy._streaming_matrix,
                stddev=stddev,
                key=key,
                dtype=dtype,
            )
        case _:
            raise TypeError(f"Unknown strategy type: {type(strategy).__name__}")


__all__ = [
    "BandMfStrategy",
    "BltStrategy",
    "IdentityStrategy",
    "LambdaCgdStrategy",
    "BisrStrategy",
    "MfStrategy",
    "band_mf_strategy",
    "blt_strategy",
    "identity_strategy",
    "lambda_cgd_strategy",
    "bisr_strategy",
    "mf_noise",
]
