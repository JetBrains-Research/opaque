"""BLT (Buffered Linear Toeplitz) strategy — multi-epoch MF mechanism.

Computes optimized BLT parameters on demand from the strategy's recipe
(``max_buffers``, ``momentum``, ``lr_schedule``) plus the
amplifier-supplied ``(n_steps, min_sep, max_participations)``.  The
L-BFGS optimization is cached so a given recipe + amplification context
runs the optimizer once across all (accounting + noise) consumers.
``lr_schedule`` is an :data:`opaque.scheduling.types.Schedule`
(``Callable[[int], float]``) materialised to a tensor at workload-coefficient
build time.

References:
    - BLT: https://arxiv.org/abs/2404.16706
    - Multi-epoch BLT: https://arxiv.org/abs/2408.08868
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import torch

from opaque.api.dpftrl.noise._strategy_codec import register_strategy
from opaque.api.engine.scheduling.types import Schedule

from ._band_mf import _momentum_workload_coef
from ._blt_math import (
    BufferedToeplitz,
    inverse_as_streaming_matrix as _blt_inverse_as_streaming_matrix,
    optimize as _blt_optimize,
    sensitivity_squared as _blt_sensitivity_squared,
    toeplitz_coefs as _blt_toeplitz_coefs,
)
from ._sensitivity import minsep_true_max_participations
from ._streaming_matrix import StreamingMatrix
from ._toeplitz import (
    minsep_sensitivity_squared as _toeplitz_minsep_sensitivity_squared,
)


def _native():
    from opaque.api.accounting.core import _native as _n

    return _n


def _lr_key(lr_schedule: Schedule | None, n: int) -> tuple[float, ...] | None:
    """Materialise the schedule at ``[0, n)`` for use as an ``lru_cache`` key."""
    return (
        None if lr_schedule is None else tuple(float(lr_schedule(t)) for t in range(n))
    )


@lru_cache(maxsize=32)
def _blt_optimize_cached(
    n_steps: int,
    min_sep: int,
    max_participations: int | None,
    max_buffers: int,
    momentum: float,
    lr_key: tuple[float, ...] | None,
) -> BufferedToeplitz:
    """Run BLT L-BFGS for the given recipe + amplification context."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    lr = torch.tensor(lr_key, dtype=torch.float64) if lr_key is not None else None
    workload_coef = _momentum_workload_coef(momentum, n_steps, lr_schedule=lr)
    return _blt_optimize(
        n=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        error="max",
        max_buffers=max_buffers,
        workload_coef=workload_coef,
    )


@register_strategy
@dataclass(frozen=True, slots=True)
class BltStrategy:
    """BLT (Buffered Linear Toeplitz) strategy — recipe only.

    Carries the genuinely free knobs (optimizer hyperparams + workload
    shape).  All derived quantities are computed via the strategy
    methods, keyed on the amplifier-supplied
    ``(n_steps, min_sep, max_participations)``.
    """

    max_buffers: int = 10
    momentum: float = 1.0
    lr_schedule: Schedule | None = field(default=None, compare=False)
    # σ-independent; reused across ``balls_in_bins`` PLD probes (same strategy).
    _gram_memo: dict[tuple[int, int, int | None], tuple[float, ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    def _blt(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> BufferedToeplitz:
        return _blt_optimize_cached(
            n_steps,
            min_sep,
            max_participations,
            self.max_buffers,
            self.momentum,
            _lr_key(self.lr_schedule, n_steps),
        )

    def coefficients(
        self, *, n_steps: int, min_sep: int = 1, max_participations: int | None = None
    ) -> torch.Tensor:
        blt = self._blt(
            n_steps=n_steps, min_sep=min_sep, max_participations=max_participations
        )
        return _blt_toeplitz_coefs(blt, n_steps)

    def gram_matrix(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> tuple[float, ...]:
        key = (n_steps, min_sep, max_participations)
        memo = self._gram_memo
        hit = memo.get(key)
        if hit is not None:
            return hit
        coefs = self.coefficients(
            n_steps=n_steps, min_sep=min_sep, max_participations=max_participations
        )
        out = tuple(
            _native().toeplitz_gram_matrix(
                coefs.tolist(),
                n_steps,
                min_sep,
                max_participations,
                True,
            )
        )
        memo[key] = out
        return out

    def streaming_matrix(
        self, *, n_steps: int, min_sep: int = 1, max_participations: int | None = None
    ) -> StreamingMatrix:
        blt = self._blt(
            n_steps=n_steps, min_sep=min_sep, max_participations=max_participations
        )
        return _blt_inverse_as_streaming_matrix(blt)

    def sensitivity(
        self, *, n_steps: int, min_sep: int = 1, max_participations: int | None = None
    ) -> float:
        blt = self._blt(
            n_steps=n_steps, min_sep=min_sep, max_participations=max_participations
        )
        k = minsep_true_max_participations(
            n=n_steps, min_sep=min_sep, max_participations=max_participations
        )
        if k == 1:
            return float(_blt_sensitivity_squared(blt, n=n_steps).sqrt())
        coefs = _blt_toeplitz_coefs(blt, n_steps)
        sens_sq = _toeplitz_minsep_sensitivity_squared(
            strategy_coef=coefs,
            min_sep=min_sep,
            max_participations=max_participations,
            skip_checks=True,
        )
        return float(sens_sq.sqrt())


def blt_strategy(
    *,
    max_buffers: int = 10,
    momentum: float = 1.0,
    lr_schedule: Schedule | None = None,
) -> BltStrategy:
    """Create a BLT (Buffered Linear Toeplitz) strategy recipe.

    The L-BFGS optimization runs lazily inside the strategy methods
    when an amplifier supplies the participation context
    ``(n_steps, min_sep, max_participations)`` — once per recipe per
    context, via a module-level cache shared with the noise side.

    Args:
        max_buffers: Maximum BLT buffer count for the optimizer (default 10).
        momentum: Polyak momentum (default 1.0 = prefix-sum workload).
        lr_schedule: Optional :data:`opaque.scheduling.types.Schedule`
            (``Callable[[int], float]``).  Materialised at ``[0, n_steps)``
            when the strategy first sees the amplifier's ``n_steps``.

    Returns:
        A :class:`BltStrategy` recipe.
    """
    if max_buffers < 1:
        raise ValueError(f"max_buffers must be >= 1, got {max_buffers}")
    return BltStrategy(
        max_buffers=max_buffers,
        momentum=momentum,
        lr_schedule=lr_schedule,
    )


__all__ = ["BltStrategy", "blt_strategy"]
