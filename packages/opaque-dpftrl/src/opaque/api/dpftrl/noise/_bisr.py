"""BISR strategy — Banded Inverse Square Root MF mechanism.

BISR (Kalinin et al., ICLR 2026) generalises lambda-CGD to arbitrary
bandwidth p.  The inverse strategy matrix :math:`C^{-1}` is banded
Toeplitz with p coefficients.

References:
    - Kalinin, McKenna, Upadhyay, Lampert (2026) "Back to Square Roots"
      https://arxiv.org/abs/2505.12128
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch

from opaque.api.dpftrl.noise._strategy_codec import register_strategy

from ._toeplitz import inverse_as_streaming_matrix

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.random.types import RngKey
    from opaque.types import PerGroup

    from ._engine import MFNoiseState
    from collections.abc import Sequence

    from ._streaming_matrix import StreamingMatrix


def _native():
    from opaque.api.accounting.core import _native as _n

    return _n


@lru_cache(maxsize=256)
def _bisr_gram_matrix_cached(
    inv: tuple[float, ...],
    normalized: bool,
    n_steps: int,
    min_sep: int,
    max_participations: int | None,
) -> tuple[float, ...]:
    """Gram sequence for BISR; cached across repeated σ / PLD probes."""
    return tuple(
        _native().bisr_gram_matrix(
            list(inv), n_steps, min_sep, max_participations, normalized
        )
    )


@lru_cache(maxsize=32)
def _bisr_inverse_coefficients_cached(bandwidth: int, beta: float) -> tuple[float, ...]:
    """Compute BISR inverse square-root coefficients (Lemma 1, arxiv:2505.12128).

    For alpha=1: c_k = sum_{j=0}^{k} r_j * beta^j * r_{k-j}
    where r_0 = 1, r_j = ((j - 3/2) / j) * r_{j-1}.
    """
    r_tilde = [0.0] * bandwidth
    r_tilde[0] = 1.0
    for j in range(1, bandwidth):
        r_tilde[j] = ((j - 1.5) / j) * r_tilde[j - 1]

    if beta == 0.0:
        return tuple(r_tilde)

    coefs = [0.0] * bandwidth
    for k in range(bandwidth):
        s = 0.0
        for j in range(k + 1):
            s += r_tilde[j] * (beta**j) * r_tilde[k - j]
        coefs[k] = s
    return tuple(coefs)


def _recover_strategy_coefficients(inv_coefs: Sequence[float], n: int) -> list[float]:
    """Recover strategy matrix first-column entries from C^{-1} coefficients."""
    alpha0 = inv_coefs[0]
    p = len(inv_coefs)
    col = [0.0] * n
    col[0] = 1.0 / alpha0
    for t in range(1, n):
        s = 0.0
        k_max = min(t, p - 1)
        for k in range(1, k_max + 1):
            s += inv_coefs[k] * col[t - k]
        col[t] = -s / alpha0
    return col


def _make_bisr_noise(
    grad_template: Any,
    strategy: "BisrStrategy",
    *,
    n_steps: int,
    key: "RngKey",
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[
    "Callable[..., tuple[Any, MFNoiseState]]",
    "MFNoiseState",
    "Callable[[int], float]",
]:
    from ._engine import _check_mf_horizon, _matrix_factorization_noise

    n_steps = int(n_steps)
    streaming = strategy.streaming_matrix(n_steps=n_steps)
    noise_fn, state = _matrix_factorization_noise(
        grad_template,
        streaming,
        key=key,
        compute_dtype=compute_dtype,
        n_steps=n_steps,
    )
    row_norms = streaming.row_norms_squared(n_steps).clamp_min(0.0).sqrt()

    def row_l2_at(step: int) -> float:
        _check_mf_horizon(step, n_steps)
        return float(row_norms[step])

    return noise_fn, state, row_l2_at


@register_strategy
@dataclass(frozen=True, slots=True)
class BisrStrategy:
    """BISR (Banded Inverse Square Root) strategy — recipe only.

    Carries the workload knobs and an optional explicit
    ``inv_coefficients`` override for :math:`C^{-1}`.  Derived
    quantities are computed on demand.
    """

    bandwidth: int
    normalized: bool = True
    momentum: float = 0.0
    inv_coefficients: tuple[float, ...] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.bandwidth < 2:
            raise ValueError(f"bandwidth must be >= 2, got {self.bandwidth}")
        if (
            self.inv_coefficients is not None
            and len(self.inv_coefficients) != self.bandwidth
        ):
            raise ValueError(
                f"inv_coefficients length ({len(self.inv_coefficients)}) must "
                f"equal bandwidth ({self.bandwidth})"
            )

    def _inv_coefs(self) -> tuple[float, ...]:
        if self.inv_coefficients is not None:
            return self.inv_coefficients
        return _bisr_inverse_coefficients_cached(self.bandwidth, self.momentum)

    def coefficients(self, *, n_steps: int, **_) -> torch.Tensor:
        inv = self._inv_coefs()
        return torch.tensor(
            _recover_strategy_coefficients(inv, n_steps), dtype=torch.float64
        )

    def gram_matrix(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> tuple[float, ...]:
        return _bisr_gram_matrix_cached(
            self._inv_coefs(),
            self.normalized,
            n_steps,
            min_sep,
            max_participations,
        )

    def streaming_matrix(self, *, n_steps: int, **_) -> StreamingMatrix:
        inv = list(self._inv_coefs())
        strategy_coefs = _native().bisr_strategy_coefficients(inv, self.bandwidth)
        return inverse_as_streaming_matrix(
            torch.tensor(strategy_coefs, dtype=torch.float64),
            column_normalize_for_n=n_steps if self.normalized else None,
        )

    def raw_noise_factory(
        self,
        grad_template: Any,
        *,
        n_steps: int,
        min_sep: int,
        max_participations: int | None,
        key: "RngKey",
        compute_dtype: torch.dtype,
    ):
        del min_sep, max_participations
        return _make_bisr_noise(
            grad_template,
            self,
            n_steps=n_steps,
            key=key,
            compute_dtype=compute_dtype,
        )

    def sensitivity(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> float:
        inv = list(self._inv_coefs())
        if self.normalized:
            sens_sq = _native().bisr_normalized_sensitivity_squared(
                inv, n_steps, min_sep, max_participations
            )
        else:
            sens_sq = _native().bisr_sensitivity_squared(
                inv, n_steps, min_sep, max_participations
            )
        return float(sens_sq**0.5)


def bisr_strategy(
    *,
    bandwidth: int,
    normalized: bool = True,
    momentum: float = 0.0,
    inv_coefficients: Sequence[float] | None = None,
) -> BisrStrategy:
    """Create a BISR (Banded Inverse Square Root) strategy recipe.

    Args:
        bandwidth: BISR bandwidth p (>= 2).
        normalized: Use column-normalized matrix (default True).
        momentum: Optimizer momentum in [0, 1) (default 0).
        inv_coefficients: Explicit :math:`C^{-1}` coefficients (default BISR optimal).

    Returns:
        A :class:`BisrStrategy` recipe.
    """
    return BisrStrategy(
        bandwidth=bandwidth,
        normalized=normalized,
        momentum=momentum,
        inv_coefficients=(
            tuple(inv_coefficients) if inv_coefficients is not None else None
        ),
    )


__all__ = ["BisrStrategy", "bisr_strategy"]
