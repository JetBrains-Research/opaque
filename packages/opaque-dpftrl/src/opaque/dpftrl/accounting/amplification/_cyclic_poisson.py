"""Cyclic Poisson amplification — BandMF or MF identity inner.

For BandMF, the ``n`` training rounds divide into
``k = ceil(n/b)`` independent groups (``b`` band width); each group is a
Poisson-subsampled Gaussian.  For MF identity (encoder ``I``), there is no
banding — every round is its own group, so ``num_steps`` is required from
the caller and acts as the group count.

The PLD is the ``num_groups``-fold (BandMF) or ``num_steps``-fold (identity)
composition of the per-group Poisson-Gaussian PLD.

References:
    - BandMF amplification: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.accounting import _native

from opaque.accounting._base import DpProcess, Pld
from opaque.accounting.discretization import get_discretization
from opaque.dpftrl.accounting.mechanisms._band_mf import BandMf
from opaque.dpftrl.accounting.mechanisms._identity import IdentityMf

#: Mechanism types accepted by :func:`cyclic_poisson`.
_Inner = BandMf | IdentityMf


@dataclass(frozen=True, slots=True)
class CyclicPoisson(DpProcess):
    """Cyclic Poisson amplification — BandMF or MF identity.

    For ``BandMf`` inner, ``num_groups`` is read from the inner mechanism.
    For ``IdentityMf`` inner, ``num_steps`` (required at construction) acts
    as the group count: every step is its own independent Poisson group.
    """

    inner: _Inner
    sample_rate: float
    num_steps: int | None = None

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )

        match self.inner:
            case BandMf():
                effective_nm = self.inner.noise_multiplier / self.inner.sensitivity
                num_groups = self.inner.num_groups
            case IdentityMf():
                if self.num_steps is None:
                    raise ValueError(
                        "CyclicPoisson with IdentityMf requires num_steps."
                    )
                effective_nm = float(self.inner.noise_multiplier)
                num_groups = int(self.num_steps)
            case _:
                raise TypeError(
                    "CyclicPoisson requires a BandMf or IdentityMf inner, got "
                    f"{type(self.inner).__name__}."
                )

        if effective_nm == 0:
            return _native.non_private_pld(config.to_native())

        per_group_pld = _native.poisson_gaussian_pld(
            effective_nm, self.sample_rate, config.to_native()
        )
        return per_group_pld.self_compose(num_groups)


def cyclic_poisson(
    inner: _Inner,
    sample_rate: float,
    *,
    num_steps: int | None = None,
) -> CyclicPoisson:
    """Cyclic Poisson amplification.

    For ``BandMf`` inner the cycle count comes from ``inner.num_groups`` (and
    ``num_steps`` may be omitted, or passed and validated to match).  For
    ``IdentityMf`` (no banding) ``num_steps`` is required and equals the total
    number of training rounds.

    Args:
        inner: ``BandMf`` or ``IdentityMf`` mechanism.
        sample_rate: Per-step (per-group) Poisson sampling probability ``∈ (0, 1]``.
        num_steps: Required for ``IdentityMf``; for ``BandMf`` either ``None`` (use
            ``inner.num_groups``) or equal to ``inner.num_groups``.

    Returns:
        A :class:`CyclicPoisson` process.

    Example::

        import opaque.dpftrl.accounting as ftrl_acc

        # BandMF
        proc = ftrl_acc.cyclic_poisson(
            ftrl_acc.band_mf(1.0, sensitivity=1.0, num_groups=100),
            sample_rate=0.01,
        )

        # MF identity (DP-SGD-style baseline through MF API)
        proc = ftrl_acc.cyclic_poisson(
            ftrl_acc.mf_identity(1.0),
            sample_rate=0.01,
            num_steps=1000,
        )
        eps = proc.epsilon_at(1e-5)
    """
    match inner:
        case BandMf():
            if num_steps is not None and num_steps != inner.num_groups:
                raise ValueError(
                    "cyclic_poisson(BandMf, num_steps=...) must match "
                    f"inner.num_groups={inner.num_groups}, got {num_steps}."
                )
            if inner.num_groups < 1:
                raise ValueError(
                    "cyclic_poisson requires BandMf with num_groups >= 1, "
                    f"got {inner.num_groups}"
                )
            resolved_num_steps = inner.num_groups
        case IdentityMf():
            if num_steps is None:
                raise ValueError(
                    "cyclic_poisson(mf_identity(...), ...) requires num_steps."
                )
            if int(num_steps) < 1:
                raise ValueError(f"num_steps must be >= 1, got {num_steps}")
            resolved_num_steps = int(num_steps)
        case _:
            raise TypeError(
                "cyclic_poisson() requires a BandMf or IdentityMf inner, got "
                f"{type(inner).__name__}."
            )
    if not 0 < sample_rate <= 1:
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")

    return CyclicPoisson(
        inner=inner, sample_rate=float(sample_rate), num_steps=resolved_num_steps
    )
