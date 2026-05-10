"""Poisson amplification for DP-FTRL — whole-process accountant.

For ``BandMf`` inner, the ``n_steps`` training rounds divide into
``num_groups = ceil(n_steps / bands)`` independent groups (``bands``
the strategy bandwidth).  Each group is a Poisson-subsampled Gaussian.
For ``IdentityMf`` (encoder ``I``), ``bands == 1`` so every round is
its own group — ``num_groups == n_steps``.

The PLD is the ``num_groups``-fold composition of the per-group
Poisson-Gaussian PLD.

References:
    - BandMF amplification: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from opaque.api.accounting.core import _native

from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core.discretization import get_discretization
from opaque.dpftrl.accounting.mechanisms._band_mf import BandMf
from opaque.dpftrl.accounting.mechanisms._identity import IdentityMf

#: Mechanism types accepted by :func:`poisson`.
_Inner = BandMf | IdentityMf


@dataclass(frozen=True, slots=True)
class MfPoisson(DpProcess):
    """Poisson-amplified MF mechanism — total privacy cost over ``n_steps``.

    For ``BandMf`` inner, ``num_groups = ceil(n_steps / bands)`` where
    ``bands = len(inner.coefficients)``.  For ``IdentityMf`` inner,
    ``num_groups = n_steps``.

    Named ``MfPoisson`` (not ``Poisson``) to avoid a class-name collision
    with :class:`opaque.dpsgd.accounting.amplification.Poisson` in the
    serialization registry.  The user-facing factory is still
    :func:`poisson`.
    """

    inner: _Inner
    sample_rate: float
    n_steps: int

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
                bands = self.inner.bands
                num_groups = math.ceil(self.n_steps / bands) if bands > 0 else 0
            case IdentityMf():
                effective_nm = float(self.inner.noise_multiplier)
                num_groups = int(self.n_steps)
            case _:
                raise TypeError(
                    "Poisson requires a BandMf or IdentityMf inner, got "
                    f"{type(self.inner).__name__}."
                )

        if effective_nm == 0:
            return _native.non_private_pld(config.to_native())

        per_group_pld = _native.poisson_gaussian_pld(
            effective_nm, self.sample_rate, config.to_native()
        )
        return per_group_pld.self_compose(num_groups)


def poisson(
    inner: _Inner,
    sample_rate: float,
    *,
    n_steps: int,
) -> MfPoisson:
    """Poisson amplification for DP-FTRL.

    Whole-process accountant: returns a :class:`DpProcess` covering all
    ``n_steps`` training rounds.  Compose nothing externally with
    ``* num_steps``.

    Args:
        inner: ``BandMf`` (banded MF strategy) or ``IdentityMf``
            (uncorrelated baseline).
        sample_rate: Per-step Poisson sampling probability ``∈ (0, 1]``.
        n_steps: Total number of training rounds.  For ``BandMf`` the
            cycle count is ``ceil(n_steps / bands)``; for ``IdentityMf``
            it equals ``n_steps``.

    This accountant matches **uncapped** Poisson draws (per-group
    Binomial counts), as produced by :class:`opaque.dpftrl.sampling.CyclicPoissonSampler`.
    It does **not** model post-draw batch-size caps.

    Returns:
        An :class:`MfPoisson` process.

    Example::

        import opaque.dpftrl.accounting as ftrl_acc

        # BandMF
        proc = ftrl_acc.poisson(
            ftrl_acc.band_mf(1.0, sensitivity=s.sensitivity,
                             coefficients=s.coefficients),
            sample_rate=0.01,
            n_steps=1000,
        )

        # MF identity (DP-SGD-style baseline through the FTRL API)
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_identity(1.0),
            sample_rate=0.01,
            n_steps=1000,
        )
        eps = proc.epsilon_at(1e-5)
    """
    match inner:
        case BandMf() | IdentityMf():
            pass
        case _:
            raise TypeError(
                "poisson() requires a BandMf or IdentityMf inner, got "
                f"{type(inner).__name__}."
            )
    if not 0 < sample_rate <= 1:
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
    if int(n_steps) < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    return MfPoisson(inner=inner, sample_rate=float(sample_rate), n_steps=int(n_steps))
