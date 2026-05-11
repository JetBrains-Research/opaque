"""Poisson amplification for DP-FTRL — whole-process accountant.

For ``BandMf`` inner, the ``n_steps`` training rounds divide into
``num_groups = ceil(n_steps / bands)`` independent groups (``bands``
the strategy bandwidth).  Each group is a Poisson-subsampled Gaussian.
For ``IdentityMf`` (encoder ``I``), ``bands == 1`` so every round is
its own group — ``num_groups == n_steps``.

The PLD is the ``num_groups``-fold composition of the per-group
Poisson-Gaussian PLD.  When ``truncated_batch_size`` and
``dataset_size`` are paired (``IdentityMf`` only), each group instead
uses the truncated Poisson-Gaussian PLD — matching a per-step batch
cap on the runtime sampler.

References:
    - BandMF amplification: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from opaque.api.accounting.core import _native

from opaque.api.accounting.core._base import Pld
from opaque.api.accounting.core.discretization import get_discretization
from opaque.api.accounting.dpftrl._base import DpFtrlProcess
from opaque.api.accounting.dpftrl.mechanisms._band_mf import BandMf
from opaque.api.accounting.dpftrl.mechanisms._identity import IdentityMf

#: Mechanism types accepted by :func:`poisson`.
_Inner = BandMf | IdentityMf


@dataclass(frozen=True, slots=True)
class CyclicPoisson(DpFtrlProcess):
    """Poisson-amplified MF mechanism — total privacy cost over ``n_steps``.

    For ``BandMf`` inner, ``num_groups = ceil(n_steps / bands)`` where
    ``bands = len(inner.coefficients)``.  For ``IdentityMf`` inner,
    ``num_groups = n_steps``.

    Plain Poisson when ``truncated_batch_size is None``; truncated
    Poisson (capped batch) when ``truncated_batch_size`` and
    ``dataset_size`` are set together.  Truncated Poisson is supported
    only for ``IdentityMf`` (the per-step PLD reduces to the DP-SGD
    truncated Poisson-Gaussian); ``BandMf`` is rejected because the
    per-group population is the BandMF group of size ``|D| / bands``,
    not the full dataset, and that analysis has not been vetted here.

    Named ``CyclicPoisson`` (not ``Poisson``) to avoid a class-name collision
    with :class:`opaque.dpsgd.accounting.amplification.Poisson` in the
    serialization registry.  The user-facing factory is still
    :func:`poisson`.
    """

    inner: _Inner
    sample_rate: float
    n_steps: int
    truncated_batch_size: int | None = None
    dataset_size: int | None = None

    @property
    def atomic_unit(self) -> int:
        # BandMf factors over per-group PLDs of width ``bands``; IdentityMf is
        # per-step (band ≡ 1).  See :meth:`pld` for the matching ``num_groups``
        # formula.
        match self.inner:
            case BandMf():
                return self.inner.bands
            case IdentityMf():
                return 1
            case _:
                raise TypeError(
                    "CyclicPoisson.atomic_unit: inner must be BandMf or "
                    f"IdentityMf, got {type(self.inner).__name__}."
                )

    def __post_init__(self):
        # Validate truncation pairing here (not only in the factory) so
        # direct construction and deserialization can't pass an unpaired
        # ``(truncated_batch_size, dataset_size)`` into
        # ``_native.truncated_poisson_gaussian_pld`` and fail at PLD time.
        if (self.truncated_batch_size is None) != (self.dataset_size is None):
            raise ValueError(
                "CyclicPoisson: truncated_batch_size and dataset_size must be set "
                "together (both None for plain Poisson, both set for truncated)."
            )
        if self.truncated_batch_size is not None:
            if int(self.truncated_batch_size) < 1:
                raise ValueError(
                    "CyclicPoisson: truncated_batch_size must be >= 1, got "
                    f"{self.truncated_batch_size}"
                )
            if int(self.dataset_size) < 1:
                raise ValueError(
                    f"CyclicPoisson: dataset_size must be >= 1, got {self.dataset_size}"
                )
            if not isinstance(self.inner, IdentityMf):
                raise ValueError(
                    "CyclicPoisson: truncated Poisson is only supported for "
                    "IdentityMf inner (BandMf per-group truncation is not "
                    "implemented). Use plain Poisson "
                    "(truncated_batch_size=None) with BandMf."
                )

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

        native_cfg = config.to_native()
        if self.truncated_batch_size is not None:
            per_group_pld = _native.truncated_poisson_gaussian_pld(
                effective_nm,
                self.sample_rate,
                self.truncated_batch_size,
                self.dataset_size,
                native_cfg,
            )
        else:
            per_group_pld = _native.poisson_gaussian_pld(
                effective_nm, self.sample_rate, native_cfg
            )
        return per_group_pld.self_compose(num_groups)


def poisson(
    inner: _Inner,
    sample_rate: float,
    *,
    n_steps: int,
    truncated_batch_size: int | None = None,
    dataset_size: int | None = None,
) -> CyclicPoisson:
    """Poisson amplification for DP-FTRL.

    Whole-process accountant: returns a :class:`DpProcess` covering all
    ``n_steps`` training rounds.  Compose nothing externally with
    ``* num_steps``.

    Plain Poisson when ``truncated_batch_size is None``; truncated
    Poisson (capped batch) when ``truncated_batch_size`` and
    ``dataset_size`` are both set.  Truncated Poisson is only supported
    for ``IdentityMf`` inner — for ``BandMf`` it is rejected because the
    per-group population (size ``|D| / bands``) doesn't match the
    truncated Poisson-Gaussian PLD's assumption that Bernoulli draws
    happen over a fixed dataset of ``dataset_size`` examples.

    Args:
        inner: ``BandMf`` (banded MF strategy) or ``IdentityMf``
            (uncorrelated baseline).
        sample_rate: Per-step Poisson sampling probability ``∈ (0, 1]``.
        n_steps: Total number of training rounds.  For ``BandMf`` the
            cycle count is ``ceil(n_steps / bands)``; for ``IdentityMf``
            it equals ``n_steps``.
        truncated_batch_size: Optional max batch-size cap; switches the
            per-step analysis to truncated Poisson.  ``IdentityMf`` only.
        dataset_size: Required when ``truncated_batch_size`` is set;
            ``|D|``.

    This accountant matches uncapped Poisson draws (per-group Binomial
    counts) by default, as produced by
    :class:`opaque.dpftrl.sampling.CyclicPoissonSampler`.  When
    ``truncated_batch_size`` is provided (``IdentityMf`` only), it
    matches the same sampler with its matching cap.

    Returns:
        A :class:`CyclicPoisson` process.

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
            ftrl_acc.identity_mf(1.0),
            sample_rate=0.01,
            n_steps=1000,
        )
        eps = proc.epsilon_at(1e-5)

        # MF identity with truncated Poisson (production batch cap)
        n, batch = 50_000, 250
        proc = ftrl_acc.poisson(
            ftrl_acc.identity_mf(0.8),
            sample_rate=batch / n,
            n_steps=1000,
            truncated_batch_size=batch,
            dataset_size=n,
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
    # Pairing + per-field bounds + IdentityMf-only check on
    # truncated_batch_size / dataset_size are validated in
    # ``CyclicPoisson.__post_init__`` so direct construction stays safe.
    return CyclicPoisson(
        inner=inner,
        sample_rate=float(sample_rate),
        n_steps=int(n_steps),
        truncated_batch_size=truncated_batch_size,
        dataset_size=dataset_size,
    )
