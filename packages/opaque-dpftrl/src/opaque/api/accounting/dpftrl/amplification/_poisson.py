"""Poisson amplification for DP-FTRL — whole-process accountant.

For a ``BandMfStrategy`` inner, the ``n_steps`` training rounds divide
into ``num_groups = ceil(n_steps / bands)`` independent groups
(``bands`` = strategy bandwidth).  Each group is a Poisson-subsampled
Gaussian.  For ``IdentityStrategy`` (encoder ``I``), ``bands == 1`` so
every round is its own group — ``num_groups == n_steps``.

The PLD is the ``num_groups``-fold composition of the per-group
Poisson-Gaussian PLD.  When ``truncated_batch_size`` and
``dataset_size`` are paired (``IdentityStrategy`` only), each group
instead uses the truncated Poisson-Gaussian PLD — matching a per-step
batch cap on the runtime sampler.

References:
    - BandMF amplification: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass

from opaque.api.accounting.core import _native

from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core.discretization import get_discretization
from opaque.api.accounting.core.mechanisms.types import Identity
from opaque.api.accounting.dpftrl._base import DpFtrlProcess
from opaque.api.accounting.dpftrl.mechanisms._mf_gaussian import MfGaussian
from opaque.api.dpftrl.noise._band_mf import BandMfStrategy
from opaque.api.dpftrl.noise._identity import IdentityStrategy

#: Mechanism types accepted by :func:`poisson`.
_Inner = MfGaussian


@dataclass(frozen=True, slots=True)
class CyclicPoisson(DpFtrlProcess):
    """Poisson-amplified MF mechanism — total privacy cost over ``n_steps``.

    For ``BandMfStrategy`` inner, ``num_groups = ceil(n_steps / bands)``
    where ``bands = inner.strategy.bands``.  For ``IdentityStrategy``,
    ``num_groups = n_steps``.

    Plain Poisson when ``truncated_batch_size is None``; truncated
    Poisson (capped batch) when ``truncated_batch_size`` and
    ``dataset_size`` are set together.  Truncated Poisson is supported
    only for ``IdentityStrategy`` (the per-step PLD reduces to the
    DP-SGD truncated Poisson-Gaussian); ``BandMfStrategy`` is rejected
    because the per-group population is the BandMF group of size
    ``|D| / bands``, not the full dataset.

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
        # BandMF factors over per-group PLDs of width ``bands``; Identity is
        # per-step (band ≡ 1).  See :meth:`pld` for the matching ``num_groups``
        # formula.
        match self.inner.strategy:
            case BandMfStrategy():
                return self.inner.strategy.bands
            case IdentityStrategy():
                return 1
            case _:
                raise TypeError(
                    "CyclicPoisson.atomic_unit: inner.strategy must be "
                    "BandMfStrategy or IdentityStrategy, got "
                    f"{type(self.inner.strategy).__name__}."
                )

    @property
    def min_sep(self) -> int:
        # Cyclic Poisson provides no worst-case separation guarantee: any
        # example could in principle be sampled on consecutive rounds.  The
        # degenerate-limit value ``1`` is what downstream consumers (BLT-family
        # noise/accounting) read; CyclicPoisson's validators already reject
        # BLT-family inners, so this value only ever shows up for BandMF or
        # Identity (which ignore it).
        return 1

    @property
    def max_participations(self) -> int:
        # Worst case: every round is a participation.  Same degenerate-limit
        # justification as :attr:`min_sep` — only BandMF/Identity ever read
        # this on a CyclicPoisson.
        return self.n_steps

    def approx_at_step(self, step: int) -> DpProcess:
        """Process truncated to its first ``step`` rounds (rounded to an atomic unit).

        Returns the *deployed-and-stopped-early* mechanism: the K-prefix
        is a deterministic projection of the full N-step output of the
        same banded Toeplitz strategy (Identity is the degenerate
        ``bands=1`` case), so ``ε(approx_at_step(K)) ≤ ε(self)`` and is
        monotone in K by the post-processing inequality.  On first
        truncation the N-tuned BandMF coefficients are pinned via
        ``coefficients_override``; Identity is horizon-independent so no
        pinning is needed there.
        """
        if step <= 0:
            return Identity()
        if step >= self.n_steps:
            return self
        unit = self.atomic_unit
        if unit < 1:
            raise ValueError(
                f"{type(self).__name__}.atomic_unit must be >= 1, got {unit}"
            )
        rounded = min(-(-step // unit) * unit, self.n_steps)
        if rounded == self.n_steps:
            return self
        s = self.inner.strategy
        if isinstance(s, BandMfStrategy) and s.coefficients_override is None:
            pinned = tuple(s.coefficients(n_steps=self.n_steps).tolist())
            new_s = dataclasses.replace(s, coefficients_override=pinned)
            new_inner = dataclasses.replace(self.inner, strategy=new_s)
            return dataclasses.replace(self, inner=new_inner, n_steps=rounded)
        return dataclasses.replace(self, n_steps=rounded)

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
            if not isinstance(self.inner.strategy, IdentityStrategy):
                raise ValueError(
                    "CyclicPoisson: truncated Poisson is only supported for "
                    "IdentityStrategy inner (BandMfStrategy per-group truncation "
                    "is not implemented). Use plain Poisson "
                    "(truncated_batch_size=None) with BandMfStrategy."
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

        s = self.inner.strategy
        if isinstance(s, BandMfStrategy):
            sensitivity = s.sensitivity(n_steps=self.n_steps)
            effective_nm = self.inner.noise_multiplier / sensitivity
            bands = s.bands
            num_groups = math.ceil(self.n_steps / bands) if bands > 0 else 0
        elif isinstance(s, IdentityStrategy):
            effective_nm = float(self.inner.noise_multiplier)
            num_groups = int(self.n_steps)
        else:
            raise TypeError(
                "Poisson requires a BandMfStrategy or IdentityStrategy "
                f"inner.strategy, got {type(s).__name__}."
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
    for ``IdentityStrategy`` inner — for ``BandMfStrategy`` it is
    rejected because the per-group population (size ``|D| / bands``)
    doesn't match the truncated Poisson-Gaussian PLD's assumption that
    Bernoulli draws happen over a fixed dataset of ``dataset_size``
    examples.

    Args:
        inner: ``mf_gaussian(nm, BandMfStrategy(...))`` or
            ``mf_gaussian(nm, identity_strategy())``.
        sample_rate: Per-step Poisson sampling probability ``∈ (0, 1]``.
        n_steps: Total number of training rounds.  For ``BandMfStrategy``
            the cycle count is ``ceil(n_steps / bands)``; for
            ``IdentityStrategy`` it equals ``n_steps``.
        truncated_batch_size: Optional max batch-size cap; switches the
            per-step analysis to truncated Poisson.  ``IdentityStrategy``
            only.
        dataset_size: Required when ``truncated_batch_size`` is set; ``|D|``.

    Returns:
        A :class:`CyclicPoisson` process.

    Example::

        import opaque.dpftrl.accounting as ftrl_acc
        from opaque.dpftrl.noise import band_mf_strategy, identity_strategy

        # BandMF
        s = band_mf_strategy(n_steps=1000, bands=10)
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, s),
            sample_rate=0.01, n_steps=1000,
        )

        # MF identity (DP-SGD-style baseline through the FTRL API)
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            sample_rate=0.01, n_steps=1000,
        )
        eps = proc.epsilon_at(1e-5)
    """
    if not isinstance(inner, MfGaussian):
        raise TypeError(
            f"poisson() requires an MfGaussian inner, got {type(inner).__name__}."
        )
    if not isinstance(inner.strategy, (BandMfStrategy, IdentityStrategy)):
        raise TypeError(
            "poisson() requires inner.strategy to be BandMfStrategy or "
            f"IdentityStrategy, got {type(inner.strategy).__name__}."
        )
    if not 0 < sample_rate <= 1:
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
    if int(n_steps) < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    return CyclicPoisson(
        inner=inner,
        sample_rate=float(sample_rate),
        n_steps=int(n_steps),
        truncated_batch_size=truncated_batch_size,
        dataset_size=dataset_size,
    )
