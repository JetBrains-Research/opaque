"""Poisson amplification for DP-FTRL — whole-process accountant.

For a ``BandMfStrategy`` inner, the ``n_steps`` training rounds divide
into ``num_groups = ceil(n_steps / bands)`` independent groups
(``bands`` = strategy bandwidth).  Each group is a Poisson-subsampled
Gaussian.  For ``IdentityStrategy`` (encoder ``I``), ``bands == 1`` so
every round is its own group — ``num_groups == n_steps``.

The PLD is the ``num_groups``-fold composition of the per-group Poisson PLD.
When ``truncated_batch_size`` and
``dataset_size`` are paired (``IdentityStrategy`` only), each group
instead uses the truncated Poisson-Gaussian PLD — matching a per-step
batch cap on the runtime sampler.

References:
    - BandMF amplification: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from opaque.api.accounting.core import _native
from opaque.api.accounting.core._horizon import DpHorizonProcess
from opaque.api.accounting.core._pld_cache import horizon_pld_cache
from opaque.api.accounting.core.discretization import get_discretization
from opaque.api.accounting.dpftrl.mechanisms._mf_gaussian import MfGaussian
from opaque.api.dpftrl.noise._band_mf import BandMfStrategy
from opaque.api.dpftrl.noise._identity import IdentityStrategy
from opaque.api.dpftrl.noise._schedule_fingerprint import strategy_cache_key
from opaque.exceptions import ConfigurationError, InputTypeError

if TYPE_CHECKING:
    from opaque.api.accounting.core._base import Pld

#: Mechanism types accepted by :func:`poisson`.
_Inner = MfGaussian


@dataclass(frozen=True, slots=True)
class CyclicPoisson(DpHorizonProcess):
    """Poisson-amplified MF mechanism — total privacy cost over ``n_steps``.

    For ``BandMfStrategy`` inner, ``num_groups = ceil(n_steps / bands)``
    where ``bands = inner.strategy.bands``.  For ``IdentityStrategy``,
    ``num_groups = n_steps``.

    Plain Poisson when ``truncated_batch_size is None``; capped Poisson when
    ``truncated_batch_size`` and ``dataset_size`` are set together.
    ``sample_rate=1.0`` represents full participation, so each group's
    release is accounted as the plain Gaussian. Truncated Poisson is supported
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
                raise InputTypeError(
                    *(
                        "CyclicPoisson.atomic_unit: inner.strategy must be "
                        "BandMfStrategy or IdentityStrategy, got "
                        f"{type(self.inner.strategy).__name__}.",
                    )
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

    def __post_init__(self):
        sample_rate = float(self.sample_rate)
        if not 0 < sample_rate <= 1:
            raise ConfigurationError(
                *(
                    f"sample_rate must be in (0, 1], got {self.sample_rate}. "
                    "For q=1 (every example participates) there is no Poisson "
                    "amplification — the per-step release is the plain Gaussian.",
                )
            )
        object.__setattr__(self, "sample_rate", sample_rate)
        if sample_rate == 1.0 and self.truncated_batch_size is not None:
            raise ConfigurationError(
                *(
                    "CyclicPoisson: sample_rate=1.0 requires plain Poisson "
                    "(truncated_batch_size=None). With q=1 the batch cap yields a "
                    "fixed-size full batch, which has no truncated-Poisson "
                    "analysis; use sample_rate<1.",
                )
            )
        # Validate truncation pairing here (not only in the factory) so
        # direct construction and deserialization can't pass an unpaired
        # ``(truncated_batch_size, dataset_size)`` into
        # ``_native.truncated_poisson_gaussian_pld`` and fail at PLD time.
        if (self.truncated_batch_size is None) != (self.dataset_size is None):
            raise ConfigurationError(
                *(
                    "CyclicPoisson: truncated_batch_size and dataset_size must be set "
                    "together (both None for plain Poisson, both set for truncated).",
                )
            )
        if self.truncated_batch_size is not None:
            if int(self.truncated_batch_size) < 1:
                raise ConfigurationError(
                    *(
                        "CyclicPoisson: truncated_batch_size must be >= 1, got "
                        f"{self.truncated_batch_size}",
                    )
                )
            if int(self.dataset_size) < 1:
                raise ConfigurationError(
                    *(
                        f"CyclicPoisson: dataset_size must be >= 1, got {self.dataset_size}",
                    )
                )
            if not isinstance(self.inner.strategy, IdentityStrategy):
                raise ConfigurationError(
                    *(
                        "CyclicPoisson: truncated Poisson is only supported for "
                        "IdentityStrategy inner (BandMfStrategy per-group truncation "
                        "is not implemented). Use plain Poisson "
                        "(truncated_batch_size=None) with BandMfStrategy.",
                    )
                )

    def _pld_cache_key(self, *, n_steps: int | None = None) -> tuple[object, ...]:
        return (
            "CyclicPoisson",
            self.inner.noise_multiplier,
            self.sample_rate,
            self.n_steps,
            self.truncated_batch_size,
            self.dataset_size,
            strategy_cache_key(self.inner.strategy, self.n_steps),
        )

    @horizon_pld_cache(maxsize=8)
    def pld_at(
        self,
        n_steps: int,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        """K-step Poisson-amplified PLD using N-tuned strategy quantities.

        ``n_steps`` is rounded up to the next ``atomic_unit`` boundary
        (1 for Identity, ``bands`` for BandMF; capped at ``self.n_steps``).
        The BandMF sensitivity is read at ``self.n_steps`` (the N-tuned
        deployed mechanism); ``n_steps`` only changes the per-group
        ``self_compose`` count.  The per-group PLD factors exactly at
        band boundaries, so the rounded result is an upper bound on the
        K-step ε that is monotone in K (the post-processing inequality
        on the K-prefix of the N-step output).
        """
        if n_steps <= 0 or n_steps > self.n_steps:
            raise ConfigurationError(
                *(f"n_steps ({n_steps}) must be in [1, {self.n_steps}]",)
            )
        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )

        s = self.inner.strategy
        if isinstance(s, BandMfStrategy):
            sensitivity = s.sensitivity(
                n_steps=self.n_steps,
                min_sep=self.min_sep,
                max_participations=self.max_participations,
            )
            effective_nm = self.inner.noise_multiplier / sensitivity
            bands = s.bands
            rounded = min(-(-n_steps // bands) * bands, self.n_steps)
            num_groups = math.ceil(rounded / bands) if bands > 0 else 0
        elif isinstance(s, IdentityStrategy):
            effective_nm = float(self.inner.noise_multiplier)
            num_groups = int(n_steps)
        else:
            raise InputTypeError(
                *(
                    "Poisson requires a BandMfStrategy or IdentityStrategy "
                    f"inner.strategy, got {type(s).__name__}.",
                )
            )

        if effective_nm == 0:
            return _native.non_private_pld(config.to_native())

        native_cfg = config.to_native()
        if self.sample_rate == 1.0:
            # q=1 is no subsampling: the per-group release is the plain
            # Gaussian at the mechanism's effective noise multiplier.
            per_group_pld = _native.gaussian_pld(effective_nm, native_cfg)
        elif self.truncated_batch_size is not None:
            per_group_pld = _native.truncated_poisson_gaussian_pld(
                effective_nm,
                self.sample_rate,
                self.truncated_batch_size,
                self.dataset_size,
                native_cfg,
            )
        else:
            per_group_pld = _native.poisson_pld(
                _native.gaussian_pld(effective_nm, native_cfg),
                self.sample_rate,
            )
        return per_group_pld.self_compose(num_groups)

    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        return self.pld_at(
            self.n_steps,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )


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
            At ``1.0`` every example participates — no amplification; each
            step is accounted as the plain Gaussian.
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
        raise InputTypeError(
            *(f"poisson() requires an MfGaussian inner, got {type(inner).__name__}.",)
        )
    if not isinstance(inner.strategy, (BandMfStrategy, IdentityStrategy)):
        raise InputTypeError(
            *(
                "poisson() requires inner.strategy to be BandMfStrategy or "
                f"IdentityStrategy, got {type(inner.strategy).__name__}.",
            )
        )
    if not 0 < sample_rate <= 1:
        raise ConfigurationError(
            *(
                f"sample_rate must be in (0, 1], got {sample_rate}. "
                "For q=1 (every example participates) there is no Poisson "
                "amplification — the per-step release is the plain Gaussian.",
            )
        )
    if int(n_steps) < 1:
        raise ConfigurationError(*(f"n_steps must be >= 1, got {n_steps}",))
    return CyclicPoisson(
        inner=inner,
        sample_rate=float(sample_rate),
        n_steps=int(n_steps),
        truncated_batch_size=truncated_batch_size,
        dataset_size=dataset_size,
    )
