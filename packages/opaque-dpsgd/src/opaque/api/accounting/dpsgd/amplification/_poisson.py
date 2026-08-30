"""Poisson-subsampled Gaussian mechanism — standard DP-SGD step.

Plain or truncated.  When ``truncated_batch_size`` and ``dataset_size`` are
provided, computes the truncated Poisson-Gaussian PLD via the
``truncated_poisson_gaussian_pld`` native primitive.
"""

from __future__ import annotations

from dataclasses import dataclass

from opaque.api.accounting.core import _native
from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core._pld_cache import pld_cache
from opaque.api.accounting.core.mechanisms._nonprivate import NonPrivate
from opaque.api.accounting.dpsgd.mechanisms._adaclip import AdaClip
from opaque.api.accounting.dpsgd.mechanisms._gaussian import Gaussian
from opaque.exceptions import ConfigurationError, InputTypeError

#: Mechanism types accepted by :func:`poisson`.
_Inner = Gaussian | AdaClip | NonPrivate


@dataclass(frozen=True, slots=True)
class Poisson(DpProcess):
    """Poisson-subsampled Gaussian mechanism (single step).

    Plain Poisson when ``truncated_batch_size is None``; truncated Poisson
    (capped batch) when ``truncated_batch_size`` and ``dataset_size`` are
    set together.  ``sample_rate`` is in ``(0, 1]``; at ``sample_rate=1.0``
    every example participates and the step is accounted as the unamplified
    Gaussian mechanism.
    """

    inner: _Inner
    sample_rate: float
    truncated_batch_size: int | None = None
    dataset_size: int | None = None

    def __post_init__(self):
        sample_rate = float(self.sample_rate)
        if not 0 < sample_rate <= 1:
            raise ConfigurationError(
                *(
                    f"sample_rate must be in (0, 1], got {self.sample_rate}. "
                    "For q=1 (every example participates) there is no Poisson "
                    "amplification — account the step with gaussian(nm) directly.",
                )
            )
        object.__setattr__(self, "sample_rate", sample_rate)
        if sample_rate == 1.0 and self.truncated_batch_size is not None:
            raise ConfigurationError(
                *(
                    "Poisson: sample_rate=1.0 requires plain Poisson "
                    "(truncated_batch_size=None). With q=1 the batch cap yields a "
                    "fixed-size full batch, which has no truncated-Poisson "
                    "analysis — account a full-batch step with gaussian(nm), or "
                    "use sample_rate<1.",
                )
            )

        # Validate truncation pairing here (not only in the factory) so direct
        # construction and deserialization can't pass an unpaired
        # ``(truncated_batch_size, dataset_size)`` into
        # ``_native.truncated_poisson_gaussian_pld`` and fail at PLD time.
        if (self.truncated_batch_size is None) != (self.dataset_size is None):
            raise ConfigurationError(
                *(
                    "Poisson: truncated_batch_size and dataset_size must be set "
                    "together (both None for plain Poisson, both set for truncated).",
                )
            )
        if self.truncated_batch_size is not None:
            if int(self.truncated_batch_size) < 1:
                raise ConfigurationError(
                    *(
                        "Poisson: truncated_batch_size must be >= 1, got "
                        f"{self.truncated_batch_size}",
                    )
                )
            if int(self.dataset_size) < 1:
                raise ConfigurationError(
                    *(f"Poisson: dataset_size must be >= 1, got {self.dataset_size}",)
                )

    @pld_cache(maxsize=8)
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
        from opaque.api.accounting.core.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )

        native_cfg = config.to_native()
        truncated = self.truncated_batch_size is not None
        # q=1 is no subsampling: the step is the plain Gaussian mechanism
        # (matches the native q→1 limit of poisson_gaussian_pld).
        full_step = self.sample_rate == 1.0

        match self.inner:
            case NonPrivate() | Gaussian(noise_multiplier=0):
                return _native.non_private_pld(native_cfg)
            case Gaussian(noise_multiplier=nm):
                if full_step:
                    return _native.gaussian_pld(nm, native_cfg)
                if truncated:
                    return _native.truncated_poisson_gaussian_pld(
                        nm,
                        self.sample_rate,
                        self.truncated_batch_size,
                        self.dataset_size,
                        native_cfg,
                    )
                return _native.poisson_gaussian_pld(nm, self.sample_rate, native_cfg)
            case AdaClip(inner=NonPrivate() | Gaussian(noise_multiplier=0)):
                return _native.non_private_pld(native_cfg)
            case AdaClip(inner=Gaussian()) as ac:
                if full_step:
                    return _native.gaussian_pld(
                        ac.effective_noise_multiplier, native_cfg
                    )
                if truncated:
                    return _native.truncated_poisson_gaussian_pld(
                        ac.effective_noise_multiplier,
                        self.sample_rate,
                        self.truncated_batch_size,
                        self.dataset_size,
                        native_cfg,
                    )
                return _native.poisson_gaussian_pld(
                    ac.effective_noise_multiplier,
                    self.sample_rate,
                    native_cfg,
                )
            case _:
                raise InputTypeError(
                    *(
                        "Poisson requires a Gaussian, AdaClip(Gaussian), or "
                        "NonPrivate inner mechanism, got "
                        f"{type(self.inner).__name__}.",
                    )
                )


def poisson(
    inner: _Inner,
    sample_rate: float,
    *,
    truncated_batch_size: int | None = None,
    dataset_size: int | None = None,
) -> Poisson:
    """Poisson-subsampled Gaussian mechanism (per-step DP-SGD factory).

    Returns a single-step process — compose externally with ``* num_steps``
    for full-training privacy.

    When ``truncated_batch_size`` is set, both it and ``dataset_size`` are
    required, and the analysis switches to the truncated Poisson-Gaussian
    PLD (production DP-SGD with capped batch size).

    Args:
        inner: The base mechanism — :func:`gaussian`, :func:`adaclip`, or
            :func:`opaque.accounting.nonprivate`.
        sample_rate: Probability of including each example
            (``E[batch_size] / |D|``), between zero and one inclusive.
            ``sample_rate=1.0`` means every example participates — no
            amplification; the step is accounted as the plain Gaussian
            (equivalent to ``gaussian(nm)``).
        truncated_batch_size: Optional max batch-size cap; switches the
            analysis to truncated Poisson (requires ``sample_rate < 1``).
        dataset_size: Required when ``truncated_batch_size`` is set;
            ``|D|``.

    Returns:
        A :class:`Poisson` process.

    Example::

        # Plain Poisson
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), sample_rate=0.01)
        training = step * 1000

        # Truncated Poisson (production)
        n, batch = 50_000, 250
        step = dpsgd_acc.poisson(
            dpsgd_acc.gaussian(0.8), batch / n,
            truncated_batch_size=batch, dataset_size=n,
        )
        eps = (step * 1000).epsilon_at(1e-5)
    """
    match inner:
        case Gaussian() | AdaClip() | NonPrivate():
            pass
        case _:
            raise InputTypeError(
                *(
                    "poisson() requires a Gaussian, AdaClip, or NonPrivate inner "
                    f"mechanism, got {type(inner).__name__}. "
                    "Example: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), rate)",
                )
            )
    # Pairing + per-field bounds on truncated_batch_size / dataset_size are
    # validated in ``Poisson.__post_init__`` so direct construction,
    # deserialization, and this factory stay consistent.
    return Poisson(
        inner=inner,
        sample_rate=float(sample_rate),
        truncated_batch_size=truncated_batch_size,
        dataset_size=dataset_size,
    )
