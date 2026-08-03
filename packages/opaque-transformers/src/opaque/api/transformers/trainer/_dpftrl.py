"""DP-FTRL integration helpers for :class:`DPTrainer`.

Pure factories: MF strategy construction, accountant amplifier
construction, sampler dispatch.  No state lives here — the trainer
owns the lifecycle.  The trainer calls these in :meth:`_setup_training`
(strategy + amplifier) and :meth:`get_train_dataloader` (sampler) when
:attr:`TrainingArguments.privacy_noise_mechanism` starts with ``"mf_"``.

The MF amplifier returned by :func:`build_amplifier_factory` is the
    *raw* whole-process accountant (a :class:`DpHorizonProcess`); the trainer
queries ``(n_steps, min_sep, max_participations)`` off it to build the
matching :func:`opaque.dpftrl.noise.mf_gaussian_noise` and then wraps
it with :func:`opaque.accounting.per_step` for the
``acc |= step`` composition idiom.  See
:func:`build_step_mechanism_factory`.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from opaque.accounting import per_step
from opaque.dpftrl import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
)
from opaque.dpftrl.accounting import (
    b_min_sep as _ftrl_b_min_sep,
)
from opaque.dpftrl.accounting import (
    balls_in_bins as _ftrl_balls_in_bins,
)
from opaque.dpftrl.accounting import mf_gaussian
from opaque.dpftrl.accounting import (
    poisson as _ftrl_poisson,
)
from opaque.dpsgd.sampling import (
    KOutOfTSampler,
    PoissonSampler,
    RandomAllocationSampler,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.random.types import RngKey


@dataclasses.dataclass(frozen=True)
class MFContext:
    """DP-FTRL provenance carried through the training loop.

    ``strategy`` is the built BandMF / BLT recipe; ``amplifier_factory``
    produces the raw DpHorizonProcess for a calibrated multiplier so callers
    (noise construction, sampler construction, checkpoint save) can read
    ``(n_steps, min_sep, max_participations, sampling_prob)`` off it on
    demand — the recipe + the amplifier are the single source of truth
    for everything privacy-derived; nothing downstream should re-parse
    the user's mechanism kwargs.
    """

    strategy: Any
    amplifier_factory: Callable[[float], Any]


# Strategy factory dispatch — keyed by ``privacy_noise_mechanism``.
_STRATEGY_FACTORIES: dict[str, Callable[..., Any]] = {
    "mf_band": band_mf_strategy,
    "mf_blt": blt_strategy,
    "mf_bisr": bisr_strategy,
    "mf_bsr": bsr_strategy,
    "mf_lambda_cgd": lambda_cgd_strategy,
    "mf_identity": identity_strategy,
}

# Strategies that consume the optimizer LR schedule for workload-aware
# Toeplitz coefficient tuning.  Other strategies ignore the LR schedule.
_LR_SCHEDULED_STRATEGIES: frozenset[str] = frozenset({"mf_band", "mf_blt"})


def build_strategy(
    mechanism: str,
    kwargs: dict[str, Any] | None,
    lr_schedule: Any = None,
) -> Any:
    """Build the MF strategy recipe for ``mechanism``.

    ``kwargs`` are forwarded verbatim to the strategy factory; the
    trainer pre-merges its per-mechanism defaults in
    ``TrainingArguments.__post_init__``.

    For BandMF / BLT, the optimizer ``lr_schedule`` is auto-injected
    when the user did not already supply one in ``kwargs`` — those
    strategies tune their Toeplitz coefficients against the schedule
    for tighter privacy at the realised workload.  Schedule recipes
    from :mod:`opaque.scheduling` round-trip cleanly through the
    accountant; raw lambdas do not (a clear error from the strategy
    codec surfaces at serialization time).
    """
    factory = _STRATEGY_FACTORIES[mechanism]
    if mechanism == "mf_identity":
        return factory()
    extra = dict(kwargs) if kwargs else {}
    if (
        mechanism in _LR_SCHEDULED_STRATEGIES
        and lr_schedule is not None
        and "lr_schedule" not in extra
    ):
        extra["lr_schedule"] = lr_schedule
    return factory(**extra)


def build_amplifier_factory(
    *,
    sampling_mode: str,
    strategy: Any,
    sample_rate: float,
    n_steps: int,
    num_bins: int,
    dataset_size: int,
    truncated_batch_size: int | None,
) -> Callable[[float], Any]:
    """Return ``nm → DpHorizonProcess`` — the *raw* amplifier instance.

    The trainer holds the result on its training context and queries it
    with the calibrated noise multiplier at noise-function construction
    time to read off ``(n_steps, min_sep, max_participations)``.
    """
    if sampling_mode == "poisson":

        def amp(
            nm: float,
            _s: Any = strategy,
            _sr: float = sample_rate,
            _ns: int = n_steps,
            _tb: int | None = truncated_batch_size,
            _ds: int = dataset_size,
        ) -> Any:
            return _ftrl_poisson(
                mf_gaussian(nm, _s),
                sample_rate=_sr,
                n_steps=_ns,
                truncated_batch_size=_tb,
                dataset_size=_ds if _tb is not None else None,
            )

    elif sampling_mode == "b_min_sep":

        def amp(
            nm: float,
            _s: Any = strategy,
            _ns: int = n_steps,
            _p0: float = sample_rate,
        ) -> Any:
            return _ftrl_b_min_sep(mf_gaussian(nm, _s), n_steps=_ns, p0=_p0)

    elif sampling_mode == "balls_in_bins":

        def amp(
            nm: float,
            _s: Any = strategy,
            _nb: int = num_bins,
            _ns: int = n_steps,
        ) -> Any:
            return _ftrl_balls_in_bins(mf_gaussian(nm, _s), num_bins=_nb, n_steps=_ns)

    else:
        raise ValueError(
            f"sampling_mode={sampling_mode!r} has no DP-FTRL amplifier "
            f"configured.  Valid: 'poisson', 'b_min_sep', 'balls_in_bins'."
        )
    return amp


def build_step_mechanism_factory(
    raw_amplifier_factory: Callable[[float], Any],
) -> Callable[[float], Any]:
    """Wrap ``nm → DpHorizonProcess`` with ``per_step`` for ``acc |= step``.

    The wrapped per-step view materialises as the true K-step PLD of the
    deployed N-step mechanism on its ``K``-prefix, so
    ``Repeated(per_step(proc), K).pld()`` equals
    ``proc.pld_at(K)``.  Lets the DP-FTRL training loop use the
    same accountant composition idiom as DP-SGD.
    """

    def step(nm: float, _f: Callable[[float], Any] = raw_amplifier_factory) -> Any:
        return per_step(_f(nm))

    return step


def build_sampler(
    *,
    sampling_mode: str,
    dataset: Any,
    sample_rate: float,
    n_steps: int,
    key: RngKey,
    sampling_kwargs: dict[str, Any] | None,
    mf: MFContext | None,
    noise_multiplier: float | None,
    num_bins: int,
    expected_batch_size: int,
) -> Any:
    """Construct the Opaque sampler matching ``sampling_mode``.

    Privacy-derived sampler parameters (``bands``, paper-``p``) are read
    off the built ``mf`` recipe / amplifier — never off ``sampling_kwargs``
    or ``mechanism_kwargs`` — so the runtime sampler cannot desync from
    the accountant.  ``sampling_kwargs`` carries only sampler-ergonomics
    knobs (e.g. ``truncated_batch_size`` for Poisson cap).
    """
    sk = dict(sampling_kwargs) if sampling_kwargs else {}
    if sampling_mode == "poisson":
        tb_raw = sk.get("truncated_batch_size", sk.get("max_batch_size"))
        truncated_batch_size = int(tb_raw) if tb_raw is not None else None
        return PoissonSampler(
            dataset,
            sample_rate=sample_rate,
            n_steps=n_steps,
            truncated_batch_size=truncated_batch_size,
            key=key,
        )
    if sampling_mode == "random_allocation":
        return RandomAllocationSampler(
            dataset,
            num_bins=num_bins,
            n_steps=n_steps,
            key=key,
        )
    if sampling_mode == "k_out_of_t":
        k_raw = sk.get("total_participations")
        if k_raw is None:
            raise ValueError(
                "sampling_mode='k_out_of_t' requires sampling_kwargs with "
                "'total_participations'."
            )
        total_participations = int(k_raw)
        if not 1 <= total_participations <= n_steps:
            raise ValueError(
                "total_participations must be in "
                f"[1, n_steps={n_steps}], got {total_participations}."
            )
        return KOutOfTSampler(
            dataset,
            total_participations=total_participations,
            n_steps=n_steps,
            key=key,
        )
    if sampling_mode == "b_min_sep":
        if mf is None or noise_multiplier is None:
            raise ValueError(
                "sampling_mode='b_min_sep' requires a built MFContext and a "
                "calibrated noise_multiplier; got mf=None or "
                "noise_multiplier=None."
            )
        amp = mf.amplifier_factory(noise_multiplier)
        return BMinSepSampler(
            dataset,
            bands=int(mf.strategy.bands),
            sampling_prob=float(amp.sampling_prob),
            n_steps=n_steps,
            key=key,
        )
    if sampling_mode == "balls_in_bins":
        return BallsInBinsSampler(
            dataset,
            num_bins=num_bins,
            n_steps=n_steps,
            key=key,
        )
    if sampling_mode == "cyclic_poisson":
        if mf is None:
            raise ValueError(
                "sampling_mode='cyclic_poisson' requires a built MFContext; "
                "got mf=None."
            )
        return CyclicPoissonSampler(
            dataset,
            sample_rate=sample_rate,
            bands=int(mf.strategy.bands),
            n_steps=n_steps,
            key=key,
        )
    if sampling_mode == "sequential":
        return SequentialBatchSampler(dataset, batch_size=expected_batch_size)
    raise ValueError(f"Unknown sampling_mode {sampling_mode!r}")


__all__ = [
    "MFContext",
    "build_amplifier_factory",
    "build_sampler",
    "build_step_mechanism_factory",
    "build_strategy",
]
