"""DP-FTRL integration helpers for :class:`DPTrainer`.

Pure factories: MF strategy construction, accountant amplifier
construction, sampler dispatch.  No state lives here — the trainer
owns the lifecycle.  The trainer calls these in :meth:`_setup_training`
(strategy + amplifier) and :meth:`get_train_dataloader` (sampler) when
:attr:`TrainingArguments.privacy_noise_mechanism` starts with ``"mf_"``.

The MF amplifier returned by :func:`build_amplifier_factory` is the
whole-process accountant (a :class:`DpHorizonProcess`); the trainer queries
``(n_steps, min_sep, max_participations)`` off it to build the matching
:func:`opaque.dpftrl.noise.mf_gaussian_noise` and accounts that process once.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from opaque.api.accounting.dpftrl.amplification._b_min_sep import BMinSep
from opaque.api.accounting.dpftrl.amplification._balls_in_bins import BallsInBins
from opaque.api.accounting.dpftrl.amplification._poisson import CyclicPoisson
from opaque.dpftrl import (
    BallsInBinsSampler,
    BMinSepSampler,
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
from opaque.dpftrl.noise.types import (
    BandMfStrategy,
    BisrStrategy,
    BltStrategy,
    BsrStrategy,
    IdentityStrategy,
    LambdaCgdStrategy,
)
from opaque.dpsgd.sampling import (
    KOutOfTSampler,
    PoissonSampler,
)
from opaque.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.random.types import RngKey

    from ._participation import ResolvedParticipationPlan


@dataclasses.dataclass(frozen=True)
class MFContext:
    """MF strategy and amplifier bound to one immutable participation plan."""

    strategy: Any
    amplifier_factory: Callable[[float], Any]
    participation_plan: ResolvedParticipationPlan

    def __post_init__(self) -> None:
        if self.participation_plan.mechanism_kind == "gaussian":
            raise ConfigurationError(
                *("MFContext cannot carry a gaussian participation plan.",)
            )


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
_STRATEGY_TYPES = {
    "mf_band": BandMfStrategy,
    "mf_blt": BltStrategy,
    "mf_bisr": BisrStrategy,
    "mf_bsr": BsrStrategy,
    "mf_lambda_cgd": LambdaCgdStrategy,
    "mf_identity": IdentityStrategy,
}


def _validate_strategy_for_plan(
    plan: ResolvedParticipationPlan,
    strategy: Any,
) -> None:
    expected = _STRATEGY_TYPES.get(plan.mechanism_kind)
    if expected is None or not isinstance(strategy, expected):
        expected_name = expected.__name__ if expected is not None else "no MF strategy"
        raise ConfigurationError(
            *(
                "MF strategy does not match the resolved participation plan: "
                f"mechanism={plan.mechanism_kind!r} expects {expected_name}, got "
                f"{type(strategy).__name__}.",
            )
        )


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
    plan: ResolvedParticipationPlan,
    strategy: Any,
) -> Callable[[float], Any]:
    """Return ``nm → DpHorizonProcess`` — the *raw* amplifier instance.

    The trainer holds the result on its training context and queries it
    with the calibrated noise multiplier at noise-function construction
    time to read off ``(n_steps, min_sep, max_participations)``.
    """
    sampling_mode = plan.sampling_mode
    if sampling_mode == "poisson":
        if not isinstance(strategy, IdentityStrategy):
            raise ConfigurationError(
                *(
                    "plain whole-dataset Poisson is only supported with MF identity; "
                    "BandMF requires the b_min_sep participation contract.",
                )
            )
        _validate_strategy_for_plan(plan, strategy)

        def amp(
            nm: float,
            _s: Any = strategy,
            _sr: float = plan.sample_rate,
            _ns: int = plan.total_steps,
            _tb: int | None = plan.sampler_kwargs_dict().get("truncated_batch_size"),
            _ds: int = plan.population_size,
        ) -> Any:
            return _ftrl_poisson(
                mf_gaussian(nm, _s),
                sample_rate=_sr,
                n_steps=_ns,
                truncated_batch_size=_tb,
                dataset_size=_ds if _tb is not None else None,
            )

    elif sampling_mode == "b_min_sep":
        if not isinstance(strategy, BandMfStrategy):
            raise ConfigurationError(
                *(
                    "the b_min_sep participation contract requires a BandMF strategy, "
                    f"got {type(strategy).__name__}.",
                )
            )
        _validate_strategy_for_plan(plan, strategy)

        def amp(
            nm: float,
            _s: Any = strategy,
            _ns: int = plan.total_steps,
            _p0: float = plan.sample_rate,
        ) -> Any:
            return _ftrl_b_min_sep(mf_gaussian(nm, _s), n_steps=_ns, p0=_p0)

    elif sampling_mode == "balls_in_bins":
        _validate_strategy_for_plan(plan, strategy)

        def amp(
            nm: float,
            _s: Any = strategy,
            _nb: int = plan.num_bins,
            _ns: int = plan.total_steps,
        ) -> Any:
            return _ftrl_balls_in_bins(mf_gaussian(nm, _s), num_bins=_nb, n_steps=_ns)

    else:
        raise ConfigurationError(
            *(
                f"sampling_mode={sampling_mode!r} has no DP-FTRL amplifier "
                f"configured.  Valid: 'poisson', 'b_min_sep', 'balls_in_bins'.",
            )
        )
    return amp


def build_sampler(
    *,
    plan: ResolvedParticipationPlan,
    dataset: Any,
    key: RngKey,
    mf: MFContext | None,
    noise_multiplier: float | None,
    process: Any | None = None,
) -> Any:
    """Construct the sampler from the accountant's resolved participation plan."""
    sampling_mode = plan.sampling_mode
    sk = plan.sampler_kwargs_dict()
    if mf is not None and mf.participation_plan != plan:
        raise ConfigurationError(
            *("MF accountant and sampler received different participation plans.",)
        )
    if sampling_mode == "poisson":
        if mf is not None and not isinstance(mf.strategy, IdentityStrategy):
            raise ConfigurationError(
                *(
                    "plain whole-dataset Poisson is only compatible with MF identity; "
                    "BandMF requires b_min_sep.",
                )
            )
        tb_raw = sk.get("truncated_batch_size", sk.get("max_batch_size"))
        truncated_batch_size = int(tb_raw) if tb_raw is not None else None
        resolved_process = (
            validate_mf_process_for_plan(
                plan,
                mf,
                noise_multiplier,
                process=process,
            )
            if mf is not None and noise_multiplier is not None
            else None
        )
        sampler = PoissonSampler(
            dataset,
            sample_rate=plan.sample_rate,
            n_steps=plan.total_steps,
            truncated_batch_size=truncated_batch_size,
            key=key,
        )
        validate_sampler_for_plan(
            plan,
            sampler,
            mf=mf,
            noise_multiplier=noise_multiplier,
            process=resolved_process,
        )
        return sampler
    if sampling_mode == "k_out_of_t":
        k_raw = sk.get("k")
        allocation = sk.get("allocation")
        if k_raw is None or allocation is None:
            raise ConfigurationError(
                *(
                    "sampling_mode='k_out_of_t' requires sampling_kwargs with "
                    "'k' and 'allocation'.",
                )
            )
        if allocation not in ("block", "total"):
            raise ConfigurationError(
                *(
                    "sampling_kwargs['allocation'] must be 'block' or 'total', got "
                    f"{allocation!r}.",
                )
            )
        sampler = KOutOfTSampler(
            dataset,
            k=int(k_raw),
            t=plan.total_steps,
            allocation=allocation,
            key=key,
        )
        validate_sampler_for_plan(plan, sampler)
        return sampler
    if sampling_mode == "b_min_sep":
        if mf is None or noise_multiplier is None:
            raise ConfigurationError(
                *(
                    "sampling_mode='b_min_sep' requires a built MFContext and a "
                    "calibrated noise_multiplier; got mf=None or "
                    "noise_multiplier=None.",
                )
            )
        amp = validate_mf_process_for_plan(
            plan,
            mf,
            noise_multiplier,
            process=process,
        )
        sampler = BMinSepSampler(
            dataset,
            bands=int(mf.strategy.bands),
            sampling_prob=float(amp.sampling_prob),
            n_steps=plan.total_steps,
            key=key,
        )
        validate_sampler_for_plan(
            plan,
            sampler,
            mf=mf,
            noise_multiplier=noise_multiplier,
            process=amp,
        )
        return sampler
    if sampling_mode == "balls_in_bins":
        if mf is None or noise_multiplier is None:
            raise ConfigurationError(
                *(
                    "sampling_mode='balls_in_bins' requires a built MFContext and "
                    "a calibrated noise_multiplier.",
                )
            )
        amp = validate_mf_process_for_plan(
            plan,
            mf,
            noise_multiplier,
            process=process,
        )
        sampler = BallsInBinsSampler(
            dataset,
            num_bins=plan.num_bins,
            n_steps=plan.total_steps,
            key=key,
        )
        validate_sampler_for_plan(
            plan,
            sampler,
            mf=mf,
            noise_multiplier=noise_multiplier,
            process=amp,
        )
        return sampler
    raise ConfigurationError(*(f"Unknown sampling_mode {sampling_mode!r}",))


def validate_mf_process_for_plan(
    plan: ResolvedParticipationPlan,
    mf: MFContext,
    noise_multiplier: float,
    *,
    process: Any | None = None,
) -> Any:
    """Return the realized MF process after checking its participation contract."""
    if mf.participation_plan != plan:
        raise ConfigurationError(
            *("MF process and sampler received different participation plans.",)
        )
    _validate_strategy_for_plan(plan, mf.strategy)
    if process is None:
        process = mf.amplifier_factory(noise_multiplier)
    expected_type = {
        "poisson": CyclicPoisson,
        "b_min_sep": BMinSep,
        "balls_in_bins": BallsInBins,
    }.get(plan.sampling_mode)
    if expected_type is None or type(process) is not expected_type:
        raise ConfigurationError(
            *(
                "MF amplifier does not realize the resolved participation contract: "
                f"mode={plan.sampling_mode!r}, process={type(process).__name__}.",
            )
        )
    if process.inner.strategy is not mf.strategy:
        raise ConfigurationError(
            *("MF amplifier does not contain the resolved strategy instance.",)
        )
    if int(process.n_steps) != plan.total_steps:
        raise ConfigurationError(
            *("MF amplifier horizon does not match the participation plan.",)
        )
    if plan.sampling_mode == "poisson":
        expected_dataset_size = (
            plan.population_size
            if plan.sampler_kwargs_dict().get("truncated_batch_size") is not None
            else None
        )
        if (
            not isinstance(mf.strategy, IdentityStrategy)
            or float(process.sample_rate) != plan.sample_rate
            or process.truncated_batch_size
            != plan.sampler_kwargs_dict().get("truncated_batch_size")
            or process.dataset_size != expected_dataset_size
        ):
            raise ConfigurationError(
                *("MF Poisson amplifier does not match the participation plan.",)
            )
    elif plan.sampling_mode == "b_min_sep":
        if (
            not isinstance(mf.strategy, BandMfStrategy)
            or float(process.p0) != plan.sample_rate
            or int(process.min_sep) != int(mf.strategy.bands)
        ):
            raise ConfigurationError(
                *("BandMF amplifier does not match the b-min-separation plan.",)
            )
    elif int(process.num_bins) != plan.num_bins:
        raise ConfigurationError(
            *("Balls-in-Bins amplifier does not match the participation plan.",)
        )
    return process


def validate_sampler_for_plan(
    plan: ResolvedParticipationPlan,
    sampler: Any,
    *,
    mf: MFContext | None = None,
    noise_multiplier: float | None = None,
    process: Any | None = None,
) -> None:
    """Validate the live sampler and MF process against one resolved plan."""
    kwargs = plan.sampler_kwargs_dict()
    if plan.sampling_mode == "poisson":
        if mf is not None:
            if noise_multiplier is None:
                raise ConfigurationError(
                    *("MF Poisson validation requires a noise multiplier.",)
                )
            if process is None:
                process = validate_mf_process_for_plan(
                    plan,
                    mf,
                    noise_multiplier,
                )
        valid = (
            type(sampler) is PoissonSampler
            and sampler.sample_rate == plan.sample_rate
            and sampler.n_steps == plan.total_steps
            and sampler._num_samples == plan.local_population_size
            and sampler.truncated_batch_size == kwargs.get("truncated_batch_size")
            and (
                mf is None
                or (
                    type(process) is CyclicPoisson
                    and process.sample_rate == plan.sample_rate
                )
            )
        )
    elif plan.sampling_mode == "k_out_of_t":
        valid = (
            type(sampler) is KOutOfTSampler
            and sampler.k == kwargs["k"]
            and sampler.t == plan.total_steps
            and sampler.allocation == kwargs["allocation"]
            and sampler._num_samples == plan.local_population_size
        )
    elif plan.sampling_mode == "b_min_sep":
        if mf is None or noise_multiplier is None:
            raise ConfigurationError(
                *("b-min-separation validation requires the realized MF process.",)
            )
        if process is None:
            process = validate_mf_process_for_plan(
                plan,
                mf,
                noise_multiplier,
            )
        valid = (
            type(sampler) is BMinSepSampler
            and sampler.num_examples == plan.local_population_size
            and sampler.n_steps == plan.total_steps
            and sampler.bands == process.min_sep
            and sampler.sampling_prob == process.sampling_prob
        )
    elif plan.sampling_mode == "balls_in_bins":
        if mf is None or noise_multiplier is None:
            raise ConfigurationError(
                *("Balls-in-Bins validation requires the realized MF process.",)
            )
        if process is None:
            process = validate_mf_process_for_plan(
                plan,
                mf,
                noise_multiplier,
            )
        valid = (
            type(sampler) is BallsInBinsSampler
            and sampler._num_samples == plan.local_population_size
            and sampler.n_steps == plan.total_steps
            and sampler.num_bins == process.num_bins == plan.num_bins
        )
    else:
        valid = False
    if not valid:
        raise ConfigurationError(
            *(
                "live sampler does not realize the resolved participation plan: "
                f"mode={plan.sampling_mode!r}, sampler={type(sampler).__name__}.",
            )
        )


__all__ = [
    "MFContext",
    "build_amplifier_factory",
    "build_sampler",
    "build_strategy",
    "validate_mf_process_for_plan",
    "validate_sampler_for_plan",
]
