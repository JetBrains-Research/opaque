"""DP-FTRL integration helpers for :class:`DPTrainer`.

Pure factories: MF strategy construction, accountant amplifier
construction, sampler dispatch.  No state lives here — the trainer
owns the lifecycle.  The trainer calls these in :meth:`_setup_training`
(strategy + amplifier) and :meth:`get_train_dataloader` (sampler) when
:attr:`TrainingArguments.privacy_noise_mechanism` starts with ``"mf_"``.

The MF amplifier returned by :func:`build_amplifier_factory` is the
*raw* whole-process accountant (a :class:`DpFtrlProcess`); the trainer
queries ``(n_steps, min_sep, max_participations)`` off it to build the
matching :func:`opaque.dpftrl.noise.mf_gaussian_noise` and then wraps
it with :func:`opaque.dpftrl.accounting.per_step` for the
``acc |= step`` composition idiom.  See
:func:`build_step_mechanism_factory`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
    balls_in_bins as _ftrl_balls_in_bins,
    b_min_sep as _ftrl_b_min_sep,
    mf_gaussian,
    per_step,
    poisson as _ftrl_poisson,
)
from opaque.dpsgd.sampling import PoissonSampler
from opaque.random.types import RngKey

# Strategy factory dispatch — keyed by ``privacy_noise_mechanism``.
_STRATEGY_FACTORIES: dict[str, Callable[..., Any]] = {
    "mf_band": band_mf_strategy,
    "mf_blt": blt_strategy,
    "mf_bisr": bisr_strategy,
    "mf_bsr": bsr_strategy,
    "mf_lambda_cgd": lambda_cgd_strategy,
    "mf_identity": identity_strategy,
}

def build_strategy(
    mechanism: str,
    kwargs: dict[str, Any] | None,
) -> Any:
    """Build the MF strategy recipe for ``mechanism``.

    ``kwargs`` are forwarded verbatim to the strategy factory; the
    trainer pre-merges its per-mechanism defaults in
    ``TrainingArguments.__post_init__``.

    The optimizer LR schedule is *not* auto-injected.  BandMF / BLT
    accept a ``lr_schedule`` kwarg for workload-aware Toeplitz tuning,
    but :mod:`opaque.serialization` rejects callable strategy fields
    (see ``opaque.api.dpftrl.noise._strategy_codec._to_wire``), so an
    auto-injected schedule would break ``accountant.json`` save / load
    and the ``dp_state.pt`` resume.  Users who explicitly opt into
    ``lr_schedule=...`` via ``privacy_noise_mechanism_kwargs`` take on
    the checkpoint-incompatibility (the strategy must be re-supplied on
    resume; not currently surfaced).
    """
    factory = _STRATEGY_FACTORIES[mechanism]
    if mechanism == "mf_identity":
        return factory()
    extra = dict(kwargs) if kwargs else {}
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
    """Return ``nm → DpFtrlProcess`` — the *raw* amplifier instance.

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
            return _ftrl_balls_in_bins(
                mf_gaussian(nm, _s), num_bins=_nb, n_steps=_ns
            )

    else:
        raise ValueError(
            f"sampling_mode={sampling_mode!r} has no DP-FTRL amplifier "
            f"configured.  Valid: 'poisson', 'b_min_sep', 'balls_in_bins'."
        )
    return amp


def build_step_mechanism_factory(
    raw_amplifier_factory: Callable[[float], Any],
) -> Callable[[float], Any]:
    """Wrap ``nm → DpFtrlProcess`` with ``per_step`` for ``acc |= step``.

    The wrapped per-step view materialises as the true K-step PLD of the
    deployed N-step mechanism on its ``K``-prefix, so
    ``Repeated(per_step(proc), K).pld()`` equals
    ``proc._pld_at_horizon(K)``.  Lets the DP-FTRL training loop use the
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
    mechanism_kwargs: dict[str, Any] | None,
    num_bins: int,
    expected_batch_size: int,
) -> Any:
    """Construct the Opaque sampler matching ``sampling_mode``.

    ``bands`` is sourced from ``mechanism_kwargs`` (the canonical
    ``privacy_noise_mechanism_kwargs['bands']`` for BandMF) with a
    ``sampling_kwargs['bands']`` fallback for power users who want to
    decouple sampler bands from mechanism bands.
    """
    sk = dict(sampling_kwargs) if sampling_kwargs else {}
    mk = dict(mechanism_kwargs) if mechanism_kwargs else {}
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
    if sampling_mode == "b_min_sep":
        bands = int(mk.get("bands", sk.get("bands", 1)))
        sampling_prob = float(sk.get("sampling_prob", sample_rate))
        return BMinSepSampler(
            dataset,
            bands=bands,
            sampling_prob=sampling_prob,
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
        bands = int(mk.get("bands", sk.get("bands", 1)))
        return CyclicPoissonSampler(
            dataset,
            sample_rate=sample_rate,
            bands=bands,
            n_steps=n_steps,
            key=key,
        )
    if sampling_mode == "sequential":
        return SequentialBatchSampler(dataset, batch_size=expected_batch_size)
    raise ValueError(f"Unknown sampling_mode {sampling_mode!r}")


__all__ = [
    "build_strategy",
    "build_amplifier_factory",
    "build_step_mechanism_factory",
    "build_sampler",
]
