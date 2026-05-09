"""b-min-sep subsampling amplification for BandMF (warm-start, Monte Carlo PLD).

Dong & Ganesh, "Privacy Amplification for BandMF via b-Min-Sep Subsampling"
(arXiv:2602.09338). Uses Monte Carlo accounting with the paper's dynamic
program for the likelihood ratio (Section 5).

The runtime sampler should use :class:`opaque.dpftrl.sampling.BMinSepSampler`
with the same ``bands`` and ``p`` derived from the target per-example
participation rate ``p_0`` via ``p = p_0 / (1 - p_0 * (bands - 1))`` for
``bands > 1``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.accounting import _native

from opaque.accounting._base import DpProcess, Pld
from opaque.accounting.discretization import get_discretization
from opaque.dpftrl.accounting.mechanisms._band_mf import BandMf

from ._b_min_sep_transcript_cache import get_handle_or_none

_Inner = BandMf


def _participation_p_from_per_example_rate(p0: float, bands: int) -> float:
    """Paper: p = p_0 / (1 - p_0 * (b - 1))."""
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"per-example rate p_0 must be in (0, 1), got {p0}")
    if bands < 1:
        raise ValueError(f"bands must be >= 1, got {bands}")
    if bands == 1:
        return p0
    denom = 1.0 - p0 * (bands - 1)
    if denom <= 0:
        raise ValueError(
            f"infeasible p_0={p0} for bands={bands}: need p_0 < 1/(bands-1)"
        )
    return p0 / denom


@dataclass(frozen=True, slots=True)
class BMinSep(DpProcess):
    """Monte Carlo PLD for BandMF + warm-start b-min-sep subsampling."""

    inner: _Inner
    n_steps: int
    p0: float

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
        num_mc_samples: int | None = None,
        seed: int | None = None,
    ) -> Pld:
        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
            num_mc_samples=num_mc_samples,
            seed=seed,
        )
        native_cfg = config.to_native()

        match self.inner:
            case BandMf():
                strategy_coefficients = self.inner.coefficients
                bands = self.inner.bands
                effective_nm = self.inner.noise_multiplier / self.inner.sensitivity
            case _:
                raise TypeError(
                    "b_min_sep requires a BandMf inner, got "
                    f"{type(self.inner).__name__}."
                )

        if bands < 1:
            raise ValueError(
                "BandMf inner must have non-empty coefficients (bands >= 1)."
            )

        p = _participation_p_from_per_example_rate(self.p0, bands)

        hid = get_handle_or_none(
            strategy_coefficients,
            self.n_steps,
            p,
            config.num_mc_samples,
            config.seed,
        )
        if hid is None:
            return _native.bandmf_b_min_sep_warm_mc_pld(
                list(strategy_coefficients),
                self.n_steps,
                p,
                effective_nm,
                native_cfg,
            )
        return _native.bandmf_b_min_sep_pld_from_transcript_handle(
            hid,
            list(strategy_coefficients),
            self.n_steps,
            p,
            effective_nm,
            native_cfg,
        )


def b_min_sep(
    inner: _Inner,
    *,
    n_steps: int,
    p0: float,
) -> BMinSep:
    """BandMF privacy accounting under warm-start b-min-sep subsampling.

    Args:
        inner: ``BandMf`` mechanism — strategy coefficients (and band width)
            are read from ``inner.coefficients``.
        n_steps: Total number of training iterations ``n``.
        p0: Per-example participation rate per iteration
            (``E[batch] / |D|``). Same ``p_0`` as cyclic Poisson /
            batch-size accounting.

    Returns:
        A :class:`BMinSep` process (asymmetric PLD from Monte Carlo).
    """
    match inner:
        case BandMf():
            pass
        case _:
            raise TypeError(
                f"b_min_sep() requires a BandMf inner, got {type(inner).__name__}."
            )
    if inner.bands < 1:
        raise ValueError("BandMf inner must have non-empty coefficients (bands >= 1).")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    return BMinSep(
        inner=inner,
        n_steps=n_steps,
        p0=float(p0),
    )
