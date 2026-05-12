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

from opaque.api.accounting.core import _native

from opaque.api.accounting.core._base import Pld
from opaque.api.accounting.core.discretization import get_discretization
from opaque.api.accounting.dpftrl._base import DpFtrlProcess
from opaque.api.accounting.dpftrl.mechanisms._mf_gaussian import MfGaussian
from opaque.api.dpftrl.noise._band_mf import BandMfStrategy

from ._b_min_sep_transcript_cache import get_handle_or_none

_Inner = MfGaussian


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
class BMinSep(DpFtrlProcess):
    """Monte Carlo PLD for BandMF + warm-start b-min-sep subsampling."""

    inner: _Inner
    n_steps: int
    p0: float

    @property
    def atomic_unit(self) -> int:
        # b-min-sep enforces one user contribution per ``bands``-row window;
        # the warm-start MC handles arbitrary ``n_steps`` natively, but the
        # accounting-meaningful quantum is one band (one full participation
        # period).  ``approx_at_step`` rounds up to a band boundary.
        return self.inner.strategy.bands

    @property
    def min_sep(self) -> int:
        # b-min-sep contract: each example participates at most once per
        # ``bands``-row window ⇒ min separation = ``bands``.
        return self.inner.strategy.bands

    @property
    def max_participations(self) -> int:
        # At most one participation per ``bands``-row window across
        # ``n_steps``.  Use ``ceil(n_steps / bands)``: a window starts at the
        # first user contribution and may extend past ``n_steps`` mid-window,
        # so the worst case is ``ceil`` not ``floor`` (e.g. bands=4,
        # n_steps=10 yields 3, not 2).
        bands = self.inner.strategy.bands
        return (self.n_steps + bands - 1) // bands

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

        s = self.inner.strategy
        if not isinstance(s, BandMfStrategy):
            raise TypeError(
                "b_min_sep requires inner.strategy to be BandMfStrategy, got "
                f"{type(s).__name__}."
            )
        bands = s.bands
        if bands < 1:
            raise ValueError(
                "BandMfStrategy inner must have non-empty coefficients (bands >= 1)."
            )

        coefs = s.coefficients(n_steps=self.n_steps).tolist()
        sensitivity = s.sensitivity(n_steps=self.n_steps)
        effective_nm = self.inner.noise_multiplier / sensitivity
        p = _participation_p_from_per_example_rate(self.p0, bands)

        hid = get_handle_or_none(
            tuple(coefs),
            self.n_steps,
            p,
            config.num_mc_samples,
            config.seed,
        )
        if hid is None:
            return _native.bandmf_b_min_sep_warm_mc_pld(
                coefs,
                self.n_steps,
                p,
                effective_nm,
                native_cfg,
            )
        return _native.bandmf_b_min_sep_pld_from_transcript_handle(
            hid,
            coefs,
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
        inner: ``mf_gaussian(nm, BandMfStrategy(...))`` — strategy
            coefficients (and band width) are read from
            ``inner.strategy.coefficients``.
        n_steps: Total number of training iterations ``n``.
        p0: Per-example participation rate per iteration
            (``E[batch] / |D|``).  Same ``p_0`` as cyclic Poisson /
            batch-size accounting.

    Returns:
        A :class:`BMinSep` process (asymmetric PLD from Monte Carlo).
    """
    if not isinstance(inner, MfGaussian):
        raise TypeError(
            f"b_min_sep() requires an MfGaussian inner, got {type(inner).__name__}."
        )
    if not isinstance(inner.strategy, BandMfStrategy):
        raise TypeError(
            "b_min_sep() requires inner.strategy to be BandMfStrategy, got "
            f"{type(inner.strategy).__name__}."
        )
    if inner.strategy.bands < 1:
        raise ValueError(
            "BandMfStrategy inner must have non-empty coefficients (bands >= 1)."
        )
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    return BMinSep(
        inner=inner,
        n_steps=n_steps,
        p0=float(p0),
    )
