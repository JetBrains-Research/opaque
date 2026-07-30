"""b-min-sep subsampling amplification for BandMF (warm-start, Monte Carlo PLD).

Dong & Ganesh, "Privacy Amplification for BandMF via b-Min-Sep Subsampling"
(arXiv:2602.09338). Uses Monte Carlo accounting with the paper's dynamic
program for the likelihood ratio (Section 5).

The runtime sampler should use :class:`opaque.dpftrl.sampling.BMinSepSampler`
with the same ``bands`` and ``p`` as the accountant.  Read these off the
:class:`BMinSep` instance: ``bands`` via ``inner.strategy.bands`` (or the
equivalent ``min_sep`` property) and ``p`` via ``sampling_prob``.  The
conversion ``p = p_0 / (1 - p_0 * (bands - 1))`` is encapsulated by
:func:`participation_p_from_per_example_rate`, so runtime callers never
duplicate the formula.
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

from ._transcript_cache import with_handle as _with_transcript_handle

_Inner = MfGaussian


def participation_p_from_per_example_rate(p0: float, bands: int) -> float:
    """Paper: ``p = p_0 / (1 - p_0 * (b - 1))``.

    Converts a target per-example participation rate ``p_0 = E[batch] / |D|``
    into the paper's per-iteration inclusion probability ``p`` that
    :class:`opaque.dpftrl.sampling.BMinSepSampler` expects.  ``bands == 1``
    degenerates to ``p_0`` (plain Poisson, no min-sep constraint).
    """
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
        # period).  ``per_step(self) * K`` rounds K up to a band boundary.
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

    @property
    def sampling_prob(self) -> float:
        # The runtime ``BMinSepSampler`` parameter — the paper's per-iteration
        # inclusion probability ``p``.  Equal to ``self.p0`` only when
        # ``bands == 1``; the conversion belongs with the privacy proof so
        # callers (runtime sampler builders) never re-derive it.
        return participation_p_from_per_example_rate(self.p0, self.inner.strategy.bands)

    @functools.lru_cache(maxsize=8)
    def _pld_at_horizon(
        self,
        n_steps: int,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        num_mc_samples: int | None = None,
        seed: int | None = None,
    ) -> Pld:
        """K-step warm-start b-min-sep MC PLD using N-tuned coefficients.

        ``n_steps`` is rounded up to the next ``bands`` boundary (capped at
        ``self.n_steps``) — within an atomic band the PLD plateaus.  The
        BandMF strategy coefficients and per-example sensitivity are
        evaluated at ``self.n_steps`` (the N-tuned deployed mechanism);
        ``n_steps`` controls only the warm-start MC transcript length.
        The Rust sampler is prefix-stable under a fixed seed, so the
        K-row evaluation is the post-processing projection of the
        N-step output — ``ε(_pld_at_horizon(K)) ≤ ε(self)`` and is
        monotone in K.
        """
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
        if n_steps <= 0 or n_steps > self.n_steps:
            raise ValueError(f"n_steps ({n_steps}) must be in [1, {self.n_steps}]")
        rounded = min(-(-n_steps // bands) * bands, self.n_steps)

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            num_mc_samples=num_mc_samples,
            seed=seed,
        )
        native_cfg = config.to_native()

        coefs = s.coefficients(n_steps=self.n_steps).tolist()
        sensitivity = s.sensitivity(n_steps=self.n_steps)
        effective_nm = self.inner.noise_multiplier / sensitivity
        p = self.sampling_prob

        # Always look up the cached corpus at ``self.n_steps`` (the full
        # horizon at which it was prepared).  For K < N the Rust side
        # slices each sample to the first K columns, which — because
        # within a single sample the per-step RNG state is deterministic
        # in the columns up to it — is byte-identical to a freshly-prepared
        # K-row transcript at the same per-sample seed.  The K-step PLD
        # is therefore a deterministic post-processing of the N-step
        # corpus, and ``ε(_pld_at_horizon(K)) ≤ ε(self)`` holds exactly
        # (no MC variance gap).
        #
        # ``_with_transcript_handle`` holds the per-cache lock around
        # both the lookup and the Rust call so a concurrent
        # ``_clear_all_native_caches()`` (e.g. from ``calibrate()``'s
        # finally clause on another thread) cannot drop the corpus
        # mid-use; on cache-miss it returns ``None`` and we fall through
        # to a fresh K-row warm MC (which loses the prefix-projection
        # property but is still a valid PLD at K).
        result = _with_transcript_handle(
            tuple(coefs),
            self.n_steps,
            p,
            config.num_mc_samples,
            config.seed,
            lambda hid: _native.bandmf_b_min_sep_pld_from_transcript_handle(
                hid,
                coefs,
                rounded,
                p,
                effective_nm,
                native_cfg,
            ),
        )
        if result is not None:
            return result
        return _native.bandmf_b_min_sep_warm_mc_pld(
            coefs,
            rounded,
            p,
            effective_nm,
            native_cfg,
        )

    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        num_mc_samples: int | None = None,
        seed: int | None = None,
    ) -> Pld:
        return self._pld_at_horizon(
            self.n_steps,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            num_mc_samples=num_mc_samples,
            seed=seed,
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
