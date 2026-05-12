"""Tests for the realized-per-step σ published on ``NoisedPytree.noise_stddev``.

Bug fix: under correlated MF noise the per-coordinate variance at step
``t`` is ``base_σ² · ‖row_t(C^{-1})‖²``, not ``base_σ²`` as the original
implementation reported.  Adam-family ``noise_bias_correction`` reads
``noise_stddev`` to debias the second-moment EMA, so reporting the wrong
value silently breaks BC.

These tests pin the corrected behavior:

- Identity strategy: ``‖row_t‖ ≡ 1`` so realized σ == base σ (DP-SGD
  reduction).
- BandMF / BLT / BISR / BSR (streaming-matrix path): per-step realized
  σ matches the analytical ``base_σ · streaming.row_norms_squared(n).sqrt()[t]``.
- λ-CGD (PRNG-replay path): per-step realized σ matches the closed-form
  ``base_σ · sqrt(1+λ²) · d_t`` (with the step-0 short-circuit and the
  ``normalized=False`` simplification).
"""

from __future__ import annotations

import math

import pytest
import torch

from opaque.api.dpftrl.noise._lambda_cgd import _column_norm
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opaque.random import key
from opaque.types import clipped


def _step(
    strategy,
    *,
    n_steps,
    min_sep=1,
    max_participations=None,
    max_norm=1.0,
    nm=1.0,
    seed=0,
    n_calls=1,
):
    """Run noise_fn for ``n_calls`` steps; return the realized σ at each step."""
    template = {"w": torch.zeros(8)}
    noise_fn, state = mf_gaussian_noise(
        template,
        strategy,
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        noise_multiplier=nm,
        key=key(seed),
    )
    grads = clipped({"w": torch.zeros(8)}, max_norm=max_norm)
    realized = []
    for _ in range(n_calls):
        out, state = noise_fn(grads, state)
        realized.append(float(out.noise_stddev))
    return realized


def _row_l2_at_zero(strategy, *, n_steps, min_sep=1, max_participations=None) -> float:
    """First-step L2 of ``C^{-1}``'s row 0 — the analytical scaling factor."""
    streaming = strategy.streaming_matrix(
        n_steps=n_steps, min_sep=min_sep, max_participations=max_participations
    )
    return float(streaming.row_norms_squared(n_steps).clamp_min(0.0).sqrt()[0])


class TestIdentityReducesToDPSGD:
    """Identity strategy ⇒ uncorrelated noise; realized σ == base σ at every
    step.  This is the DP-SGD reduction; matches the standard BC formula
    against ``gaussian_noise`` exactly."""

    def test_realized_equals_base(self):
        # base σ = noise_multiplier · max_norm = 1.5 · 0.7 = 1.05
        sigmas = _step(identity_strategy(), n_steps=10, nm=1.5, max_norm=0.7, n_calls=4)
        for s in sigmas:
            assert s == pytest.approx(1.5 * 0.7, rel=1e-12)


class TestStreamingMatrixRealizedSigma:
    """Streaming-matrix MF strategies: realized σ = base σ · row_l2(t)."""

    def _build(self, strategy, n_steps, min_sep, max_participations):
        template = {"w": torch.zeros(8)}
        return mf_gaussian_noise(
            template,
            strategy,
            n_steps=n_steps,
            min_sep=min_sep,
            max_participations=max_participations,
            noise_multiplier=1.0,
            key=key(0),
        )

    @pytest.mark.parametrize(
        "make_strategy,part",
        [
            (
                lambda: band_mf_strategy(bands=4, momentum=0.9),
                dict(n_steps=20, min_sep=1, max_participations=20),
            ),
            (
                lambda: blt_strategy(momentum=0.9),
                dict(n_steps=20, min_sep=4, max_participations=5),
            ),
            (
                lambda: bisr_strategy(bandwidth=4, momentum=0.5),
                dict(n_steps=20, min_sep=4, max_participations=5),
            ),
            (
                lambda: bsr_strategy(bandwidth=4, alpha=1.0, beta=0.5),
                dict(n_steps=20, min_sep=4, max_participations=5),
            ),
        ],
        ids=["band_mf", "blt", "bisr", "bsr"],
    )
    def test_realized_matches_row_l2(self, make_strategy, part):
        strategy = make_strategy()
        # Analytical row-0 L2 norm of C^{-1}.
        expected_row_l2 = _row_l2_at_zero(strategy, **part)
        # Run one step with base σ = 1.0 · 0.5 = 0.5.
        nm, max_norm = 1.0, 0.5
        template = {"w": torch.zeros(8)}
        noise_fn, state = mf_gaussian_noise(
            template,
            strategy,
            **part,
            noise_multiplier=nm,
            key=key(0),
        )
        grads = clipped({"w": torch.zeros(8)}, max_norm=max_norm)
        out, _ = noise_fn(grads, state)
        # Realized σ at step 0 should equal base σ · row_l2_0.
        assert out.noise_stddev == pytest.approx(
            nm * max_norm * expected_row_l2, rel=1e-9
        )


class TestLambdaCgdRealizedSigma:
    """λ-CGD (PRNG-replay): realized σ = base σ · sqrt(1+λ²) · d_t."""

    def test_normalized_step_zero(self):
        """At step 0 there is no z_{t-1} term; realized σ = base σ · d_0."""
        lam = 0.7
        n_steps = 30
        strategy = lambda_cgd_strategy(lambda_=lam, normalized=True)
        d_0 = _column_norm(lam, n_steps, 0)
        nm, max_norm = 1.5, 0.4
        sigmas = _step(
            strategy,
            n_steps=n_steps,
            min_sep=1,
            max_participations=1,
            nm=nm,
            max_norm=max_norm,
            n_calls=1,
        )
        assert sigmas[0] == pytest.approx(nm * max_norm * d_0, rel=1e-12)

    def test_normalized_step_one(self):
        """At step ≥ 1: realized σ = base σ · sqrt(1+λ²) · d_t."""
        lam = 0.7
        n_steps = 30
        strategy = lambda_cgd_strategy(lambda_=lam, normalized=True)
        nm, max_norm = 1.5, 0.4
        sigmas = _step(
            strategy,
            n_steps=n_steps,
            min_sep=1,
            max_participations=1,
            nm=nm,
            max_norm=max_norm,
            n_calls=2,
        )
        d_1 = _column_norm(lam, n_steps, 1)
        expected = nm * max_norm * math.sqrt(1.0 + lam * lam) * d_1
        assert sigmas[1] == pytest.approx(expected, rel=1e-12)

    def test_unnormalized_step_one(self):
        """Unnormalized: no d_t factor; just base σ · sqrt(1+λ²) at step ≥ 1."""
        lam = 0.5
        strategy = lambda_cgd_strategy(lambda_=lam, normalized=False)
        nm, max_norm = 1.0, 1.0
        sigmas = _step(
            strategy,
            n_steps=20,
            min_sep=1,
            max_participations=1,
            nm=nm,
            max_norm=max_norm,
            n_calls=2,
        )
        # Step 0: just base σ (no previous-step term).
        assert sigmas[0] == pytest.approx(nm * max_norm, rel=1e-12)
        # Step 1: base σ · sqrt(1 + λ²).
        assert sigmas[1] == pytest.approx(
            nm * max_norm * math.sqrt(1.0 + lam * lam), rel=1e-12
        )

    def test_lambda_zero_reduces_to_iid(self):
        """λ=0 ⇒ no correlation; realized σ = base σ regardless of step."""
        strategy = lambda_cgd_strategy(lambda_=0.0, normalized=False)
        nm, max_norm = 1.2, 0.8
        sigmas = _step(
            strategy,
            n_steps=20,
            min_sep=1,
            max_participations=1,
            nm=nm,
            max_norm=max_norm,
            n_calls=3,
        )
        for s in sigmas:
            assert s == pytest.approx(nm * max_norm, rel=1e-12)
