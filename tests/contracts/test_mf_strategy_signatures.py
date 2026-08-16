"""Behavioral tests for MF strategy factories and noise construction.

Verifies that strategy factories produce strategies with finite, positive
``sensitivity()`` and that ``mf_gaussian_noise()`` returns a working
callable+state pair.

Regression target: issue #416 (docs once used wrong signatures).
"""

from __future__ import annotations

import torch

from opaque.backend import ensure_backend
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
from opaque.types import ClippedPytree


# ---------------------------------------------------------------------------
# Strategy factories return objects with valid sensitivity
# ---------------------------------------------------------------------------


class TestStrategyOutputs:
    """Strategies produce valid sensitivity bounds.

    sensitivity() is a method (not a property) that takes participation
    context: ``n_steps`` (required) plus ``min_sep`` / ``max_participations``
    for strategies that need them (BiSR, BSR, λCGD). BandMF and identity
    only need ``n_steps``.
    """

    _PART = dict(n_steps=100, min_sep=10, max_participations=5)
    _PART_BAND = dict(n_steps=100)

    def test_band_mf_sensitivity(self) -> None:
        sens = band_mf_strategy(bands=8).sensitivity(**self._PART_BAND)
        assert 0 < sens < float("inf"), f"got {sens}"

    def test_blt_sensitivity(self) -> None:
        sens = blt_strategy(max_buffers=10).sensitivity(**self._PART)
        assert 0 < sens < float("inf"), f"got {sens}"

    def test_bisr_sensitivity(self) -> None:
        sens = bisr_strategy(bandwidth=4).sensitivity(**self._PART)
        assert 0 < sens < float("inf"), f"got {sens}"

    def test_bsr_sensitivity(self) -> None:
        sens = bsr_strategy(bandwidth=4, alpha=0.9, beta=0.0).sensitivity(**self._PART)
        assert 0 < sens < float("inf"), f"got {sens}"

    def test_lambda_cgd_sensitivity(self) -> None:
        sens = lambda_cgd_strategy(lambda_=0.9).sensitivity(**self._PART)
        assert 0 < sens < float("inf"), f"got {sens}"

    def test_identity_sensitivity(self) -> None:
        sens = identity_strategy().sensitivity(**self._PART_BAND)
        assert 0 < sens < float("inf"), f"got {sens}"


# ---------------------------------------------------------------------------
# mf_gaussian_noise returns callable + state tuple
# ---------------------------------------------------------------------------


class TestMfGaussianNoiseOutput:
    """mf_gaussian_noise returns (callable, state)."""

    def test_returns_callable_and_state(self) -> None:
        strategy = band_mf_strategy(bands=4)
        grad_template = (torch.randn(10), torch.randn(5))

        result = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=100,
            noise_multiplier=1.0,
            key=key(0),
        )
        assert isinstance(result, tuple) and len(result) == 2
        noise_fn, state = result
        assert callable(noise_fn), "first return value must be a callable"

    def test_noise_fn_produces_valid_output(self) -> None:
        """Noise function produces finite values with correct shapes."""
        strategy = band_mf_strategy(bands=4)
        grad_template = (torch.randn(10), torch.randn(5))
        ensure_backend(grad_template)

        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=100,
            noise_multiplier=1.0,
            key=key(0),
        )

        grads = ClippedPytree(
            (torch.randn(10), torch.randn(5)),
            max_norm=1.0,
        )
        noisy, new_state = noise_fn(grads, state)

        assert noisy.pytree is not None
        assert len(noisy.pytree) == len(grad_template)
        for orig, noised in zip(grad_template, noisy.pytree):
            assert noised.shape == orig.shape
            assert noised.isfinite().all()
