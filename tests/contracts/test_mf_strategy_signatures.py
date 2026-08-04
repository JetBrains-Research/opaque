"""Contract tests for MF strategy factory signatures.

Ensures the documented API contract between strategy creation and noise
function construction:

- ``*_strategy()`` factories accept only static workload knobs (e.g.
  ``bands``, ``max_buffers``, ``lambda_``) — NOT ``n_steps``.
- ``mf_gaussian_noise()`` requires ``n_steps`` as a keyword-only argument.
- Returned strategies expose a finite, positive ``sensitivity()`` method.

Regression target: issue #416 (docs once used wrong signatures).
"""

from __future__ import annotations

import inspect
import typing

import torch

from opaque.dpftrl.noise import (
    band_mf_strategy,
    blt_strategy,
    bisr_strategy,
    bsr_strategy,
    lambda_cgd_strategy,
    identity_strategy,
    mf_gaussian_noise,
)
from opaque.random import key
from opaque.types import ClippedPytree


def _keyword_only_params(func: typing.Callable) -> set[str]:
    """Return the names of KEYWORD-ONLY parameters of *func*."""
    sig = inspect.signature(func)
    return {
        name
        for name, param in sig.parameters.items()
        if param.kind == inspect.Parameter.KEYWORD_ONLY
    }


# ---------------------------------------------------------------------------
# Strategy factories must NOT accept n_steps
# ---------------------------------------------------------------------------

_STRATEGY_FACTORIES = [
    ("band_mf_strategy", band_mf_strategy),
    ("blt_strategy", blt_strategy),
    ("bisr_strategy", bisr_strategy),
    ("bsr_strategy", bsr_strategy),
    ("lambda_cgd_strategy", lambda_cgd_strategy),
    ("identity_strategy", identity_strategy),
]


class TestStrategyFactoriesNoNSteps:
    """Strategy factories should not accept n_steps."""

    def test_no_n_steps_in_any_strategy(self) -> None:
        for name, factory in _STRATEGY_FACTORIES:
            params = set(inspect.signature(factory).parameters)
            assert (
                "n_steps" not in params
            ), f"{name}() must not accept n_steps (belongs at mf_gaussian_noise)"

    def test_no_min_sep_in_any_strategy(self) -> None:
        for name, factory in _STRATEGY_FACTORIES:
            params = set(inspect.signature(factory).parameters)
            assert (
                "min_sep" not in params
            ), f"{name}() must not accept min_sep (belongs at mf_gaussian_noise)"

    def test_no_max_participations_in_any_strategy(self) -> None:
        for name, factory in _STRATEGY_FACTORIES:
            params = set(inspect.signature(factory).parameters)
            assert (
                "max_participations" not in params
            ), f"{name}() must not accept max_participations (belongs at mf_gaussian_noise)"


# ---------------------------------------------------------------------------
# mf_gaussian_noise requires n_steps (keyword-only)
# ---------------------------------------------------------------------------


class TestMfGaussianNoise:
    """mf_gaussian_noise must require n_steps as keyword-only."""

    def test_n_steps_required(self) -> None:
        sig = inspect.signature(mf_gaussian_noise)
        assert "n_steps" in sig.parameters, "mf_gaussian_noise must accept n_steps"
        param = sig.parameters["n_steps"]
        assert param.default == inspect.Parameter.empty, (
            "n_steps must be required (no default)"
        )

    def test_n_steps_keyword_only(self) -> None:
        kwonly = _keyword_only_params(mf_gaussian_noise)
        assert (
            "n_steps" in kwonly
        ), "n_steps must be keyword-only in mf_gaussian_noise"

    def test_requires_grad_template_and_strategy(self) -> None:
        sig = inspect.signature(mf_gaussian_noise)
        for name in ("grad_template", "strategy"):
            assert name in sig.parameters
            assert (
                sig.parameters[name].default == inspect.Parameter.empty
            ), f"{name} must be required in mf_gaussian_noise"


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
        sens = bsr_strategy(
            bandwidth=4, alpha=0.9, beta=0.0
        ).sensitivity(**self._PART)
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
