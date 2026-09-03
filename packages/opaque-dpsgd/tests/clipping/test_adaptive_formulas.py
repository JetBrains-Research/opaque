"""Backend-neutral adaptive-clipping formula and validation behavior.

These tests exercise pure-float internals and construction-time validation
that never touch a native array, so they run once rather than through the
provider matrix.
"""

from __future__ import annotations

import math

import pytest

from opaque.api.dpsgd.clipping._adaptive import (
    _adaptive_clipping_norm_update,
    adaptive_clipped_grad,
)
from opaque.random import key


class TestGeometricUpdateFormula:
    """Direct unit tests of ``C_{t+1} = C_t * exp(η * (ρ̃ - γ))``."""

    def test_clipping_rate_above_target_increases_threshold(self):
        base, lr, target = 1.0, 0.2, 0.5
        result = _adaptive_clipping_norm_update(
            base_clipping_norm=base,
            noisy_clipping_rate=0.8,
            target_quantile=target,
            learning_rate=lr,
            clipping_norm_min=0.01,
            clipping_norm_max=100.0,
        )
        expected = base * math.exp(lr * (0.8 - target))
        assert abs(result - expected) < 1e-6

    def test_clipping_rate_below_target_decreases_threshold(self):
        base, lr, target = 1.0, 0.2, 0.5
        result = _adaptive_clipping_norm_update(
            base_clipping_norm=base,
            noisy_clipping_rate=0.2,
            target_quantile=target,
            learning_rate=lr,
            clipping_norm_min=0.01,
            clipping_norm_max=100.0,
        )
        expected = base * math.exp(lr * (0.2 - target))
        assert abs(result - expected) < 1e-6

    def test_clipping_rate_at_target_is_a_fixed_point(self):
        base, lr, target = 1.0, 0.2, 0.5
        result = _adaptive_clipping_norm_update(
            base_clipping_norm=base,
            noisy_clipping_rate=target,
            target_quantile=target,
            learning_rate=lr,
            clipping_norm_min=0.01,
            clipping_norm_max=100.0,
        )
        assert abs(result - base) < 1e-6

    def test_step_size_is_proportional_to_deviation_from_target(self):
        base, lr, target = 1.0, 0.2, 0.5
        result_small = _adaptive_clipping_norm_update(
            base_clipping_norm=base,
            noisy_clipping_rate=0.55,
            target_quantile=target,
            learning_rate=lr,
            clipping_norm_min=0.01,
            clipping_norm_max=100.0,
        )
        result_large = _adaptive_clipping_norm_update(
            base_clipping_norm=base,
            noisy_clipping_rate=0.95,
            target_quantile=target,
            learning_rate=lr,
            clipping_norm_min=0.01,
            clipping_norm_max=100.0,
        )
        assert abs(result_large - base) > abs(result_small - base)


class TestAdaptiveClippedGradValidation:
    """Construction-time parameter validation never touches a native array."""

    def test_invalid_initial_clipping_norm(self):
        with pytest.raises(ValueError, match="initial_clipping_norm must be positive"):
            adaptive_clipped_grad(
                lambda params: params.sum(), initial_clipping_norm=-1.0, key=key(0)
            )

    def test_invalid_target_quantile(self):
        with pytest.raises(ValueError, match="target_quantile must be in"):
            adaptive_clipped_grad(
                lambda params: params.sum(), target_quantile=0.0, key=key(0)
            )
        with pytest.raises(ValueError, match="target_quantile must be in"):
            adaptive_clipped_grad(
                lambda params: params.sum(), target_quantile=1.0, key=key(0)
            )

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            adaptive_clipped_grad(
                lambda params: params.sum(), learning_rate=-0.1, key=key(0)
            )

    def test_invalid_clipping_norm_min(self):
        with pytest.raises(ValueError, match="clipping_norm_min must be positive"):
            adaptive_clipped_grad(
                lambda params: params.sum(), clipping_norm_min=-0.1, key=key(0)
            )

    def test_invalid_clipping_norm_max(self):
        with pytest.raises(
            ValueError, match=r"clipping_norm_max.*must be.*clipping_norm_min"
        ):
            adaptive_clipped_grad(
                lambda params: params.sum(),
                clipping_norm_min=10.0,
                clipping_norm_max=5.0,
                key=key(0),
            )

    def test_rejects_stats_with_per_example_auxiliary_output(self):
        with pytest.raises(ValueError, match="return_stats"):
            adaptive_clipped_grad(
                lambda params, data: (params - data).sum(),
                initial_clipping_norm=1.0,
                key=key(0),
                batch_argnums=1,
                return_aux=True,
                return_stats=True,
            )
