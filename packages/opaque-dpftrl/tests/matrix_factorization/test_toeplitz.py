"""Tests for Toeplitz matrix operations."""

import warnings

import numpy as np
import pytest

from opaque.api.dpftrl.noise import _toeplitz
from opaque.api.dpftrl.noise._band_mf import _momentum_workload_coef
from opaque.api.dpftrl.noise._toeplitz import (
    _mean_loss_and_gradient,
    inverse_as_streaming_matrix,
    inverse_coef,
    loss,
    materialize_lower_triangular,
    max_error,
    mean_error,
    minsep_sensitivity_squared,
    multiply,
    optimal_max_error_strategy_coefs,
    optimize,
    pad_coefs_to_n,
    per_query_error,
    sensitivity_squared,
)


class TestMaterializeLowerTriangular:
    def test_identity(self):
        coef = np.array([1.0], dtype=np.float64)
        M = materialize_lower_triangular(coef, n=3)
        np.testing.assert_allclose(M, np.eye(3, dtype=np.float64))

    def test_basic(self):
        coef = np.array([1.0, 0.5, 0.25], dtype=np.float64)
        M = materialize_lower_triangular(coef)
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.25, 0.5, 1.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(M, expected)

    def test_padded(self):
        coef = np.array([1.0, 0.5], dtype=np.float64)
        M = materialize_lower_triangular(coef, n=4)
        assert M.shape == (4, 4)
        assert M[3, 0] == pytest.approx(0.0)  # Beyond band


class TestPadCoefsToN:
    def test_pad(self):
        coef = np.array([1.0, 2.0], dtype=np.float64)
        result = pad_coefs_to_n(coef, 5)
        expected = np.array([1.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_truncate(self):
        coef = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        result = pad_coefs_to_n(coef, 2)
        expected = np.array([1.0, 2.0], dtype=np.float64)
        np.testing.assert_allclose(result, expected)


class TestMultiply:
    def test_identity_multiply(self):
        a = np.array([1.0], dtype=np.float64)
        b = np.array([1.0, 0.5, 0.25], dtype=np.float64)
        result = multiply(a, b, n=3)
        np.testing.assert_allclose(result, b)

    def test_convolution(self):
        a = np.array([1.0, 1.0], dtype=np.float64)
        b = np.array([1.0, 1.0], dtype=np.float64)
        result = multiply(a, b, n=3)
        expected = np.array([1.0, 2.0, 1.0], dtype=np.float64)
        np.testing.assert_allclose(result, expected)


class TestInverseCoef:
    def test_identity_inverse(self):
        coef = np.array([1.0], dtype=np.float64)
        inv = inverse_coef(coef, n=3)
        expected = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        np.testing.assert_allclose(inv, expected, atol=1e-10, rtol=1e-10)

    def test_inverse_roundtrip(self):
        coef = np.array([1.0, 0.5, 0.25], dtype=np.float64)
        inv = inverse_coef(coef)
        # C @ C^{-1} should give identity coefs
        product = multiply(coef, inv, n=3)
        expected = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        np.testing.assert_allclose(product, expected, atol=1e-10, rtol=1e-10)


class TestInverseAsStreamingMatrix:
    def test_vs_dense_inverse(self):
        coef = np.array([1.0, 0.5, 0.25], dtype=np.float64)
        C = materialize_lower_triangular(coef)
        C_inv = np.linalg.inv(C)

        streaming = inverse_as_streaming_matrix(coef)
        M = streaming.materialize(3)

        np.testing.assert_allclose(M, C_inv, atol=1e-10, rtol=1e-10)


class TestOptimalCoefs:
    def test_first_coef_is_one(self):
        coef = optimal_max_error_strategy_coefs(10)
        assert coef[0] == pytest.approx(1.0)

    def test_decreasing(self):
        coef = optimal_max_error_strategy_coefs(10)
        for i in range(1, len(coef)):
            assert coef[i] <= coef[i - 1]

    def test_second_coef(self):
        coef = optimal_max_error_strategy_coefs(5)
        assert coef[1] == pytest.approx(0.5)


class TestSensitivitySquared:
    def test_identity(self):
        coef = np.array([1.0], dtype=np.float64)
        result = sensitivity_squared(coef)
        assert result == pytest.approx(1.0)

    def test_two_bands(self):
        coef = np.array([1.0, 0.5], dtype=np.float64)
        result = sensitivity_squared(coef)
        # ||[1.0, 0.5]||^2 = 1.25
        assert result == pytest.approx(1.25)


class TestMinsepSensitivitySquared:
    def test_single_participation(self):
        coef = np.array([1.0, 0.5], dtype=np.float64)
        result = minsep_sensitivity_squared(coef, min_sep=1, max_participations=1)
        # For single participation, result should be positive
        assert float(result) > 0

    def test_decreasing_required(self):
        coef = np.array([1.0, 2.0], dtype=np.float64)
        with pytest.raises(ValueError, match="non-increasing"):
            minsep_sensitivity_squared(
                coef, min_sep=1, max_participations=1, skip_checks=False
            )


class TestPerQueryError:
    def test_identity_mechanism(self):
        """Identity strategy: error should grow linearly."""
        coef = np.array([1.0], dtype=np.float64)
        error = per_query_error(strategy_coef=coef, n=5)
        # Prefix sum error: cumsum of 1^2 = [1, 2, 3, 4, 5]
        expected = np.arange(1, 6, dtype=np.float64)
        np.testing.assert_allclose(error, expected)

    def test_noising_coef(self):
        """Error from noising coefficients."""
        noising = np.array([1.0, -0.5, 0.25], dtype=np.float64)
        error = per_query_error(noising_coef=noising)
        assert error.shape == (3,)
        assert np.all(error > 0)

    def test_query_weights_apply_on_training_step_axis(self):
        """Schedules scale rows of diag(eta) @ momentum, not Toeplitz lags."""
        n = 4
        momentum = 0.5
        strategy_coef = np.array([1.0, 0.25], dtype=np.float64)
        learning_rates = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        momentum_matrix = materialize_lower_triangular(
            _momentum_workload_coef(momentum, n), n
        )
        strategy = materialize_lower_triangular(strategy_coef, n)
        expected = (learning_rates[:, None] * momentum_matrix) @ np.linalg.inv(strategy)
        expected_error = np.square(expected).sum(axis=1)

        error = per_query_error(
            strategy_coef=strategy_coef,
            n=n,
            workload_coef=_momentum_workload_coef(momentum, n),
            query_weights=learning_rates,
        )

        np.testing.assert_allclose(error, expected_error)

    def test_max_error_uses_weighted_step_maximum(self):
        error = max_error(
            strategy_coef=np.array([1.0], dtype=np.float64),
            n=3,
            query_weights=np.array([3.0, 1.0, 1.0], dtype=np.float64),
        )
        assert error == pytest.approx(9.0)

    def test_both_specified_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            per_query_error(
                strategy_coef=np.ones(3),
                noising_coef=np.ones(3),
            )


class TestMaxAndMeanError:
    def test_max_error(self):
        coef = np.array([1.0], dtype=np.float64)
        error = max_error(strategy_coef=coef, n=5)
        assert error == pytest.approx(5.0)

    def test_mean_error(self):
        coef = np.array([1.0], dtype=np.float64)
        error = mean_error(strategy_coef=coef, n=5)
        assert error == pytest.approx(3.0)  # mean of [1,2,3,4,5]

    def test_custom_error_without_optional_keywords(self):
        def custom_error(*, strategy_coef, n):
            return np.square(strategy_coef).sum() / n

        result = loss(
            np.array([1.0], dtype=np.float64),
            n=2,
            error_fn=custom_error,
        )

        assert result == pytest.approx(0.5)

    def test_custom_loss_without_optional_keywords(self):
        def custom_loss(strategy_coef, *, n):
            return np.square(strategy_coef - 1).sum() + 1 / n

        coefs = optimize(
            n=2,
            bands=1,
            loss_fn=custom_loss,
            max_optimizer_steps=1,
        )

        np.testing.assert_allclose(coefs, np.ones(1, dtype=np.float64))

    def test_mean_loss_gradient_matches_centered_difference(self):
        params = np.array([1.0, 0.35, 0.1], dtype=np.float64)
        workload = np.power(0.9, np.arange(20))
        query_weights = np.linspace(0.5, 1.5, 20)

        value, gradient = _mean_loss_and_gradient(
            params,
            n=20,
            workload_coef=workload,
            query_weights=query_weights,
        )
        step = 1e-7
        centered_gradient = np.array(
            [
                (
                    _mean_loss_and_gradient(
                        params + np.eye(len(params))[i] * step,
                        n=20,
                        workload_coef=workload,
                        query_weights=query_weights,
                    )[0]
                    - _mean_loss_and_gradient(
                        params - np.eye(len(params))[i] * step,
                        n=20,
                        workload_coef=workload,
                        query_weights=query_weights,
                    )[0]
                )
                / (2.0 * step)
                for i in range(len(params))
            ]
        )

        assert np.isfinite(value)
        np.testing.assert_allclose(gradient, centered_gradient, rtol=1e-6, atol=1e-6)


class TestMomentumWorkloadCoef:
    """Tests for _momentum_workload_coef and workload-aware optimization."""

    def test_zero_momentum_coefficients(self):
        """β=0 produces identity workload [1, 0, 0, ...] with no NaN/Inf."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coef = _momentum_workload_coef(0.0, 10)
        assert coef.shape == (10,)
        assert coef[0] == pytest.approx(1.0)
        assert np.all(np.asarray(coef[1:]) == 0.0)
        assert np.all(np.isfinite(np.asarray(coef)))

    def test_zero_momentum_emits_warning(self):
        """β=0 emits a UserWarning about identity workload."""
        with pytest.warns(UserWarning, match="identity workload"):
            _momentum_workload_coef(0.0, 5)

    def test_negative_momentum_raises(self):
        """Negative momentum is rejected."""
        with pytest.raises(ValueError, match="momentum must be >= 0"):
            _momentum_workload_coef(-0.1, 10)

    def test_typical_momentum_coefficients(self):
        """β=0.9 gives [1, 0.9, 0.81, ...]."""
        coef = _momentum_workload_coef(0.9, 4)
        expected = np.array([1.0, 0.9, 0.81, 0.729], dtype=np.float64)
        np.testing.assert_allclose(coef, expected)

    def test_prefix_sum_momentum(self):
        """β=1.0 gives [1, 1, 1, ...] (prefix-sum workload)."""
        coef = _momentum_workload_coef(1.0, 5)
        expected = np.ones(5, dtype=np.float64)
        np.testing.assert_allclose(coef, expected)

    def test_identity_workload_error_is_constant(self):
        """With identity workload, per-query error is constant (= 1) for identity strategy."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wc = _momentum_workload_coef(0.0, 5)
        # Identity strategy: C = I, coef = [1, 0, 0, ...]
        error = per_query_error(
            strategy_coef=np.array([1.0], dtype=np.float64),
            n=5,
            workload_coef=wc,
        )
        # B_coef = solve(I, [1,0,0,0,0]) = [1,0,0,0,0]
        # cumsum([1,0,0,0,0]^2) = [1,1,1,1,1]
        expected = np.ones(5, dtype=np.float64)
        np.testing.assert_allclose(error, expected)

    def test_identity_workload_optimization_converges(self):
        """Optimizer produces finite loss with identity workload (β=0).

        With identity workload, the optimal single-band strategy is C=I,
        giving loss = sensitivity² × error = 1.0 × 1.0 = 1.0.
        The optimizer should converge to this or better (multi-band).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wc = _momentum_workload_coef(0.0, 20)

        coefs = optimize(n=20, bands=3, workload_coef=wc)

        # Basic sanity: result is finite, unit norm
        assert np.all(np.isfinite(coefs))
        assert np.linalg.norm(coefs) == pytest.approx(1.0, abs=1e-6)

        # Loss should be finite and ≤ identity baseline (1.0)
        opt_loss = loss(coefs, n=20, workload_coef=wc)
        assert np.isfinite(opt_loss)
        assert opt_loss <= 1.0 + 1e-6

    def test_momentum_workload_vs_prefix_sum(self):
        """Momentum workload produces different (better) loss than prefix-sum
        when the optimizer actually uses momentum."""
        n = 30
        bands = 5
        wc_momentum = _momentum_workload_coef(0.95, n)
        wc_prefix = _momentum_workload_coef(1.0, n)

        coefs_mom = optimize(n=n, bands=bands, workload_coef=wc_momentum)
        coefs_pfx = optimize(n=n, bands=bands, workload_coef=wc_prefix)

        # Evaluate BOTH strategies under the momentum workload
        loss_mom = loss(coefs_mom, n=n, workload_coef=wc_momentum)
        loss_pfx = loss(coefs_pfx, n=n, workload_coef=wc_momentum)

        # Strategy optimized for momentum workload should be at least as good
        # (lower loss) as the one optimized for prefix-sum, when evaluated
        # under the momentum workload.
        assert loss_mom <= loss_pfx + 1e-6


def test_default_band_mf_uses_analytic_gradient(monkeypatch):
    """The default objective must not silently revert to finite differences."""
    optimizer = _toeplitz._lbfgs_optimize
    analytic_gradient_flags = []

    def record_optimizer(loss_fn, params, **kwargs):
        analytic_gradient_flags.append(kwargs.get("grad"))
        return optimizer(loss_fn, params, **kwargs)

    monkeypatch.setattr(_toeplitz, "_lbfgs_optimize", record_optimizer)

    optimize(n=40, bands=5)

    assert analytic_gradient_flags == [True]
