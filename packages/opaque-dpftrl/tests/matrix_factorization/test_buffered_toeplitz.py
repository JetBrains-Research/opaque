"""Tests for Buffered Linear Toeplitz (BLT) mechanisms."""

import numpy as np
import pytest

from opaque.api.dpftrl.noise._blt_math import (
    BufferedToeplitz,
    LossFn,
    Parameterization,
    as_streaming_matrix,
    blt_pair_from_theta_pair,
    geometric_sum,
    get_init_blt,
    get_parameterized_loss,
    get_parameterized_loss_and_gradient,
    inverse,
    inverse_as_streaming_matrix,
    iteration_error,
    limit_max_error,
    limit_max_loss,
    loss,
    materialize,
    max_error,
    min_buf_decay_gap,
    optimize,
    optimize_loss,
    penalized_loss,
    robust_max_error_Gamma_j,
    robust_max_error_Gamma_jk,
    sensitivity_squared,
    toeplitz_coefs,
)
from opaque.api.dpftrl.noise._plan import toeplitz_execution_plan
from opaque.api.dpftrl.noise._toeplitz import materialize_lower_triangular


class TestBufferedToeplitz:
    def test_build(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.9, 0.5],
            output_scale=[0.3, 0.2],
        )
        assert blt._num_buffers == 2
        assert blt.dtype == np.float64

    def test_canonicalize_sorts_descending(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.3, 0.9, 0.5],
            output_scale=[0.1, 0.2, 0.3],
        )
        # After canonicalization, buf_decay should be descending
        for i in range(1, blt._num_buffers):
            assert float(blt.buf_decay[i]) <= float(blt.buf_decay[i - 1])

    def test_toeplitz_coefs(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.5],
            output_scale=[0.3],
        )
        coefs = toeplitz_coefs(blt, 5)
        assert coefs[0] == pytest.approx(1.0)
        assert coefs[1] == pytest.approx(0.3)
        assert coefs[2] == pytest.approx(0.3 * 0.5)
        assert coefs[3] == pytest.approx(0.3 * 0.5**2)

    def test_materialize(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.5],
            output_scale=[0.3],
        )
        M = materialize(blt, 3)
        expected = materialize_lower_triangular(toeplitz_coefs(blt, 3))
        np.testing.assert_allclose(M, expected)

    def test_empty_blt(self):
        blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
        coefs = toeplitz_coefs(blt, 3)
        expected = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        np.testing.assert_allclose(coefs, expected)

    def test_pillutla_score(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.8, 0.4],
            output_scale=[0.3, 0.1],
        )
        score = blt.pillutla_score()
        expected = 0.3 / 0.8 + 0.1 / 0.4
        assert score == pytest.approx(expected)


class TestBLTInverse:
    def test_inverse_roundtrip(self):
        """C @ C^{-1} should be identity."""
        blt = BufferedToeplitz.build(
            buf_decay=[0.8, 0.4],
            output_scale=[0.3, 0.1],
        )
        inv_blt = inverse(blt)

        n = 6
        C = materialize(blt, n)
        C_inv = materialize(inv_blt, n)
        product = C @ C_inv
        np.testing.assert_allclose(
            product, np.eye(n, dtype=np.float64), atol=1e-8, rtol=1e-8
        )

    def test_single_buffer_inverse(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.5],
            output_scale=[0.3],
        )
        inv_blt = inverse(blt)
        n = 5
        C = materialize(blt, n)
        C_inv = materialize(inv_blt, n)
        product = C @ C_inv
        np.testing.assert_allclose(
            product, np.eye(n, dtype=np.float64), atol=1e-8, rtol=1e-8
        )


class TestBLTStreamingMatrix:
    def test_streaming_matches_dense(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.7, 0.3],
            output_scale=[0.4, 0.2],
        )
        n = 5
        dense = materialize(blt, n)
        streaming = as_streaming_matrix(blt)
        streaming_dense = streaming.materialize(n)
        np.testing.assert_allclose(streaming_dense, dense, atol=1e-8, rtol=1e-8)

    def test_inverse_streaming_matches_dense(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.7, 0.3],
            output_scale=[0.4, 0.2],
        )
        n = 5
        C = materialize(blt, n)
        C_inv_dense = np.linalg.inv(C)
        C_inv_streaming = inverse_as_streaming_matrix(blt)
        C_inv_streaming_dense = C_inv_streaming.materialize(n)
        np.testing.assert_allclose(
            C_inv_streaming_dense, C_inv_dense, atol=1e-8, rtol=1e-8
        )

    @pytest.mark.parametrize(
        ("buf_decay", "output_scale"),
        [([0.7, 0.3], [0.4, 0.2]), ([0.5], [0.3]), ([], [])],
    )
    def test_plan_row_norms_match_probing_and_dense(self, buf_decay, output_scale):
        blt = BufferedToeplitz.build(buf_decay=buf_decay, output_scale=output_scale)
        n = 6
        plan = toeplitz_execution_plan(
            toeplitz_coefs(blt, n),
            mode="blt",
            buffer_decay=blt.buf_decay,
            output_scale=blt.output_scale,
        )
        C_inv = inverse_as_streaming_matrix(blt)
        np.testing.assert_allclose(
            np.square(plan.row_l2),
            C_inv.row_norms_squared(n),
            atol=1e-10,
            rtol=1e-10,
        )
        dense = np.linalg.inv(materialize(blt, n))
        np.testing.assert_allclose(
            np.square(plan.row_l2),
            np.square(dense).sum(axis=1),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_plan_row_norms_high_buffer_near_one_decays(self):
        # The default 10-buffer init has decays within 1e-6 of 1.0 — a
        # regime where an expanded rational-transfer-function filter
        # drifts from the truth by whole percents at long horizons. The
        # plan's direct Toeplitz-inversion recurrence must agree with the
        # buffer-recurrence streaming inverse.
        blt = get_init_blt(10)
        n = 1024
        plan = toeplitz_execution_plan(
            toeplitz_coefs(blt, n),
            mode="blt",
            buffer_decay=blt.buf_decay,
            output_scale=blt.output_scale,
        )
        C_inv = inverse_as_streaming_matrix(blt)
        np.testing.assert_allclose(
            np.square(plan.row_l2),
            C_inv.row_norms_squared(n),
            atol=1e-10,
            rtol=1e-10,
        )


class TestGeometricSum:
    def test_finite(self):
        a = np.array(1.0, dtype=np.float64)
        r = np.array(0.5, dtype=np.float64)
        result = geometric_sum(a, r, num=4)
        expected = 1 + 0.5 + 0.25 + 0.125
        assert float(result) == pytest.approx(expected)

    def test_infinite(self):
        a = np.array(1.0, dtype=np.float64)
        r = np.array(0.5, dtype=np.float64)
        result = geometric_sum(a, r)
        assert float(result) == pytest.approx(2.0)

    def test_zero_ratio(self):
        a = np.array(3.0, dtype=np.float64)
        r = np.array(0.0, dtype=np.float64)
        result = geometric_sum(a, r, num=5)
        assert float(result) == pytest.approx(3.0)

    def test_near_one_ratio(self):
        """geometric_sum should be accurate when r is very close to 1."""
        a = np.array(1.0, dtype=np.float64)
        r = np.array(1 - 1e-10, dtype=np.float64)
        result = geometric_sum(a, r, num=100)
        # For r very close to 1, the sum should be close to n=100
        assert float(result) == pytest.approx(100.0, rel=1e-4)


class TestRobustGammaJ:
    """Tests for the numerically robust Gamma_j computation."""

    def test_matches_brute_force_small_theta(self):
        """robust_max_error_Gamma_j matches brute force for theta far from 1."""
        omega = np.array(0.3, dtype=np.float64)
        theta = np.array(0.5, dtype=np.float64)
        n = 10
        result = robust_max_error_Gamma_j(omega, theta, n)
        # Brute force: sum_{i=1}^{n-1} geometric_sum(omega, theta, i) / n
        total = sum(float(geometric_sum(omega, theta, num=i)) for i in range(1, n))
        expected = total / n
        assert float(result) == pytest.approx(expected, rel=1e-6)

    def test_near_one_theta_finite(self):
        """robust_max_error_Gamma_j returns finite values for theta near 1."""
        omega = np.array(0.3, dtype=np.float64)
        theta = np.array(1 - 1e-10, dtype=np.float64)
        result = robust_max_error_Gamma_j(omega, theta, 1000)
        assert np.isfinite(result)
        assert float(result) > 0


class TestRobustGammaJK:
    """Tests for the numerically robust Gamma_jk computation."""

    def test_matches_brute_force_small_thetas(self):
        """robust_max_error_Gamma_jk matches brute force for small thetas."""
        omega1 = np.array(0.3, dtype=np.float64)
        theta1 = np.array(0.5, dtype=np.float64)
        omega2 = np.array(0.2, dtype=np.float64)
        theta2 = np.array(0.4, dtype=np.float64)
        n = 10
        result = robust_max_error_Gamma_jk(omega1, theta1, omega2, theta2, n)
        # Brute force
        total = sum(
            float(geometric_sum(omega1, theta1, num=i))
            * float(geometric_sum(omega2, theta2, num=i))
            for i in range(1, n)
        )
        expected = total / n
        assert float(result) == pytest.approx(expected, rel=1e-6)

    def test_both_near_one_finite(self):
        """robust_max_error_Gamma_jk handles both thetas near 1."""
        omega1 = np.array(0.3, dtype=np.float64)
        theta1 = np.array(1 - 1e-10, dtype=np.float64)
        omega2 = np.array(0.2, dtype=np.float64)
        theta2 = np.array(1 - 1e-10, dtype=np.float64)
        result = robust_max_error_Gamma_jk(omega1, theta1, omega2, theta2, 1000)
        assert np.isfinite(result)

    def test_one_near_one_finite(self):
        """robust_max_error_Gamma_jk handles one theta near 1."""
        omega1 = np.array(0.3, dtype=np.float64)
        theta1 = np.array(1 - 1e-10, dtype=np.float64)
        omega2 = np.array(0.2, dtype=np.float64)
        theta2 = np.array(0.5, dtype=np.float64)
        result = robust_max_error_Gamma_jk(omega1, theta1, omega2, theta2, 1000)
        assert np.isfinite(result)


class TestIterationErrorRobust:
    """Test that iteration_error is finite even for BLTs with buf_decay near 1."""

    def test_normal_blt(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.5, 0.3],
            output_scale=[-0.2, -0.1],
        )
        err = iteration_error(blt, 50)
        assert np.isfinite(err)
        assert float(err) > 0

    def test_identity_error(self):
        blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
        error = max_error(blt, n=5)
        assert float(error) == pytest.approx(5.0)


class TestLimitMaxError:
    def test_identity(self):
        """Identity BLT (no buffers) should have limit_max_error = 1."""
        inv_blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
        result = limit_max_error(inv_blt)
        assert float(result) == pytest.approx(1.0)

    def test_converges_to_finite_n(self):
        """limit_max_error should approximate max_error(n)/n for large n."""
        blt = BufferedToeplitz.build(
            buf_decay=[0.8, 0.4],
            output_scale=[0.3, 0.1],
        )
        inv_blt = inverse(blt)
        limit = float(limit_max_error(inv_blt))

        # For large n, max_error(n)/n should converge to limit_max_error
        n_large = 10000
        finite = float(max_error(inv_blt, n_large)) / n_large
        assert limit == pytest.approx(finite, rel=1e-3)

    def test_single_buffer(self):
        """Sanity check for single-buffer BLT."""
        inv_blt = BufferedToeplitz.build(
            buf_decay=[0.5],
            output_scale=[-0.3],
        )
        result = limit_max_error(inv_blt)
        # 1 + 2*(-0.3)/(1-0.5) + (-0.3)^2/(1-0.5)^2
        #   = 1 + 2*(-0.6) + 0.36 = 1 - 1.2 + 0.36 = 0.16
        expected = 1 + 2 * (-0.3 / 0.5) + (-0.3 / 0.5) ** 2
        assert float(result) == pytest.approx(expected, rel=1e-10)


class TestLimitMaxLoss:
    def test_identity(self):
        """Identity BLT should have limit_max_loss = 1."""
        blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
        result = limit_max_loss(blt)
        assert float(result) == pytest.approx(1.0)

    def test_converges_to_finite_n(self):
        """limit_max_loss should approximate loss for large n."""
        blt = BufferedToeplitz.build(
            buf_decay=[0.8, 0.4],
            output_scale=[0.3, 0.1],
        )
        limit = float(limit_max_loss(blt))

        inv_blt = inverse(blt)
        n_large = 10000
        finite_error = float(max_error(inv_blt, n_large)) / n_large
        finite_sens = float(sensitivity_squared(blt, n_large))
        finite_loss = finite_error * finite_sens
        assert limit == pytest.approx(finite_loss, rel=1e-3)


class TestSensitivitySquared:
    def test_identity(self):
        blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
        result = sensitivity_squared(blt, n=100)
        assert float(result) == pytest.approx(1.0)

    def test_single_buffer(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.5],
            output_scale=[0.3],
        )
        result = sensitivity_squared(blt, n=10)
        # sens^2 = 1 + sum of geometric series
        assert float(result) > 1.0


class TestMinBufDecayGap:
    def test_basic(self):
        theta = np.array([0.9, 0.5, 0.3], dtype=np.float64)
        gap = min_buf_decay_gap(theta)
        assert float(gap) == pytest.approx(0.2)

    def test_duplicates(self):
        theta = np.array([0.5, 0.5], dtype=np.float64)
        gap = min_buf_decay_gap(theta)
        assert float(gap) == pytest.approx(0.0)


class TestBLTPairFromThetaPair:
    def test_inverse_property(self):
        theta = np.array([0.8, 0.4], dtype=np.float64)
        theta_hat = np.array([0.6, 0.2], dtype=np.float64)
        blt, inv_blt = blt_pair_from_theta_pair(theta, theta_hat)

        n = 6
        C = materialize(blt, n)
        C_inv = materialize(inv_blt, n)
        product = C @ C_inv
        np.testing.assert_allclose(
            product, np.eye(n, dtype=np.float64), atol=1e-6, rtol=1e-6
        )


class TestGetInitBLT:
    def test_zero_buffers(self):
        blt = get_init_blt(num_buffers=0)
        assert blt._num_buffers == 0

    def test_default_init(self):
        blt = get_init_blt(num_buffers=3)
        assert blt._num_buffers == 3
        assert blt.dtype == np.float64

    def test_wrong_num_buffers(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.5],
            output_scale=[0.3],
        )
        with pytest.raises(ValueError, match="num_buffers"):
            get_init_blt(num_buffers=3, init_blt=blt)


class TestLossFn:
    def test_single_participation(self):
        loss_fn = LossFn.build_closed_form_single_participation(n=10)
        blt = get_init_blt(num_buffers=2)
        loss_val = loss(loss_fn, blt)
        assert float(loss_val) > 0
        assert np.isfinite(loss_val)

    def test_min_sep(self):
        loss_fn = LossFn.build_min_sep(n=20, min_sep=2, max_participations=3)
        blt = get_init_blt(num_buffers=2)
        loss_val = loss(loss_fn, blt, skip_checks=True)
        assert float(loss_val) > 0

    def test_penalized_loss(self):
        loss_fn = LossFn.build_closed_form_single_participation(n=10)
        blt = get_init_blt(num_buffers=2)
        inv_blt = inverse(blt)
        penalized = penalized_loss(loss_fn, blt, inv_blt)
        plain = loss(loss_fn, blt)
        # Penalized loss should be close to plain loss (penalty is small)
        assert np.isfinite(penalized)
        # penalty_strength is 1e-8, so difference should be small
        assert float(penalized) == pytest.approx(float(plain), abs=1.0)


class TestParameterization:
    def test_buf_decay_pair_roundtrip(self):
        """Parameters -> BLT -> parameters should approximately round-trip."""
        param = Parameterization.buf_decay_pair()
        blt = get_init_blt(num_buffers=2)
        params = param.params_from_blt(blt)
        assert params.shape == (4,)  # 2 theta + 2 theta_hat

        blt_out, inv_blt_out = param.blt_and_inverse_from_params(params)
        n = 10
        C = materialize(blt_out, n)
        C_inv = materialize(inv_blt_out, n)
        product = C @ C_inv
        np.testing.assert_allclose(
            product, np.eye(n, dtype=np.float64), atol=1e-6, rtol=1e-6
        )

    def test_get_loss_fn(self):
        loss_fn = LossFn.build_closed_form_single_participation(n=10)
        param = Parameterization.buf_decay_pair()
        blt = get_init_blt(num_buffers=2)
        params = param.params_from_blt(blt)
        opt_loss_fn = get_parameterized_loss(param, loss_fn)
        result = opt_loss_fn(params)
        assert np.isfinite(result)
        assert float(result) > 0

    def test_min_sep_gradient_matches_centered_difference(self):
        n = 30
        loss_fn = LossFn.build_min_sep(
            n=n,
            min_sep=5,
            max_participations=3,
            workload_coef=np.power(0.9, np.arange(n)),
        )
        param = Parameterization.buf_decay_pair()
        params = np.array([0.8, 0.35, 0.65, 0.15])
        value_and_gradient = get_parameterized_loss_and_gradient(param, loss_fn)
        assert value_and_gradient is not None

        value, gradient = value_and_gradient(params)
        scalar_loss = get_parameterized_loss(param, loss_fn)
        step = 1e-7
        centered_gradient = np.array(
            [
                (
                    scalar_loss(params + np.eye(len(params))[i] * step)
                    - scalar_loss(params - np.eye(len(params))[i] * step)
                )
                / (2.0 * step)
                for i in range(len(params))
            ]
        )

        assert np.isfinite(value)
        np.testing.assert_allclose(gradient, centered_gradient, rtol=1e-6, atol=1e-6)


class TestOptimizeLoss:
    def test_zero_buffers(self):
        loss_fn = LossFn.build_closed_form_single_participation(n=10)
        blt, loss = optimize_loss(loss_fn, num_buffers=0)
        assert blt._num_buffers == 0
        assert float(loss) == pytest.approx(10.0)

    def test_single_buffer_improves(self):
        """L-BFGS should not make the loss worse (within tolerance)."""
        loss_fn = LossFn.build_closed_form_single_participation(n=100)
        init_blt = get_init_blt(num_buffers=1)
        init_loss = float(loss(loss_fn, init_blt))

        _opt_blt, opt_loss = optimize_loss(
            loss_fn, num_buffers=1, max_optimizer_steps=50
        )
        # Allow tiny floating-point tolerance
        assert float(opt_loss) <= init_loss * (1 + 1e-10)
        assert np.isfinite(opt_loss)

    def test_two_buffers(self):
        loss_fn = LossFn.build_closed_form_single_participation(n=50)
        opt_blt, opt_loss = optimize_loss(
            loss_fn, num_buffers=2, max_optimizer_steps=50
        )
        assert opt_blt._num_buffers == 2
        assert np.isfinite(opt_loss)
        assert float(opt_loss) > 0

    def test_min_sep_momentum_escapes_initialization(self):
        """Analytic BLT gradients must avoid the old finite-difference stall."""
        n = 100
        loss_fn = LossFn.build_min_sep(
            n=n,
            min_sep=10,
            max_participations=None,
            workload_coef=np.power(0.9, np.arange(n)),
        )
        initial_loss = float(
            loss(loss_fn, get_init_blt(num_buffers=1), skip_checks=True)
        )

        _blt, optimized_loss = optimize_loss(loss_fn, num_buffers=1)

        assert float(optimized_loss) < initial_loss * 0.5

    def test_representative_min_sep_momentum_converges_with_bounded_work(self):
        """The n=500 BLT recovery must retain quality without the old fallback work."""
        n = 500
        loss_fn = LossFn.build_min_sep(
            n=n,
            min_sep=10,
            max_participations=1,
            workload_coef=np.power(0.9, np.arange(n)),
        )
        evaluations = []
        blt = optimize(
            n=n,
            min_sep=10,
            max_participations=1,
            max_buffers=3,
            workload_coef=np.power(0.9, np.arange(n)),
            callback=lambda event: evaluations.append(event),
        )
        optimized_loss = float(loss(loss_fn, blt, skip_checks=True))
        optimized_sensitivity = float(np.sqrt(sensitivity_squared(blt, n=n)))

        # The feasible Armijo restart runs a second short L-BFGS-B pass.  This
        # range distinguishes it from an immediate stall and the former
        # 100-step fallback without depending on wall-clock timing.
        assert 20 <= len(evaluations) <= 40
        assert np.all(np.isfinite(blt.buf_decay))
        assert np.all(np.isfinite(blt.output_scale))
        assert optimized_loss <= 2.126025545777228
        assert optimized_sensitivity <= 1.2023459630813556


class TestFromRationalApprox:
    def test_basic(self):
        blt = BufferedToeplitz.from_rational_approx_to_sqrt_x(
            num_buffers=3,
            max_buf_decay=1 - 1e-6,
            max_pillutla_score=1 - 1e-6,
        )
        assert blt._num_buffers == 3
        assert np.all(blt.buf_decay > 0)
        assert np.all(blt.buf_decay < 1)

    def test_invalid_num_buffers(self):
        with pytest.raises(ValueError, match="num_buffers must be >= 1"):
            BufferedToeplitz.from_rational_approx_to_sqrt_x(num_buffers=0)
