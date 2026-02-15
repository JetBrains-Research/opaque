"""Tests for Buffered Linear Toeplitz (BLT) mechanisms."""

import pytest
import torch

from opaque.matrix_factorization.buffered_toeplitz import (
    BufferedToeplitz,
    LossFn,
    blt_pair_from_theta_pair,
    geometric_sum,
    get_init_blt,
    iteration_error,
    max_error,
    min_buf_decay_gap,
    sensitivity_squared,
)
from opaque.matrix_factorization.toeplitz import materialize_lower_triangular


class TestBufferedToeplitz:
    def test_build(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.9, 0.5],
            output_scale=[0.3, 0.2],
        )
        assert blt._num_buffers == 2
        assert blt.dtype == torch.float64

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
        coefs = blt.toeplitz_coefs(5)
        assert coefs[0] == pytest.approx(1.0)
        assert coefs[1] == pytest.approx(0.3)
        assert coefs[2] == pytest.approx(0.3 * 0.5)
        assert coefs[3] == pytest.approx(0.3 * 0.5**2)

    def test_materialize(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.5],
            output_scale=[0.3],
        )
        M = blt.materialize(3)
        expected = materialize_lower_triangular(blt.toeplitz_coefs(3))
        torch.testing.assert_close(M, expected)

    def test_empty_blt(self):
        blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
        coefs = blt.toeplitz_coefs(3)
        expected = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(coefs, expected)

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
        inv_blt = blt.inverse()

        n = 6
        C = blt.materialize(n)
        C_inv = inv_blt.materialize(n)
        product = C @ C_inv
        torch.testing.assert_close(
            product, torch.eye(n, dtype=torch.float64), atol=1e-8, rtol=1e-8
        )

    def test_single_buffer_inverse(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.5],
            output_scale=[0.3],
        )
        inv_blt = blt.inverse()
        n = 5
        C = blt.materialize(n)
        C_inv = inv_blt.materialize(n)
        product = C @ C_inv
        torch.testing.assert_close(
            product, torch.eye(n, dtype=torch.float64), atol=1e-8, rtol=1e-8
        )


class TestBLTStreamingMatrix:
    def test_streaming_matches_dense(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.7, 0.3],
            output_scale=[0.4, 0.2],
        )
        n = 5
        dense = blt.materialize(n)
        streaming = blt.as_streaming_matrix()
        streaming_dense = streaming.materialize(n)
        torch.testing.assert_close(streaming_dense, dense, atol=1e-8, rtol=1e-8)

    def test_inverse_streaming_matches_dense(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.7, 0.3],
            output_scale=[0.4, 0.2],
        )
        n = 5
        C = blt.materialize(n)
        C_inv_dense = torch.linalg.inv(C)
        C_inv_streaming = blt.inverse_as_streaming_matrix()
        C_inv_streaming_dense = C_inv_streaming.materialize(n)
        torch.testing.assert_close(
            C_inv_streaming_dense, C_inv_dense, atol=1e-8, rtol=1e-8
        )


class TestGeometricSum:
    def test_finite(self):
        a = torch.tensor(1.0, dtype=torch.float64)
        r = torch.tensor(0.5, dtype=torch.float64)
        result = geometric_sum(a, r, num=4)
        expected = 1 + 0.5 + 0.25 + 0.125
        assert float(result) == pytest.approx(expected)

    def test_infinite(self):
        a = torch.tensor(1.0, dtype=torch.float64)
        r = torch.tensor(0.5, dtype=torch.float64)
        result = geometric_sum(a, r)
        assert float(result) == pytest.approx(2.0)

    def test_zero_ratio(self):
        a = torch.tensor(3.0, dtype=torch.float64)
        r = torch.tensor(0.0, dtype=torch.float64)
        result = geometric_sum(a, r, num=5)
        assert float(result) == pytest.approx(3.0)


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


class TestMaxError:
    def test_identity_error(self):
        blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
        error = max_error(blt, n=5)
        assert float(error) == pytest.approx(5.0)


class TestMinBufDecayGap:
    def test_basic(self):
        theta = torch.tensor([0.9, 0.5, 0.3], dtype=torch.float64)
        gap = min_buf_decay_gap(theta)
        assert float(gap) == pytest.approx(0.2)

    def test_duplicates(self):
        theta = torch.tensor([0.5, 0.5], dtype=torch.float64)
        gap = min_buf_decay_gap(theta)
        assert float(gap) == pytest.approx(0.0)


class TestBLTPairFromThetaPair:
    def test_inverse_property(self):
        theta = torch.tensor([0.8, 0.4], dtype=torch.float64)
        theta_hat = torch.tensor([0.6, 0.2], dtype=torch.float64)
        blt, inv_blt = blt_pair_from_theta_pair(theta, theta_hat)

        n = 6
        C = blt.materialize(n)
        C_inv = inv_blt.materialize(n)
        product = C @ C_inv
        torch.testing.assert_close(
            product, torch.eye(n, dtype=torch.float64), atol=1e-6, rtol=1e-6
        )


class TestGetInitBLT:
    def test_zero_buffers(self):
        blt = get_init_blt(num_buffers=0)
        assert blt._num_buffers == 0

    def test_default_init(self):
        blt = get_init_blt(num_buffers=3)
        assert blt._num_buffers == 3
        assert blt.dtype == torch.float64

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
        loss = loss_fn.loss(blt)
        assert float(loss) > 0
        assert torch.isfinite(loss)

    def test_min_sep(self):
        loss_fn = LossFn.build_min_sep(n=20, min_sep=2, max_participations=3)
        blt = get_init_blt(num_buffers=2)
        loss = loss_fn.loss(blt, skip_checks=True)
        assert float(loss) > 0


class TestFromRationalApprox:
    def test_basic(self):
        blt = BufferedToeplitz.from_rational_approx_to_sqrt_x(
            num_buffers=3,
            max_buf_decay=1 - 1e-6,
            max_pillutla_score=1 - 1e-6,
        )
        assert blt._num_buffers == 3
        assert torch.all(blt.buf_decay > 0)
        assert torch.all(blt.buf_decay < 1)

    def test_invalid_num_buffers(self):
        with pytest.raises(ValueError):
            BufferedToeplitz.from_rational_approx_to_sqrt_x(num_buffers=0)
