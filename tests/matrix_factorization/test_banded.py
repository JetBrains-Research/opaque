"""Tests for ColumnNormalizedBanded matrix operations."""

import pytest
import torch

from opaque.matrix_factorization.banded import (
    ColumnNormalizedBanded,
    minsep_sensitivity_squared,
)


class TestColumnNormalizedBanded:
    def test_default_creation(self):
        cnb = ColumnNormalizedBanded.default(n=10, bands=3)
        assert cnb.n == 10
        assert cnb.bands == 3
        assert cnb.params.shape == (10, 3)

    def test_materialize_square(self):
        cnb = ColumnNormalizedBanded.default(n=5, bands=2)
        M = cnb.materialize()
        assert M.shape == (5, 5)
        # Lower triangular (up to numerical precision)
        upper = torch.triu(M, diagonal=1)
        assert torch.all(torch.abs(upper) < 1e-10)

    def test_column_normalized(self):
        """Verify each column has L2 norm 1."""
        cnb = ColumnNormalizedBanded.default(n=8, bands=3)
        M = cnb.materialize()
        col_norms = torch.linalg.norm(M, dim=0)
        expected = torch.ones(8, dtype=torch.float64)
        torch.testing.assert_close(col_norms, expected, atol=1e-10, rtol=1e-10)

    def test_banded_structure(self):
        """Verify matrix has correct banding."""
        cnb = ColumnNormalizedBanded.default(n=6, bands=2)
        M = cnb.materialize()
        # Entries more than 2 rows below diagonal should be zero
        for i in range(6):
            for j in range(6):
                if i - j >= 2 or j > i:
                    assert abs(float(M[i, j])) < 1e-10, (
                        f"M[{i},{j}]={M[i, j]} should be 0"
                    )

    def test_from_banded_toeplitz(self):
        coefs = torch.tensor([1.0, 0.5], dtype=torch.float64)
        cnb = ColumnNormalizedBanded.from_banded_toeplitz(5, coefs)
        assert cnb.n == 5
        assert cnb.bands == 2


class TestMinsepSensitivity:
    def test_basic(self):
        cnb = ColumnNormalizedBanded.default(n=10, bands=3)
        sens = minsep_sensitivity_squared(cnb, min_sep=3, max_participations=2)
        assert sens == 2  # = max_participations for column-normalized

    def test_min_sep_too_small(self):
        cnb = ColumnNormalizedBanded.default(n=10, bands=3)
        with pytest.raises(ValueError, match="min_sep"):
            minsep_sensitivity_squared(cnb, min_sep=2, max_participations=2)
