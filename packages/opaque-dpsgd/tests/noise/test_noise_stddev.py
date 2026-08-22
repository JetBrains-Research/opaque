"""Tests for ``opaque.dpsgd.noise.noise_stddev``.

The public free function and ``ClippedPytree.noise_stddev_for`` must be one
computation: the method is the bound form, reached when you already hold a
clipped pytree, and the function is what a calibration sweep or a telemetry
line calls before any step has run.
"""

import math

import pytest

from opaque.dpsgd.noise import noise_stddev
from opaque.types import PerGroup, clipped


def _bound(values):
    return PerGroup(
        groups={f"p{i}": f"g{i}" for i in range(len(values))},
        values={f"g{i}": v for i, v in enumerate(values)},
    )


class TestAgreesWithTheBoundForm:
    """Whatever the caller holds, the answer is the same."""

    @pytest.mark.parametrize("allocation", ["optimal", "isotropic"])
    @pytest.mark.parametrize("max_norm", [1.0, 2.5, _bound([1.0, 4.0, 9.0])])
    def test_free_function_matches_the_method(self, max_norm, allocation):
        expected = clipped({}, max_norm=max_norm).noise_stddev_for(
            noise_multiplier=1.3, allocation=allocation
        )
        actual = noise_stddev(max_norm, noise_multiplier=1.3, allocation=allocation)
        if isinstance(expected, PerGroup):
            assert dict(actual.values) == dict(expected.values)
            assert actual.groups == expected.groups
        else:
            assert actual == expected

    def test_needs_no_pytree(self):
        """The point of the free function: a bound is the whole input."""
        assert noise_stddev(2.0, noise_multiplier=1.5) == 3.0


class TestAllocation:
    def test_scalar_is_the_plain_product(self):
        assert noise_stddev(4.0, noise_multiplier=0.5) == 2.0

    def test_optimal_is_mahalanobis(self):
        """``σᵢ = nm · √(Bᵢ · ΣⱼBⱼ)`` — the MSE-optimal allocation."""
        max_norm = _bound([1.0, 4.0])
        result = noise_stddev(max_norm, noise_multiplier=2.0)
        total = 5.0
        assert result.values["g0"] == pytest.approx(2.0 * math.sqrt(1.0 * total))
        assert result.values["g1"] == pytest.approx(2.0 * math.sqrt(4.0 * total))

    def test_isotropic_collapses_to_the_effective_norm(self):
        max_norm = _bound([3.0, 4.0])
        result = noise_stddev(max_norm, noise_multiplier=2.0, allocation="isotropic")
        assert result == pytest.approx(2.0 * max_norm.effective)

    def test_allocation_is_ignored_for_a_scalar_bound(self):
        both = {
            a: noise_stddev(2.0, noise_multiplier=1.0, allocation=a)
            for a in ("optimal", "isotropic")
        }
        assert both["optimal"] == both["isotropic"] == 2.0


class TestNonPrivateRun:
    """``nm == 0`` is zero noise — never ``0 * inf = NaN``.

    A disabled clip carries ``max_norm = +inf``.  Both ``gaussian_noise`` and
    ``mf_gaussian_noise`` short-circuit on that product internally; the shared
    helper has to agree with them, or a caller reading the stddev for a log
    line gets ``nan`` where the mechanism applied ``0``.
    """

    @pytest.mark.parametrize("allocation", ["optimal", "isotropic"])
    def test_infinite_scalar_bound_gives_zero_not_nan(self, allocation):
        result = noise_stddev(math.inf, noise_multiplier=0.0, allocation=allocation)
        assert result == 0.0

    @pytest.mark.parametrize("allocation", ["optimal", "isotropic"])
    def test_infinite_per_group_bound_gives_zero_not_nan(self, allocation):
        result = noise_stddev(
            _bound([math.inf, math.inf]), noise_multiplier=0.0, allocation=allocation
        )
        values = result.values.values() if isinstance(result, PerGroup) else [result]
        assert all(v == 0.0 for v in values)

    def test_finite_bound_at_zero_multiplier_is_still_zero(self):
        assert noise_stddev(5.0, noise_multiplier=0.0) == 0.0


class TestValidation:
    def test_negative_multiplier_rejected(self):
        with pytest.raises(ValueError, match="noise_multiplier must be non-negative"):
            noise_stddev(1.0, noise_multiplier=-0.1)

    def test_unknown_allocation_rejected(self):
        with pytest.raises(ValueError, match="allocation must be"):
            noise_stddev(1.0, noise_multiplier=1.0, allocation="uniform")

    def test_negative_per_group_bound_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            noise_stddev(_bound([1.0, -2.0]), noise_multiplier=1.0)
