"""Tests for per_group_noise_stddev — MSE-optimal noise allocation."""

import math

import pytest

from opaque.clipping.types import FixedClipState
from opaque.noise.per_group_noise import per_group_noise_stddev
from opaque.utils.per_group import PerGroup


def _make_clip_state(values, normalize_by=1.0):
    """Helper: create FixedClipState with PerGroup clipping_norm."""
    groups = {f"p{i}": f"g{i}" for i in range(len(values))}
    vals = {f"g{i}": v for i, v in enumerate(values)}
    pg = PerGroup(groups=groups, values=vals)
    return FixedClipState(clipping_norm=pg, normalize_by=normalize_by)


class TestPerGroupNoiseStddev:
    """Tests for the per_group_noise_stddev function."""

    def test_returns_pergroup(self):
        cs = _make_clip_state([1.0, 2.0, 3.0])
        result = per_group_noise_stddev(cs, 1.0)
        assert isinstance(result, PerGroup)
        assert result.groups == cs.clipping_norm.groups

    def test_raises_for_scalar_clipping(self):
        cs = FixedClipState(clipping_norm=1.0)
        with pytest.raises(TypeError, match="per-group clipping_norm"):
            per_group_noise_stddev(cs, 1.0)

    def test_mahalanobis_constraint(self):
        """Σ (C_i/n)² / σ_i² == 1/nm² for all configurations."""
        for values in [[1.0, 1.0, 1.0], [0.1, 0.5, 2.0], [0.01, 0.1, 1.0, 5.0]]:
            for nm in [0.5, 1.0, 2.0]:
                for n in [1.0, 10.0]:
                    cs = _make_clip_state(values, normalize_by=n)
                    stddev = per_group_noise_stddev(cs, nm)
                    # Check Σ (C_i/n)² / σ_i² = 1/nm²
                    mahal = sum(
                        (c / n) ** 2 / stddev.values[g] ** 2
                        for g, c in cs.clipping_norm.values.items()
                    )
                    assert mahal == pytest.approx(1.0 / nm**2, rel=1e-10), (
                        f"Mahalanobis constraint violated: {mahal} != {1/nm**2} "
                        f"for values={values}, nm={nm}, n={n}"
                    )

    def test_stddev_proportional_to_sqrt_clipping_norm(self):
        """σ_i / σ_j = √(C_i / C_j) for any two groups."""
        cs = _make_clip_state([0.1, 0.4, 0.9])
        stddev = per_group_noise_stddev(cs, 1.0)
        vals = list(stddev.values.values())
        norms = [0.1, 0.4, 0.9]
        ratio_01 = vals[0] / vals[1]
        expected_01 = math.sqrt(norms[0] / norms[1])
        assert ratio_01 == pytest.approx(expected_01, rel=1e-10)

    def test_single_group_equals_isotropic(self):
        """With K=1, per-group noise == isotropic noise."""
        cs = _make_clip_state([2.0], normalize_by=5.0)
        stddev = per_group_noise_stddev(cs, 1.5)
        # σ = nm * sqrt(C * C) / n = nm * C / n = nm * sensitivity
        expected = 1.5 * 2.0 / 5.0
        assert list(stddev.values.values())[0] == pytest.approx(expected)

    def test_equal_norms_gives_equal_stddev(self):
        """With equal C_i, all σ_i should be the same."""
        cs = _make_clip_state([1.0, 1.0, 1.0])
        stddev = per_group_noise_stddev(cs, 2.0)
        vals = list(stddev.values.values())
        assert all(v == pytest.approx(vals[0]) for v in vals)

    def test_less_noise_on_small_groups(self):
        """Small clipping norm groups get less noise than isotropic."""
        cs = _make_clip_state([0.1, 1.0])
        nm = 1.0
        stddev = per_group_noise_stddev(cs, nm)
        iso = nm * cs.sensitivity  # isotropic = nm * effective / n
        small_opt = stddev.values["g0"]
        assert small_opt < iso, (
            f"Optimal noise {small_opt} should be < isotropic {iso} for small group"
        )

    def test_mse_less_than_isotropic(self):
        """Total noise MSE < isotropic when clipping norms differ."""
        cs = _make_clip_state([0.1, 0.5, 2.0])
        nm = 1.0
        stddev = per_group_noise_stddev(cs, nm)
        iso_sigma = nm * cs.sensitivity
        mse_opt = sum(v**2 for v in stddev.values.values())
        mse_iso = len(cs.clipping_norm.values) * iso_sigma**2
        assert mse_opt < mse_iso, (
            f"Optimal MSE {mse_opt} should be < isotropic MSE {mse_iso}"
        )

    def test_normalize_by_scales_correctly(self):
        """Doubling normalize_by should halve all stddevs."""
        cs_1 = _make_clip_state([1.0, 2.0], normalize_by=1.0)
        cs_2 = _make_clip_state([1.0, 2.0], normalize_by=2.0)
        nm = 1.0
        stddev_1 = per_group_noise_stddev(cs_1, nm)
        stddev_2 = per_group_noise_stddev(cs_2, nm)
        for g in stddev_1.values:
            assert stddev_2.values[g] == pytest.approx(stddev_1.values[g] / 2.0)

    def test_linear_in_noise_multiplier(self):
        """Doubling nm should double all stddevs."""
        cs = _make_clip_state([1.0, 3.0])
        stddev_1 = per_group_noise_stddev(cs, 1.0)
        stddev_2 = per_group_noise_stddev(cs, 2.0)
        for g in stddev_1.values:
            assert stddev_2.values[g] == pytest.approx(2.0 * stddev_1.values[g])
