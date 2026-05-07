"""Tests for ``per_group_paired_noise_stddevs`` — MSE-optimal joint
allocation for the paired first + second-moment Gaussian release with
per-group sensitivities.
"""

import math

import pytest

from opaque.dpsgd.noise import per_group_paired_noise_stddevs
from opaque.types import PerGroup


def _make_bound(values, prefix="g", normalize_by=1.0):
    """Build a ``PerGroup`` with one parameter per group (1:1 mapping)."""
    groups = {f"p{i}": f"{prefix}{i}" for i in range(len(values))}
    vals = {f"{prefix}{i}": v / normalize_by for i, v in enumerate(values)}
    return PerGroup(groups=groups, values=vals)


class TestPerGroupPairedNoiseStddevs:
    def test_returns_two_pergroups_with_matching_keys(self):
        first = _make_bound([1.0, 2.0, 3.0])
        squared = _make_bound([1.0, 4.0, 9.0])
        s_first, s_second = per_group_paired_noise_stddevs(first, squared, 1.0)
        assert isinstance(s_first, PerGroup)
        assert isinstance(s_second, PerGroup)
        assert s_first.groups == first.groups
        assert s_second.groups == squared.groups
        assert set(s_first.values) == set(first.values)
        assert set(s_second.values) == set(squared.values)

    def test_raises_for_scalar_inputs(self):
        with pytest.raises(TypeError, match="PerGroup first_max_norm"):
            per_group_paired_noise_stddevs(1.0, 1.0, 1.0)
        first = _make_bound([1.0, 2.0])
        with pytest.raises(TypeError, match="PerGroup squared_max_norm"):
            per_group_paired_noise_stddevs(first, 1.0, 1.0)

    def test_raises_for_mismatched_groups_mapping(self):
        first = _make_bound([1.0, 2.0])
        # Same group names, different parameter assignments.
        squared = PerGroup(
            groups={"q0": "g0", "q1": "g1"},
            values={"g0": 1.0, "g1": 4.0},
        )
        with pytest.raises(ValueError, match="same groups mapping"):
            per_group_paired_noise_stddevs(first, squared, 1.0)

    def test_raises_for_mismatched_group_sets(self):
        first = _make_bound([1.0, 2.0])
        squared = PerGroup(
            groups=first.groups,
            values={"g0": 1.0, "g99": 4.0},
        )
        with pytest.raises(ValueError, match="identical group sets"):
            per_group_paired_noise_stddevs(first, squared, 1.0)

    def test_raises_for_negative_noise_multiplier(self):
        first = _make_bound([1.0, 2.0])
        squared = _make_bound([1.0, 4.0])
        with pytest.raises(ValueError, match="non-negative"):
            per_group_paired_noise_stddevs(first, squared, -0.5)

    def test_raises_for_negative_sensitivities(self):
        first = _make_bound([1.0, -2.0])
        squared = _make_bound([1.0, 4.0])
        with pytest.raises(ValueError, match="first-stream"):
            per_group_paired_noise_stddevs(first, squared, 1.0)
        first2 = _make_bound([1.0, 2.0])
        squared2 = _make_bound([-1.0, 4.0])
        with pytest.raises(ValueError, match="second-stream"):
            per_group_paired_noise_stddevs(first2, squared2, 1.0)

    def test_mahalanobis_constraint_holds_with_equality(self):
        """``Σ (Δ¹_g/σ¹_g)² + (Δ²_g/σ²_g)² == 1/nm²`` for all configs.

        The MSE-optimal joint allocation is calibrated so the joint
        mechanism's Mahalanobis distance equals 1/nm — equivalent to
        a single Gaussian release with that ``noise_multiplier``.
        """
        cases = [
            ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
            ([0.5, 2.0], [0.25, 4.0]),  # Δ²_g = (Δ¹_g)² (typical SM use)
            ([0.1, 0.4, 0.9, 1.6], [0.01, 0.16, 0.81, 2.56]),
        ]
        for first_vals, squared_vals in cases:
            for nm in [0.3, 1.0, 2.5]:
                for normalize_by in [1.0, 16.0, 128.0]:
                    first = _make_bound(first_vals, normalize_by=normalize_by)
                    squared = _make_bound(
                        squared_vals, normalize_by=normalize_by
                    )
                    s_first, s_second = per_group_paired_noise_stddevs(
                        first, squared, nm
                    )
                    mahal = sum(
                        first.values[g] ** 2 / s_first.values[g] ** 2
                        + squared.values[g] ** 2 / s_second.values[g] ** 2
                        for g in first.values
                    )
                    assert mahal == pytest.approx(1.0 / nm**2, rel=1e-10), (
                        f"Mahalanobis violated: {mahal} != {1 / nm**2} for "
                        f"first={first_vals}, sq={squared_vals}, nm={nm}, "
                        f"n={normalize_by}"
                    )

    def test_stddev_proportional_to_sqrt_sensitivity(self):
        """σ_g / σ_h = √(Δ_g / Δ_h) within each stream."""
        first = _make_bound([0.25, 1.0, 4.0])
        squared = _make_bound([0.0625, 1.0, 16.0])
        s_first, s_second = per_group_paired_noise_stddevs(first, squared, 1.0)
        # First stream: ratios should match sqrt(Δ¹) ratios.
        keys = list(first.values)
        for g, h in [(keys[0], keys[1]), (keys[1], keys[2])]:
            ratio = s_first.values[g] / s_first.values[h]
            expected = math.sqrt(first.values[g] / first.values[h])
            assert ratio == pytest.approx(expected)
        for g, h in [(keys[0], keys[1]), (keys[1], keys[2])]:
            ratio = s_second.values[g] / s_second.values[h]
            expected = math.sqrt(squared.values[g] / squared.values[h])
            assert ratio == pytest.approx(expected)

    def test_explicit_formula(self):
        """Direct check: ``σ_g = nm · sqrt(Δ_g · S)`` per stream."""
        first = _make_bound([1.0, 2.0])
        squared = _make_bound([1.0, 4.0])
        nm = 1.5
        s_first, s_second = per_group_paired_noise_stddevs(first, squared, nm)
        s = sum(first.values.values()) + sum(squared.values.values())  # = 8.0
        for g in first.values:
            assert s_first.values[g] == pytest.approx(
                nm * math.sqrt(first.values[g] * s)
            )
            assert s_second.values[g] == pytest.approx(
                nm * math.sqrt(squared.values[g] * s)
            )

    def test_noise_multiplier_zero_gives_zero_stddevs(self):
        first = _make_bound([1.0, 2.0])
        squared = _make_bound([1.0, 4.0])
        s_first, s_second = per_group_paired_noise_stddevs(first, squared, 0.0)
        assert all(v == 0.0 for v in s_first.values.values())
        assert all(v == 0.0 for v in s_second.values.values())

    def test_zero_sensitivity_stays_zero(self):
        """A group with zero sensitivity gets zero noise on that stream."""
        first = _make_bound([0.0, 1.0])
        squared = _make_bound([0.0, 1.0])
        s_first, s_second = per_group_paired_noise_stddevs(first, squared, 1.0)
        keys = list(first.values)
        # Group 0 has zero sensitivity on both streams → zero noise on both.
        assert s_first.values[keys[0]] == 0.0
        assert s_second.values[keys[0]] == 0.0
        # Group 1 still gets non-zero noise.
        assert s_first.values[keys[1]] > 0.0
        assert s_second.values[keys[1]] > 0.0


# ----- End-to-end: gaussian_noise + per-group paired clipping ----------------


class TestGaussianNoisePerGroupPairedIntegration:
    """``gaussian_noise`` accepts a paired ``SecondMomentClippingOutput`` whose
    streams carry per-group ``max_norm`` and emits per-group stddevs on both."""

    def test_per_group_paired_returns_per_group_stddevs(self):
        import torch

        from opaque.dpsgd.noise import gaussian_noise
        from opaque.random import key
        from opaque.types import (
            SecondMomentClippingOutput,
            SecondMomentNoiseOutput,
            clipped,
        )

        first_norm = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 1.0, "g2": 3.0},
        )
        squared_norm = first_norm * first_norm
        paired = SecondMomentClippingOutput(
            grads=clipped(
                {"a": torch.zeros(4), "b": torch.zeros(4)}, max_norm=first_norm
            ),
            squared_grads=clipped(
                {"a": torch.zeros(4), "b": torch.zeros(4)},
                max_norm=squared_norm,
            ),
        )
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        out, _ = noise_fn(paired, state)
        assert isinstance(out, SecondMomentNoiseOutput)
        assert isinstance(out.noisy_grads.noise_stddev, PerGroup)
        assert isinstance(out.noisy_squared_grads.noise_stddev, PerGroup)
        # Verify the joint-allocation formula (S = Σ Δ¹ + Σ Δ²).
        s = sum(first_norm.values.values()) + sum(squared_norm.values.values())
        for g in first_norm.values:
            assert out.noisy_grads.noise_stddev.values[g] == pytest.approx(
                math.sqrt(first_norm.values[g] * s)
            )
            assert out.noisy_squared_grads.noise_stddev.values[
                g
            ] == pytest.approx(math.sqrt(squared_norm.values[g] * s))

    def test_per_group_paired_mismatched_kinds_rejected(self):
        import torch

        from opaque.dpsgd.noise import gaussian_noise
        from opaque.random import key
        from opaque.types import SecondMomentClippingOutput, clipped

        per_group_norm = PerGroup(
            groups={"weight": "g"}, values={"g": 1.0}
        )
        paired = SecondMomentClippingOutput(
            grads=clipped({"weight": torch.zeros(4)}, max_norm=per_group_norm),
            squared_grads=clipped({"weight": torch.zeros(4)}, max_norm=1.0),
        )
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        with pytest.raises(TypeError, match="matching max_norm kinds"):
            noise_fn(paired, state)
