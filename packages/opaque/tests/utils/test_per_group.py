"""Tests for PerGroup type and per_group helper."""

import pytest
import torch

from opaque.utils.per_group import PerGroup, per_group


class TestPerGroup:
    """Tests for the PerGroup dataclass."""

    def test_basic_construction(self):
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 1.0, "g2": 2.0},
        )
        assert pg.groups == {"a": "g1", "b": "g2"}
        assert pg.values == {"g1": 1.0, "g2": 2.0}

    def test_effective(self):
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 3.0, "g2": 4.0},
        )
        assert pg.effective == pytest.approx(5.0)  # sqrt(9 + 16)

    def test_effective_single_group(self):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": 7.0})
        assert pg.effective == pytest.approx(7.0)

    def test_rmul(self):
        pg = PerGroup(groups={"a": "g1", "b": "g2"}, values={"g1": 1.0, "g2": 2.0})
        result = 3.0 * pg
        assert isinstance(result, PerGroup)
        assert result.values == {"g1": 3.0, "g2": 6.0}
        assert result.groups == pg.groups

    def test_mul(self):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": 5.0})
        result = pg * 2.0
        assert result.values == {"g1": 10.0}

    def test_truediv(self):
        pg = PerGroup(groups={"a": "g1", "b": "g2"}, values={"g1": 6.0, "g2": 10.0})
        result = pg / 2.0
        assert result.values == {"g1": 3.0, "g2": 5.0}

    def test_for_key(self):
        pg = PerGroup(
            groups={"param_a": "attn", "param_b": "mlp"},
            values={"attn": 1.0, "mlp": 2.0},
        )
        assert pg.for_key("param_a") == 1.0
        assert pg.for_key("param_b") == 2.0

    def test_for_key_missing_raises(self):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": 1.0})
        with pytest.raises(KeyError):
            pg.for_key("nonexistent")

    def test_frozen(self):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": 1.0})
        with pytest.raises(AttributeError):
            pg.groups = {}

    def test_arithmetic_chain(self):
        """noise_multiplier * (clipping_norm / normalize_by) should work."""
        pg = PerGroup(groups={"a": "g1", "b": "g2"}, values={"g1": 2.0, "g2": 4.0})
        sensitivity = pg / 10.0  # normalize_by=10
        stddev = 1.1 * sensitivity  # noise_multiplier=1.1
        assert stddev.values["g1"] == pytest.approx(0.22)
        assert stddev.values["g2"] == pytest.approx(0.44)


class TestPerGroupHelper:
    """Tests for the per_group() factory."""

    def test_flat_dict_two_groups(self):
        params = {
            "layers.0.self_attn.q_proj.weight": torch.zeros(1),
            "layers.0.self_attn.k_proj.weight": torch.zeros(1),
            "layers.0.mlp.gate_proj.weight": torch.zeros(1),
            "layers.0.mlp.up_proj.weight": torch.zeros(1),
        }
        pg = per_group(params, self_attn=1.0, mlp=2.0)
        assert pg.groups["layers.0.self_attn.q_proj.weight"] == "self_attn"
        assert pg.groups["layers.0.self_attn.k_proj.weight"] == "self_attn"
        assert pg.groups["layers.0.mlp.gate_proj.weight"] == "mlp"
        assert pg.groups["layers.0.mlp.up_proj.weight"] == "mlp"
        assert pg.values == {"self_attn": 1.0, "mlp": 2.0}

    def test_patterns_dict_for_dotted_keys(self):
        params = {
            "layers.0.weight": torch.zeros(1),
            "layers.1.weight": torch.zeros(1),
        }
        pg = per_group(params, patterns={"layers.0": 0.5, "layers.1": 1.0})
        assert pg.groups["layers.0.weight"] == "layers.0"
        assert pg.groups["layers.1.weight"] == "layers.1"

    def test_patterns_merged_with_kwargs(self):
        params = {
            "layers.0.attn.weight": torch.zeros(1),
            "layers.0.mlp.weight": torch.zeros(1),
        }
        pg = per_group(params, patterns={"layers.0.attn": 0.5}, mlp=1.0)
        assert pg.groups["layers.0.attn.weight"] == "layers.0.attn"
        assert pg.groups["layers.0.mlp.weight"] == "mlp"

    def test_nested_dict(self):
        params = {
            "layer1": {"attn": torch.zeros(1), "mlp": torch.zeros(1)},
            "layer2": {"attn": torch.zeros(1), "mlp": torch.zeros(1)},
        }
        pg = per_group(params, attn=1.0, mlp=2.0)
        assert pg.groups["layer1.attn"] == "attn"
        assert pg.groups["layer1.mlp"] == "mlp"
        assert pg.groups["layer2.attn"] == "attn"
        assert pg.groups["layer2.mlp"] == "mlp"

    def test_no_match_raises(self):
        params = {"weight": torch.zeros(1)}
        with pytest.raises(ValueError, match="did not match any pattern"):
            per_group(params, attn=1.0)

    def test_multiple_matches_raises(self):
        params = {"self_attn_mlp": torch.zeros(1)}
        with pytest.raises(ValueError, match="matched multiple patterns"):
            per_group(params, self_attn=1.0, mlp=2.0)

    def test_no_patterns_raises(self):
        params = {"w": torch.zeros(1)}
        with pytest.raises(ValueError, match="At least one pattern"):
            per_group(params)

    def test_negative_value_raises(self):
        params = {"w": torch.zeros(1)}
        with pytest.raises(ValueError, match="must be positive"):
            per_group(params, w=-1.0)

    def test_zero_value_raises(self):
        params = {"w": torch.zeros(1)}
        with pytest.raises(ValueError, match="must be positive"):
            per_group(params, w=0.0)

    def test_many_fine_grained_groups(self):
        """Per-layer clipping norms."""
        params = {f"layers.{i}.weight": torch.zeros(1) for i in range(4)}
        norms = {f"layers.{i}": float(i + 1) for i in range(4)}
        pg = per_group(params, **norms)
        assert len(pg.values) == 4
        assert pg.for_key("layers.2.weight") == 3.0

    def test_other_catches_unmatched(self):
        """'other' pattern catches params that don't match any explicit pattern."""
        params = {
            "layers.0.self_attn.q_proj.weight": torch.zeros(1),
            "layers.0.mlp.gate_proj.weight": torch.zeros(1),
            "layers.0.norm.weight": torch.zeros(1),
        }
        pg = per_group(params, self_attn=1.0, mlp=2.0, other=0.5)
        assert pg.groups["layers.0.self_attn.q_proj.weight"] == "self_attn"
        assert pg.groups["layers.0.mlp.gate_proj.weight"] == "mlp"
        assert pg.groups["layers.0.norm.weight"] == "other"
        assert pg.values["other"] == 0.5

    def test_other_not_in_values_when_unused(self):
        """If all params match explicit patterns, 'other' is excluded from values."""
        params = {
            "attn.weight": torch.zeros(1),
            "mlp.weight": torch.zeros(1),
        }
        pg = per_group(params, attn=1.0, mlp=2.0, other=0.5)
        assert "other" not in pg.values
        assert pg.effective == pytest.approx((1.0**2 + 2.0**2) ** 0.5)

    def test_other_effective_includes_other(self):
        """Effective norm should include the 'other' group."""
        params = {
            "attn.weight": torch.zeros(1),
            "unknown.weight": torch.zeros(1),
        }
        pg = per_group(params, attn=3.0, other=4.0)
        assert pg.effective == pytest.approx(5.0)  # sqrt(9 + 16)

    def test_no_match_error_suggests_other(self):
        """Error message suggests adding 'other' when param unmatched."""
        params = {"weight": torch.zeros(1)}
        with pytest.raises(ValueError, match="other"):
            per_group(params, attn=1.0)

    def test_other_only(self):
        """'other' as the sole group catches everything."""
        params = {"a": torch.zeros(1), "b": torch.zeros(1)}
        pg = per_group(params, other=1.0)
        assert pg.groups == {"a": "other", "b": "other"}
        assert pg.values == {"other": 1.0}


class TestOptimalNoiseStddev:
    """Tests for PerGroup.optimal_noise_stddev (MSE-optimal allocation)."""

    def _make_pg(self, values):
        groups = {f"p{i}": f"g{i}" for i in range(len(values))}
        vals = {f"g{i}": v for i, v in enumerate(values)}
        return PerGroup(groups=groups, values=vals)

    def test_returns_pergroup(self):
        pg = self._make_pg([1.0, 2.0, 3.0])
        result = pg.optimal_noise_stddev(1.0)
        assert isinstance(result, PerGroup)
        assert result.groups == pg.groups

    def test_mahalanobis_constraint_satisfied(self):
        """Σ v_i² / σ_i² == 1/nm² for all configurations."""
        import math

        for values in [[1.0, 1.0, 1.0], [0.1, 0.5, 2.0], [0.01, 0.1, 1.0, 5.0]]:
            for nm in [0.5, 1.0, 2.0]:
                pg = self._make_pg(values)
                stddev = pg.optimal_noise_stddev(nm)
                # Check Σ v_i² / σ_i² = 1/nm²
                mahal = sum(
                    v ** 2 / stddev.values[g] ** 2
                    for g, v in pg.values.items()
                )
                assert mahal == pytest.approx(1.0 / nm**2, rel=1e-10), (
                    f"Mahalanobis constraint violated: {mahal} != {1/nm**2} "
                    f"for values={values}, nm={nm}"
                )

    def test_stddev_proportional_to_sqrt_sensitivity(self):
        """σ_i / σ_j = √(v_i / v_j) for any two groups."""
        import math

        pg = self._make_pg([0.1, 0.4, 0.9])
        stddev = pg.optimal_noise_stddev(1.0)
        vals = list(stddev.values.values())
        sens = [0.1, 0.4, 0.9]
        # σ_i ∝ √v_i → σ_i/σ_j = √(v_i/v_j)
        ratio_01 = vals[0] / vals[1]
        expected_01 = math.sqrt(sens[0] / sens[1])
        assert ratio_01 == pytest.approx(expected_01, rel=1e-10)

    def test_single_group_equals_proportional(self):
        """With K=1, optimal = proportional = isotropic."""
        pg = self._make_pg([2.0])
        stddev = pg.optimal_noise_stddev(1.5)
        # σ = nm * √(v * v) = nm * v
        assert list(stddev.values.values())[0] == pytest.approx(1.5 * 2.0)

    def test_equal_sensitivities(self):
        """With equal v_i, σ_i should all be the same (= isotropic)."""
        pg = self._make_pg([1.0, 1.0, 1.0])
        stddev = pg.optimal_noise_stddev(2.0)
        vals = list(stddev.values.values())
        assert all(v == pytest.approx(vals[0]) for v in vals)

    def test_less_noise_on_small_groups(self):
        """Small-sensitivity groups get less noise than isotropic."""
        import math

        pg = self._make_pg([0.1, 1.0])
        nm = 1.0
        stddev_opt = pg.optimal_noise_stddev(nm)
        # Isotropic stddev
        iso = nm * pg.effective  # nm * √(0.01 + 1) ≈ 1.005
        # Optimal stddev for the small group
        small_opt = stddev_opt.values["g0"]
        assert small_opt < iso, (
            f"Optimal noise {small_opt} should be < isotropic {iso} for small group"
        )

    def test_mse_less_than_isotropic(self):
        """Total MSE of optimal allocation ≤ isotropic (strictly < when v_i differ)."""
        pg = self._make_pg([0.1, 0.5, 2.0])
        nm = 1.0
        stddev_opt = pg.optimal_noise_stddev(nm)
        iso_sigma = nm * pg.effective
        mse_opt = sum(v**2 for v in stddev_opt.values.values())
        mse_iso = len(pg.values) * iso_sigma**2
        assert mse_opt < mse_iso, (
            f"Optimal MSE {mse_opt} should be < isotropic MSE {mse_iso}"
        )
