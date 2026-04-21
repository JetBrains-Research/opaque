"""Tests for PerGroup type and per_group helper."""

import pytest
import torch

from opaque.core.clipping.per_group import PerGroup, per_group


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

    def test_fallback_catches_unmatched(self):
        """fallback catches params that don't match any explicit pattern."""
        params = {
            "layers.0.self_attn.q_proj.weight": torch.zeros(1),
            "layers.0.mlp.gate_proj.weight": torch.zeros(1),
            "layers.0.norm.weight": torch.zeros(1),
        }
        pg = per_group(params, self_attn=1.0, mlp=2.0, fallback=0.5)
        assert pg.groups["layers.0.self_attn.q_proj.weight"] == "self_attn"
        assert pg.groups["layers.0.mlp.gate_proj.weight"] == "mlp"
        assert pg.groups["layers.0.norm.weight"] == "fallback"
        assert pg.values["fallback"] == 0.5

    def test_fallback_not_in_values_when_unused(self):
        """If all params match explicit patterns, fallback is excluded from values."""
        params = {
            "attn.weight": torch.zeros(1),
            "mlp.weight": torch.zeros(1),
        }
        pg = per_group(params, attn=1.0, mlp=2.0, fallback=0.5)
        assert "fallback" not in pg.values
        assert pg.effective == pytest.approx((1.0**2 + 2.0**2) ** 0.5)

    def test_fallback_effective_includes_fallback(self):
        """Effective norm should include the fallback group."""
        params = {
            "attn.weight": torch.zeros(1),
            "unknown.weight": torch.zeros(1),
        }
        pg = per_group(params, attn=3.0, fallback=4.0)
        assert pg.effective == pytest.approx(5.0)  # sqrt(9 + 16)

    def test_no_match_error_suggests_fallback(self):
        """Error message suggests using fallback when param unmatched."""
        params = {"weight": torch.zeros(1)}
        with pytest.raises(ValueError, match="fallback"):
            per_group(params, attn=1.0)

    def test_fallback_only(self):
        """fallback as the sole group catches everything."""
        params = {"a": torch.zeros(1), "b": torch.zeros(1)}
        pg = per_group(params, fallback=1.0)
        assert pg.groups == {"a": "fallback", "b": "fallback"}
        assert pg.values == {"fallback": 1.0}
