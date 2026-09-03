"""Portable construction and validation behavior for per-group clipping."""

from __future__ import annotations

import pytest

from opaque import ops
from opaque.api.engine.clipping._per_group import per_group


@pytest.fixture(autouse=True)
def _activate_matrix(backend_case):
    return backend_case


def _zero():
    return ops.scalar(0.0)


class TestPerGroupHelper:
    """Tests for the per_group() factory."""

    def test_flat_dict_two_groups(self):
        params = {
            "layers.0.self_attn.q_proj.weight": _zero(),
            "layers.0.self_attn.k_proj.weight": _zero(),
            "layers.0.mlp.gate_proj.weight": _zero(),
            "layers.0.mlp.up_proj.weight": _zero(),
        }
        pg = per_group(params, self_attn=1.0, mlp=2.0)
        assert pg.groups[("layers.0.self_attn.q_proj.weight",)] == "self_attn"
        assert pg.groups[("layers.0.self_attn.k_proj.weight",)] == "self_attn"
        assert pg.groups[("layers.0.mlp.gate_proj.weight",)] == "mlp"
        assert pg.groups[("layers.0.mlp.up_proj.weight",)] == "mlp"
        assert pg.values == {"self_attn": 1.0, "mlp": 2.0}

    def test_patterns_dict_for_dotted_keys(self):
        params = {
            "layers.0.weight": _zero(),
            "layers.1.weight": _zero(),
        }
        pg = per_group(params, patterns={"layers.0": 0.5, "layers.1": 1.0})
        assert pg.groups[("layers.0.weight",)] == "layers.0"
        assert pg.groups[("layers.1.weight",)] == "layers.1"

    def test_patterns_merged_with_kwargs(self):
        params = {
            "layers.0.attn.weight": _zero(),
            "layers.0.mlp.weight": _zero(),
        }
        pg = per_group(params, patterns={"layers.0.attn": 0.5}, mlp=1.0)
        assert pg.groups[("layers.0.attn.weight",)] == "layers.0.attn"
        assert pg.groups[("layers.0.mlp.weight",)] == "mlp"

    def test_nested_dict(self):
        """Nested params compile to multi-segment optree paths."""
        params = {
            "layer1": {"attn": _zero(), "mlp": _zero()},
            "layer2": {"attn": _zero(), "mlp": _zero()},
        }
        pg = per_group(params, attn=1.0, mlp=2.0)
        assert pg.groups[("layer1", "attn")] == "attn"
        assert pg.groups[("layer1", "mlp")] == "mlp"
        assert pg.groups[("layer2", "attn")] == "attn"
        assert pg.groups[("layer2", "mlp")] == "mlp"

    def test_flat_named_parameters_are_one_segment_paths(self):
        params = {
            "layers.0.self_attn.weight": _zero(),
            "layers.0.mlp.weight": _zero(),
        }
        pg = per_group(params, self_attn=1.0, mlp=2.0)
        assert pg.groups[("layers.0.self_attn.weight",)] == "self_attn"
        assert ("layers", 0, "self_attn", "weight") not in pg.groups

    def test_no_match_raises(self):
        params = {"weight": _zero()}
        with pytest.raises(ValueError, match="did not match any pattern"):
            per_group(params, attn=1.0)

    def test_multiple_matches_raises(self):
        params = {"self_attn_mlp": _zero()}
        with pytest.raises(ValueError, match="matched multiple patterns"):
            per_group(params, self_attn=1.0, mlp=2.0)

    def test_no_patterns_raises(self):
        params = {"w": _zero()}
        with pytest.raises(ValueError, match="At least one pattern"):
            per_group(params)

    def test_negative_value_raises(self):
        params = {"w": _zero()}
        with pytest.raises(ValueError, match="must be positive"):
            per_group(params, w=-1.0)

    def test_zero_value_raises(self):
        params = {"w": _zero()}
        with pytest.raises(ValueError, match="must be positive"):
            per_group(params, w=0.0)

    def test_many_fine_grained_groups(self):
        """Per-layer clipping norms."""
        params = {f"layers.{i}.weight": _zero() for i in range(4)}
        norms = {f"layers.{i}": float(i + 1) for i in range(4)}
        pg = per_group(params, **norms)
        assert len(pg.values) == 4
        assert pg.for_path("layers.2.weight") == 3.0

    def test_fallback_catches_unmatched(self):
        """fallback catches params that don't match any explicit pattern."""
        params = {
            "layers.0.self_attn.q_proj.weight": _zero(),
            "layers.0.mlp.gate_proj.weight": _zero(),
            "layers.0.norm.weight": _zero(),
        }
        pg = per_group(params, self_attn=1.0, mlp=2.0, fallback=0.5)
        assert pg.groups[("layers.0.self_attn.q_proj.weight",)] == "self_attn"
        assert pg.groups[("layers.0.mlp.gate_proj.weight",)] == "mlp"
        assert pg.groups[("layers.0.norm.weight",)] == "fallback"
        assert pg.values["fallback"] == 0.5

    def test_fallback_not_in_values_when_unused(self):
        """If all params match explicit patterns, fallback is excluded from values."""
        params = {
            "attn.weight": _zero(),
            "mlp.weight": _zero(),
        }
        pg = per_group(params, attn=1.0, mlp=2.0, fallback=0.5)
        assert "fallback" not in pg.values
        assert pg.effective == pytest.approx((1.0**2 + 2.0**2) ** 0.5)

    def test_fallback_effective_includes_fallback(self):
        """Effective norm should include the fallback group."""
        params = {
            "attn.weight": _zero(),
            "unknown.weight": _zero(),
        }
        pg = per_group(params, attn=3.0, fallback=4.0)
        assert pg.effective == pytest.approx(5.0)  # sqrt(9 + 16)

    def test_no_match_error_suggests_fallback(self):
        """Error message suggests using fallback when param unmatched."""
        params = {"weight": _zero()}
        with pytest.raises(ValueError, match="fallback"):
            per_group(params, attn=1.0)

    def test_fallback_only(self):
        """fallback as the sole group catches everything."""
        params = {"a": _zero(), "b": _zero()}
        pg = per_group(params, fallback=1.0)
        assert pg.groups == {("a",): "fallback", ("b",): "fallback"}
        assert pg.values == {"fallback": 1.0}
