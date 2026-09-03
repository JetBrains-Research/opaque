"""Backend-free tests for the ``PerGroup`` value object."""

import pytest

from opaque.types import PerGroup


class TestPerGroup:
    """Tests for the PerGroup dataclass."""

    def test_basic_construction(self):
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 1.0, "g2": 2.0},
        )
        # str keys normalize to one-segment ParamPaths
        assert pg.groups == {("a",): "g1", ("b",): "g2"}
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

    def test_for_path(self):
        pg = PerGroup(
            groups={"param_a": "attn", "param_b": "mlp"},
            values={"attn": 1.0, "mlp": 2.0},
        )
        assert pg.for_path("param_a") == 1.0
        assert pg.for_path("param_b") == 2.0

    def test_for_path_missing_raises(self):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": 1.0})
        with pytest.raises(KeyError):
            pg.for_path("nonexistent")

    def test_frozen(self):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": 1.0})
        with pytest.raises(AttributeError):
            pg.groups = {}

    @pytest.mark.parametrize(
        ("attribute", "key", "value"),
        [
            ("groups", ("a",), "g2"),
            ("values", "g1", 2.0),
        ],
    )
    def test_mappings_are_immutable(self, attribute, key, value):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": 1.0})

        with pytest.raises(TypeError):
            getattr(pg, attribute)[key] = value

    def test_constructor_inputs_are_defensively_copied(self):
        groups = {"a": "g1"}
        values = {"g1": 1.0}
        pg = PerGroup(groups=groups, values=values)

        groups["a"] = "g2"
        values["g1"] = 2.0

        assert pg.groups == {("a",): "g1"}
        assert pg.values == {"g1": 1.0}

    def test_arithmetic_chain(self):
        """noise_multiplier * (clipping_norm / normalize_by) should work."""
        pg = PerGroup(groups={"a": "g1", "b": "g2"}, values={"g1": 2.0, "g2": 4.0})
        sensitivity = pg / 10.0  # normalize_by=10
        stddev = 1.1 * sensitivity  # noise_multiplier=1.1
        assert stddev.values["g1"] == pytest.approx(0.22)
        assert stddev.values["g2"] == pytest.approx(0.44)
        with pytest.raises(TypeError):
            stddev.values["g1"] = 1.0
