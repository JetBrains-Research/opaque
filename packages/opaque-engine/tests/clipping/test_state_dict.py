"""Round-trip tests for ``state_dict`` / ``from_state_dict`` on clipping types."""

from __future__ import annotations

import json

from opaque.types import PerGroup


class TestPerGroupStateDict:
    def test_roundtrip_via_json(self):
        pg = PerGroup(
            groups={"layer.0.q": "attn", "layer.0.mlp": "mlp"},
            values={"attn": 1.5, "mlp": 0.75},
        )
        state = pg.state_dict()
        # JSON-serializable
        encoded = json.dumps(state)
        decoded = json.loads(encoded)
        restored = PerGroup.from_state_dict(decoded)
        assert restored == pg

    def test_state_dict_returns_plain_dicts(self):
        pg = PerGroup(groups={"a": "g"}, values={"g": 2.0})
        state = pg.state_dict()
        assert state == {"groups": {"a": "g"}, "values": {"g": 2.0}}

    def test_int_values_coerced_to_float(self):
        state = {"groups": {"a": "g"}, "values": {"g": 3}}
        pg = PerGroup.from_state_dict(state)
        assert isinstance(pg.values["g"], float)
