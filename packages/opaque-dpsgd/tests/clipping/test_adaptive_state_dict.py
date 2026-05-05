"""Round-trip tests for ``AdaptiveClipState.state_dict``."""

from __future__ import annotations

import json

from opaque.clipping.per_group import PerGroup
from opaque.dpsgd.clipping.adaptive import AdaptiveClipState
from opaque.random import RngKey


def _scalar_state(**overrides) -> AdaptiveClipState:
    base = dict(
        clipping_norm=1.5,
        normalize_by=2.0,
        next_clipping_norm=1.6,
        step=7,
        _rng_key=RngKey(seed=42),
        _fraction_noise_std=0.05,
        _learning_rate=0.1,
        _target_quantile=0.9,
        _clipping_norm_min=0.01,
        _clipping_norm_max=10.0,
        _num_clipped=3.0,
        _batch_size=64.0,
    )
    base.update(overrides)
    return AdaptiveClipState(**base)


class TestAdaptiveClipStateStateDict:
    def test_scalar_roundtrip(self):
        cs = _scalar_state()
        state = cs.state_dict()
        encoded = json.dumps(state)
        restored = AdaptiveClipState.from_state_dict(json.loads(encoded))
        assert restored == cs

    def test_per_group_norm_roundtrip(self):
        pg = PerGroup(
            groups={"layer.0.q": "attn", "layer.0.mlp": "mlp"},
            values={"attn": 1.0, "mlp": 2.0},
        )
        cs = _scalar_state(
            clipping_norm=pg,
            next_clipping_norm=PerGroup(
                groups=pg.groups,
                values={"attn": 1.1, "mlp": 2.1},
            ),
        )
        state = cs.state_dict()
        encoded = json.dumps(state)
        restored = AdaptiveClipState.from_state_dict(json.loads(encoded))
        assert restored == cs
        assert isinstance(restored.clipping_norm, PerGroup)
        assert isinstance(restored.next_clipping_norm, PerGroup)

    def test_per_group_num_clipped_roundtrip(self):
        cs = _scalar_state(_num_clipped={"attn": 2.0, "mlp": 5.0})
        state = cs.state_dict()
        encoded = json.dumps(state)
        restored = AdaptiveClipState.from_state_dict(json.loads(encoded))
        assert restored == cs
        assert restored._num_clipped == {"attn": 2.0, "mlp": 5.0}

    def test_rng_key_roundtrip(self):
        cs = _scalar_state(_rng_key=RngKey(seed=2**40, impl="custom_impl"))
        restored = AdaptiveClipState.from_state_dict(cs.state_dict())
        assert restored._rng_key == cs._rng_key
