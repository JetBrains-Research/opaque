"""MLX lazy-array serialization behavior."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from opaque.serialization import from_state_dict, state_dict


def test_mlx_serialization_materializes_an_independent_host_copy(monkeypatch) -> None:
    calls = []
    original_eval = mx.eval

    def record_eval(*values):
        calls.append(values)
        return original_eval(*values)

    monkeypatch.setattr(mx, "eval", record_eval)
    value = mx.array([1.0, 2.0], dtype=mx.float32)

    saved = state_dict({"buffer": value})
    saved["buffer"][0] = 99.0

    assert calls == [(value,)]
    np.testing.assert_array_equal(np.array(value), [1.0, 2.0])


def test_mlx_restore_follows_the_template_dtype_and_shape() -> None:
    saved = state_dict({"buffer": mx.array([1.0, 2.0], dtype=mx.float32)})

    restored = from_state_dict({"buffer": mx.zeros((2,), dtype=mx.float16)}, saved)

    assert restored["buffer"].dtype == mx.float16
    np.testing.assert_array_equal(np.array(restored["buffer"]), [1.0, 2.0])
    with pytest.raises(ValueError, match=r"shape \(2,\); template expects \(3,\)"):
        from_state_dict({"buffer": mx.zeros((3,))}, saved)
