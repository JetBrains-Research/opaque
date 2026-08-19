"""Restore policy for ndarray leaves: shape is checked, dtype follows.

Restore is template-driven, so a checkpoint value that does not fit its slot
must fail at load rather than broadcast into a silently different run. The
engine owns the ``numpy.ndarray`` handlers and stays provider-free here; the
tensor half of the same rule is exercised against the Torch provider in
``packages/opaque-torch/tests/serialization``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from opaque.serialization import from_state_dict, state_dict


@dataclass
class _Buffers:
    """Inner state holding a tuple of array leaves."""

    values: tuple[Any, ...]


@dataclass
class _State:
    """Outer state: dataclass → mapping → dataclass → tuple → leaf."""

    inner: dict[str, _Buffers]
    step: int


def _nested(*leaves: Any) -> _State:
    return _State(inner={"noise": _Buffers(values=leaves)}, step=3)


def test_nested_shape_mismatch_rejected() -> None:
    """An ndarray mismatch raises ``ValueError``, naming the key."""
    saved = state_dict(_nested(np.zeros(2), np.zeros(3)))

    with pytest.raises(
        ValueError, match=r"shape \(3,\); template expects \(4,\)"
    ) as excinfo:
        from_state_dict(_nested(np.zeros(2), np.zeros(4)), saved)

    assert "inner.noise.values[1]" in " ".join(excinfo.value.__notes__)


def test_missing_leaf_keeps_template() -> None:
    """An absent key is forward compatibility, not a mismatch."""
    template = _nested(np.zeros(2), np.zeros(4))

    restored = from_state_dict(template, {})

    assert restored.inner["noise"].values[1] is template.inner["noise"].values[1]


def test_dtype_follows_the_template() -> None:
    """The template carries the live dtype; the saved value is cast to it."""
    saved = state_dict({"buf": np.ones(3, dtype=np.float64)})

    restored = from_state_dict({"buf": np.zeros(3, dtype=np.float32)}, saved)

    assert restored["buf"].dtype == np.float32
    assert np.array_equal(restored["buf"], np.ones(3, dtype=np.float32))
