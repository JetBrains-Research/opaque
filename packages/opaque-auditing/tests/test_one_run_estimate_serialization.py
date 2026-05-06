"""Checkpoint round-trip for :class:`OneRunEstimate` (NumPy fields)."""

from __future__ import annotations

import numpy as np

from opaque.auditing.types import OneRunEstimate
from opaque.serialization import from_state_dict, state_dict


def _minimal_estimate() -> OneRunEstimate:
    return OneRunEstimate(
        n_in=2,
        n_out=3,
        thresholds=np.array([0.1, 0.9], dtype=np.float64),
        tn_counts=np.array([1, 0], dtype=np.int64),
        fn_counts=np.array([0, 1], dtype=np.int64),
        tp_counts=np.array([0, 1], dtype=np.int64),
        fp_counts=np.array([1, 0], dtype=np.int64),
        in_scores=np.array([0.2, 0.8]),
        out_scores=np.array([0.15, 0.5, 0.85], dtype=np.float64),
    )


def test_one_run_estimate_roundtrip() -> None:
    est = _minimal_estimate()
    flat = state_dict(est)
    template = OneRunEstimate(
        n_in=0,
        n_out=0,
        thresholds=np.zeros(2, dtype=np.float64),
        tn_counts=np.zeros(2, dtype=np.int64),
        fn_counts=np.zeros(2, dtype=np.int64),
        tp_counts=np.zeros(2, dtype=np.int64),
        fp_counts=np.zeros(2, dtype=np.int64),
        in_scores=np.zeros(2, dtype=np.float64),
        out_scores=np.zeros(3, dtype=np.float64),
    )
    restored = from_state_dict(template, flat)
    assert restored.n_in == est.n_in
    assert restored.n_out == est.n_out
    np.testing.assert_array_equal(restored.thresholds, est.thresholds)
    np.testing.assert_array_equal(restored.out_scores, est.out_scores)
    np.testing.assert_array_equal(restored.in_scores, est.in_scores)
