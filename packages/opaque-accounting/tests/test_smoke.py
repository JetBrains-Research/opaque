"""Sanity checks for ``opaque.accounting`` (+ native build, ``opaque-core``).

Algorithm-specific PLD factories live in ``opaque.dpsgd.accounting`` /
``opaque.dpftrl.accounting``; their tests ship with those packages instead.
"""

from __future__ import annotations

import pytest


def test_import_root_accounting_surface():
    import opaque.accounting as acc

    assert hasattr(acc, "nonprivate")
    assert hasattr(acc, "compose")
    assert hasattr(acc, "calibrate")


def test_invalid_generic_mechanism_parameter_raises_value_error():
    import opaque.accounting as acc

    with pytest.raises(ValueError, match="epsilon must be non-negative"):
        acc.eps_delta(-1.0).pld()


@pytest.mark.parametrize(
    "mechanism",
    ["identity", "eps_delta"],
)
def test_mixed_estimate_modes_raise_runtime_error(mechanism: str):
    import opaque.accounting as acc

    process = acc.identity() if mechanism == "identity" else acc.eps_delta(1.0, 1e-5)
    pessimistic = process.pld(pessimistic_estimate=True)
    optimistic = process.pld(pessimistic_estimate=False)

    with pytest.raises(RuntimeError, match="Pessimistic estimate settings differ"):
        pessimistic.compose(optimistic)
