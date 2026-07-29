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


def test_incompatible_pld_operands_raise_value_error():
    import opaque.accounting as acc

    process = acc.eps_delta(1.0, 1e-5)
    fine = process.pld(discretization=0.1)
    incompatible = process.pld(discretization=0.3)

    with pytest.raises(ValueError, match="Discretization intervals differ"):
        fine.compose(incompatible)
