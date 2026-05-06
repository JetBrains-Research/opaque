"""Sanity checks that run with only ``opaque-accounting`` (+ its native build).

Algorithm-specific PLD factories live in ``opaque.dpsgd.accounting`` /
``opaque.dpftrl.accounting``; their tests ship with those packages instead.
"""

from __future__ import annotations


def test_import_root_accounting_surface():
    import opaque.accounting as acc

    assert hasattr(acc, "nonprivate")
    assert hasattr(acc, "compose")
    assert hasattr(acc, "second_moment")
