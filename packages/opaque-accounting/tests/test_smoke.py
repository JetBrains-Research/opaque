"""Sanity checks for ``opaque.accounting`` (+ native build, ``opaque-core``).

Algorithm-specific PLD factories live in ``opaque.dpsgd.accounting`` /
``opaque.dpftrl.accounting``; their tests ship with those packages instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


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


@pytest.mark.parametrize("count", [0, -1])
@pytest.mark.parametrize(
    "compose",
    [
        lambda pld, count: pld.self_compose(count),
        lambda pld, count: pld * count,
        lambda pld, count: count * pld,
    ],
)
def test_invalid_pld_self_composition_count_raises_value_error(compose, count: int):
    import opaque.accounting as acc

    pld = acc.eps_delta(1.0, 1e-5).pld()

    with pytest.raises(ValueError, match="count must be greater than zero"):
        compose(pld, count)


@pytest.mark.parametrize(
    "compose",
    [
        lambda pld, count: pld.self_compose(count),
        lambda pld, count: pld * count,
        lambda pld, count: count * pld,
    ],
)
def test_overflowing_pld_self_composition_count_raises_overflow_error(compose):
    import opaque.accounting as acc

    pld = acc.eps_delta(1.0, 1e-5).pld()

    with pytest.raises(OverflowError, match="count must not exceed"):
        compose(pld, 2**32)


def test_discretization_config_stub_matches_native_surface():
    stub_path = (
        Path(__file__).parents[1]
        / "src/opaque/api/accounting/core/opaque_accounting.pyi"
    )
    module = ast.parse(stub_path.read_text())
    config = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DiscretizationConfig"
    )
    members = {
        node.name
        for node in config.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    docstring = ast.get_docstring(config)

    assert {"tail_mass_truncation", "num_mc_samples", "seed"} <= members
    assert docstring is not None
    for argument in ("tail_mass_truncation", "num_mc_samples", "seed"):
        assert f"{argument}:" in docstring
