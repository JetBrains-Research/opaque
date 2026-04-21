# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the re-rooted ``opaque.accounting`` namespace."""

import pathlib

import pytest


def test_import_accounting():
    import opaque.accounting as acc

    assert acc.__version__
    assert callable(acc.gaussian)
    assert callable(acc.poisson)
    assert callable(acc.calibrate)
    assert acc.Accountant is not None
    assert acc.DiscretizationConfig is not None


def test_native_extension_location():
    """The PyO3 extension must install under ``opaque.accounting._native``."""
    from opaque.accounting import _native

    path = pathlib.Path(_native.__file__)
    assert path.name.startswith("_native"), f"unexpected native filename: {path.name}"
    # Must live inside the opaque/accounting package directory
    assert path.parent.name == "accounting", (
        f"expected _native under opaque/accounting/, got {path}"
    )
    assert path.parent.parent.name == "opaque", (
        f"expected _native under opaque/accounting/, got {path}"
    )


def test_top_level_opaque_accounting_is_gone():
    with pytest.raises(ModuleNotFoundError):
        __import__("opaque_accounting")


def test_public_symbols_available():
    from opaque.accounting import (  # noqa: F401
        Accountant,
        DiscretizationConfig,
        balls_in_bins,
        band_mf,
        bisr,
        blt,
        bsr,
        calibrate,
        compose,
        gaussian,
        identity,
        lambda_cgd,
        poisson,
        repeat,
    )
