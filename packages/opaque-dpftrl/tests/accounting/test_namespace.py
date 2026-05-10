"""Surface tests for ``opaque.dpftrl.accounting``.

Mirror of the ``opaque.dpsgd.accounting`` surface test: confirms each
documented re-export is reachable, ``__all__`` matches spec, and that
the lazy-import in :mod:`opaque.dpftrl` does not eagerly load
``opaque.accounting`` for callers that only need ``noise`` / ``sampling``.
"""

from __future__ import annotations

import subprocess
import sys


_HEADLINE = (
    "band_mf",
    "blt",
    "bisr",
    "bsr",
    "lambda_cgd",
    "identity_mf",
    "poisson",
    "b_min_sep",
    "balls_in_bins",
)

_TYPES = (
    "BandMf",
    "Blt",
    "Bisr",
    "Bsr",
    "LambdaCgd",
    "IdentityMf",
    "MfGaussian",
    "PoissonMf",
    "BMinSep",
    "BallsInBins",
)


class TestNamespaceSurface:
    """Each documented re-export is reachable as a callable."""

    def test_all_headline_factories_callable(self):
        import opaque.dpftrl.accounting as ftrl_acc

        for name in _HEADLINE:
            assert callable(getattr(ftrl_acc, name)), name

    def test_all_in_dunder_all(self):
        import opaque.dpftrl.accounting as ftrl_acc

        assert set(ftrl_acc.__all__) == set(_HEADLINE)

    def test_types_module_re_exports(self):
        import opaque.dpftrl.accounting.types as ftrl_types

        for name in _TYPES:
            assert hasattr(ftrl_types, name), name
        assert set(ftrl_types.__all__) == set(_TYPES)


class TestEndToEndCalibration:
    """Constructed mechanisms compute valid PLDs through the new namespace."""

    def test_band_mf_poisson(self):
        import math
        import opaque.dpftrl.accounting as ftrl_acc

        proc = ftrl_acc.poisson(
            ftrl_acc.band_mf(1.0, sensitivity=1.0, coefficients=(1.0, 0.5)),
            sample_rate=0.01,
            n_steps=20,
        )
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_blt_standalone(self):
        import math
        import opaque.dpftrl.accounting as ftrl_acc

        proc = ftrl_acc.blt(1.0, sensitivity=1.0)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestLazyImport:
    """``import opaque.dpftrl`` must not eagerly load ``opaque.accounting``."""

    def test_opaque_accounting_not_loaded_by_dpftrl_import(self):
        """Run a fresh subprocess so module cache is clean.

        ``opaque.dpftrl``'s ``__init__`` must defer the ``accounting``
        subpackage import (PEP 562 ``__getattr__``) so callers that only
        need noise / sampling do not pay the Rust PLD extension's
        startup cost.
        """
        code = (
            "import sys\n"
            "import opaque.dpftrl  # noqa: F401\n"
            "loaded = 'opaque.accounting' in sys.modules\n"
            "print('LOADED' if loaded else 'NOT_LOADED')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "NOT_LOADED" in result.stdout, (
            f"opaque.accounting was eagerly loaded by 'import opaque.dpftrl'. "
            f"stdout: {result.stdout}; stderr: {result.stderr}"
        )

    def test_accessing_accounting_loads_it(self):
        """First ``opaque.dpftrl.accounting`` access triggers the import."""
        code = (
            "import sys\n"
            "import opaque.dpftrl\n"
            "_ = opaque.dpftrl.accounting  # triggers PEP 562 __getattr__\n"
            "loaded = 'opaque.accounting' in sys.modules\n"
            "print('LOADED' if loaded else 'NOT_LOADED')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "LOADED" in result.stdout, result.stdout
