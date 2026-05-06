"""Surface tests for ``opaque.dpsgd.accounting``.

The namespace is a thin re-export layer over :mod:`opaque.accounting`;
these tests pin the shape (every advertised symbol is reachable, the
``__all__`` list matches the spec) and verify that the lazy-import
in :mod:`opaque.dpsgd` does not eagerly load ``opaque.accounting``
when the user only needs ``noise`` / ``clipping`` / ``sampling``.
"""

from __future__ import annotations

import subprocess
import sys


_HEADLINE = (
    "gaussian",
    "adaclip",
    "poisson",
    "truncated_poisson",
    "parallel_poisson",
)

_TYPES = (
    "Gaussian",
    "AdaClip",
    "Poisson",
    "TruncatedPoisson",
    "ParallelPoisson",
)


class TestNamespaceSurface:
    """Each documented re-export is reachable as a callable."""

    def test_all_headline_factories_callable(self):
        import opaque.dpsgd.accounting as dpsgd_acc

        for name in _HEADLINE:
            assert callable(getattr(dpsgd_acc, name)), name

    def test_all_in_dunder_all(self):
        import opaque.dpsgd.accounting as dpsgd_acc

        assert set(dpsgd_acc.__all__) == set(_HEADLINE)

    def test_types_module_re_exports(self):
        import opaque.dpsgd.accounting.types as dpsgd_types

        for name in _TYPES:
            assert hasattr(dpsgd_types, name), name
        assert set(dpsgd_types.__all__) == set(_TYPES)

    def test_factories_match_root_implementations(self):
        """The new namespace re-exports the *same* objects as ``opaque.accounting``."""
        import opaque.accounting as acc
        import opaque.dpsgd.accounting as dpsgd_acc

        for name in _HEADLINE:
            assert getattr(dpsgd_acc, name) is getattr(acc, name), name


class TestEndToEndCalibration:
    """Constructed mechanisms compute valid PLDs through the new namespace."""

    def test_poisson_gaussian(self):
        import math
        import opaque.dpsgd.accounting as dpsgd_acc

        proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01) * 1000
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_truncated_poisson_gaussian(self):
        import math
        import opaque.dpsgd.accounting as dpsgd_acc

        proc = dpsgd_acc.truncated_poisson(dpsgd_acc.gaussian(0.8), 0.01, 128, 10_000)
        eps = (proc * 100).epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestLazyImport:
    """``import opaque.dpsgd`` must not eagerly load ``opaque.accounting``."""

    def test_opaque_accounting_not_loaded_by_dpsgd_import(self):
        """Run a fresh subprocess so module cache is clean.

        ``opaque.dpsgd``'s ``__init__`` must defer the ``accounting``
        subpackage import (PEP 562 ``__getattr__``) so callers that only
        need clipping / noise / sampling do not pay the Rust PLD
        extension's startup cost.
        """
        code = (
            "import sys\n"
            "import opaque.dpsgd  # noqa: F401\n"
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
            f"opaque.accounting was eagerly loaded by 'import opaque.dpsgd'. "
            f"stdout: {result.stdout}; stderr: {result.stderr}"
        )

    def test_accessing_accounting_loads_it(self):
        """First ``opaque.dpsgd.accounting`` access triggers the import."""
        code = (
            "import sys\n"
            "import opaque.dpsgd\n"
            "_ = opaque.dpsgd.accounting  # triggers PEP 562 __getattr__\n"
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
