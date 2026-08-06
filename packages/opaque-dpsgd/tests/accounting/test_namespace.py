"""Surface tests for ``opaque.dpsgd.accounting``."""

from __future__ import annotations

import subprocess
import sys


class TestNamespaceSurface:
    """The declared public surface is importable, callable, and leak-free."""

    def test_all_declared_factories_callable(self):
        import opaque.dpsgd.accounting as dpsgd_acc

        exports = getattr(dpsgd_acc, "__all__", None)
        assert exports, "accounting facade must declare __all__"
        for name in exports:
            assert callable(getattr(dpsgd_acc, name)), name

    def test_all_declared_types_importable(self):
        import opaque.dpsgd.accounting.types as dpsgd_types

        exports = getattr(dpsgd_types, "__all__", None)
        assert exports, "types facade must declare __all__"
        for name in exports:
            assert hasattr(dpsgd_types, name), name

    def test_no_public_leak_outside_all(self):
        import types

        import opaque.dpsgd.accounting as dpsgd_acc

        declared = set(dpsgd_acc.__all__)
        leaked = {
            name
            for name in dir(dpsgd_acc)
            if not name.startswith("_")
            and name not in declared
            and not isinstance(getattr(dpsgd_acc, name), types.ModuleType)
            and (getattr(getattr(dpsgd_acc, name), "__module__", "") or "").startswith(
                "opaque"
            )
        }
        assert not leaked, f"public names not in __all__: {sorted(leaked)}"


class TestEndToEndCalibration:
    """Constructed mechanisms compute valid PLDs through the new namespace."""

    def test_poisson_gaussian(self):
        import math

        import opaque.dpsgd.accounting as dpsgd_acc

        proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01) * 1000
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0

    def test_truncated_poisson_gaussian(self):
        import math

        import opaque.dpsgd.accounting as dpsgd_acc

        proc = dpsgd_acc.poisson(
            dpsgd_acc.gaussian(0.8),
            0.01,
            truncated_batch_size=128,
            dataset_size=10_000,
        )
        eps = (proc * 100).epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0


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
