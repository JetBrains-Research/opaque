"""Surface tests for ``opaque.dpftrl.accounting``."""

from __future__ import annotations

import subprocess
import sys

_REQUIRED_FACTORIES = (
    "mf_gaussian",
    "poisson",
    "b_min_sep",
    "balls_in_bins",
    "per_step",
)

_REQUIRED_TYPES = (
    "MfGaussian",
    "CyclicPoisson",
    "BMinSep",
    "BallsInBins",
    "DpFtrlProcess",
    "PerStep",
)


class TestNamespaceSurface:
    """Each documented re-export is reachable as a callable."""

    def test_required_headline_factories_callable(self):
        import opaque.dpftrl.accounting as ftrl_acc

        for name in _REQUIRED_FACTORIES:
            assert callable(getattr(ftrl_acc, name)), name

    def test_types_module_re_exports(self):
        import opaque.dpftrl.accounting.types as ftrl_types

        for name in _REQUIRED_TYPES:
            assert hasattr(ftrl_types, name), name


class TestEndToEndCalibration:
    """Constructed mechanisms compute valid PLDs through the new namespace."""

    def test_band_mf_poisson(self):
        import math

        import opaque.dpftrl.accounting as ftrl_acc
        from opaque.dpftrl.noise import band_mf_strategy

        strategy = band_mf_strategy(bands=2)
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, strategy),
            sample_rate=0.01,
            n_steps=20,
        )
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0

    def test_blt_standalone(self):
        import math

        import opaque.dpftrl.accounting as ftrl_acc
        from opaque.dpftrl.noise import blt_strategy

        s = blt_strategy(momentum=1.0)
        eps = ftrl_acc.mf_gaussian(
            1.0, s, n_steps=10, min_sep=10, max_participations=1
        ).epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0


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
