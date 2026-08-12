"""Lazy public-import coverage for the optional JAX backend."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_facade_import_does_not_load_jax() -> None:
    """The façade remains importable without loading JAX."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import opaque.jax as facade; "
            "assert facade.__all__ == ['jax_backend']; "
            "assert 'jax' not in sys.modules; "
            "assert 'jax.numpy' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_factory_reports_missing_jax_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory calls fail with installation guidance rather than an import leak."""
    import opaque.api.jax.backend as backend_module

    real_import_module = backend_module.importlib.import_module

    def _missing_jax(name: str):
        if name == "jax":
            raise ModuleNotFoundError("No module named 'jax'", name="jax")
        return real_import_module(name)

    monkeypatch.setattr(backend_module.importlib, "import_module", _missing_jax)

    from opaque.jax import jax_backend

    with pytest.raises(ImportError, match="JAX support requires the 'jax' dependency"):
        jax_backend()
