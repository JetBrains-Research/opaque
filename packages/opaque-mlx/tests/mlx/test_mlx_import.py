"""Lazy public-import coverage for the optional MLX backend."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_facade_import_does_not_load_mlx() -> None:
    """The façade remains importable when MLX is not available."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import opaque.mlx as facade; "
            "assert facade.__all__ == ['mlx_backend']; "
            "assert 'mlx' not in sys.modules; "
            "assert 'mlx.core' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_factory_reports_missing_mlx_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory calls fail with installation guidance rather than an import leak."""
    import opaque.api.mlx.backend as backend_module

    real_import_module = backend_module.importlib.import_module

    def _missing_mlx(name: str):
        if name == "mlx.core":
            raise ModuleNotFoundError("No module named 'mlx'", name="mlx")
        return real_import_module(name)

    monkeypatch.setattr(backend_module.importlib, "import_module", _missing_mlx)

    from opaque.mlx import mlx_backend

    with pytest.raises(ImportError, match="MLX support requires the 'mlx' dependency"):
        mlx_backend()
