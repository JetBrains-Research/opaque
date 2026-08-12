"""First-party providers keep core, runtime, and serialization integrations split."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_first_party_provider_backend_layout() -> None:
    for provider in ("torch", "jax", "mlx"):
        backend = (
            REPO_ROOT
            / "packages"
            / f"opaque-{provider}"
            / "src"
            / "opaque"
            / "api"
            / provider
            / "backend"
        )
        assert {"_core.py", "_runtime.py", "_serialization.py"} <= {
            path.name for path in backend.iterdir()
        }


def test_jax_and_mlx_legacy_monoliths_are_absent() -> None:
    for provider in ("jax", "mlx"):
        backend = (
            REPO_ROOT
            / "packages"
            / f"opaque-{provider}"
            / "src"
            / "opaque"
            / "api"
            / provider
            / "backend"
        )
        assert not (backend / f"_{provider}.py").exists()
