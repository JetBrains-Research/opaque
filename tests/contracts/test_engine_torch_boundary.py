"""Torch imports in opaque-engine stay in providers or named compatibility seams."""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENGINE_SOURCE = REPO_ROOT / "packages" / "opaque-engine" / "src"

# The legacy backend retains its method-shaped compatibility API.  The RNG
# bridge exposes ``torch.Generator`` for callers that still require it, and
# distributed state keeps its Torch-specific exact scalar/device behavior.
# New backend-dispatched computation must live under ``backend/torch`` instead.
COMPATIBILITY_ALLOWLIST = frozenset(
    {
        pathlib.PurePosixPath("opaque/api/engine/backend/_torch.py"),
        pathlib.PurePosixPath("opaque/api/engine/distributed/_state.py"),
        pathlib.PurePosixPath("opaque/api/engine/random/_engine.py"),
        pathlib.PurePosixPath("opaque/api/engine/random/_helpers.py"),
    }
)
TORCH_PROVIDER_ROOT = pathlib.PurePosixPath("opaque/api/engine/backend/torch")


def _imports_torch(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                name.name == "torch" or name.name.startswith("torch.")
                for name in node.names
            ):
                return True
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "torch" or node.module.startswith("torch."))
        ):
            return True
    return False


def test_engine_torch_imports_are_isolated_to_provider_or_compatibility_seams() -> None:
    violations: list[str] = []
    for path in ENGINE_SOURCE.rglob("*.py"):
        if not _imports_torch(path):
            continue
        relative = pathlib.PurePosixPath(path.relative_to(ENGINE_SOURCE))
        if relative in COMPATIBILITY_ALLOWLIST or relative.is_relative_to(
            TORCH_PROVIDER_ROOT
        ):
            continue
        violations.append(str(relative))

    assert not violations, (
        "Direct Torch imports must live in the Torch provider or an explicit "
        "compatibility seam:\n" + "\n".join(f"  - {path}" for path in violations)
    )
