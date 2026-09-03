"""Portable execution tests must not select a provider directly."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_PROVIDER_MODULES = (
    "torch",
    "mlx",
    "opaque.torch",
    "opaque.mlx",
    "opaque.api.torch",
    "opaque.api.mlx",
)


def _imports_provider(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == provider or alias.name.startswith(f"{provider}.")
                for alias in node.names
                for provider in _PROVIDER_MODULES
            ):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            if any(
                node.module == provider or node.module.startswith(f"{provider}.")
                for provider in _PROVIDER_MODULES
            ):
                return True
            if node.module in {"opaque", "opaque.api"} and any(
                alias.name in {"torch", "mlx"} for alias in node.names
            ):
                return True
        elif (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value in _PROVIDER_MODULES
        ):
            target = node.func
            if (isinstance(target, ast.Name) and target.id == "__import__") or (
                isinstance(target, ast.Attribute) and target.attr == "import_module"
            ):
                return True
    return False


def test_portable_execution_modules_do_not_import_a_provider() -> None:
    portable_modules = sorted(
        path
        for path in (REPO_ROOT / "packages").glob(
            "opaque-*/tests/execution/portable/**/*.py"
        )
        if path.name != "__init__.py"
    )
    violations = [
        path.relative_to(REPO_ROOT)
        for path in portable_modules
        if _imports_provider(path)
    ]

    assert portable_modules, "Expected at least one portable execution test module"
    assert not violations, "Portable modules import a provider:\n" + "\n".join(
        f"  - {path}" for path in violations
    )
