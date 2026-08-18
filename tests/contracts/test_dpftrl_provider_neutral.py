"""``opaque-dpftrl`` must not depend on a provider runtime."""

from __future__ import annotations

import ast
import pathlib
import re
import tomllib

PACKAGES_DIR = pathlib.Path(__file__).resolve().parents[2] / "packages"
DPFTRL_DIR = PACKAGES_DIR / "opaque-dpftrl"
_PROVIDER_IMPORTS = (
    "torch",
    "jax",
    "mlx",
    "opaque.torch",
    "opaque.jax",
    "opaque.mlx",
    "opaque.api.torch",
    "opaque.api.jax",
    "opaque.api.mlx",
)


def _imports_provider(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "opaque" and any(
                alias.name in {"torch", "jax", "mlx"} for alias in node.names
            ):
                return True
            names = (node.module or "",)
        else:
            continue
        if any(
            name == provider or name.startswith(f"{provider}.")
            for name in names
            for provider in _PROVIDER_IMPORTS
        ):
            return True
    return False


def _dependency_name(spec: str) -> str:
    return re.split(r"[<>=!~;\s]", spec, maxsplit=1)[0].split("[", 1)[0].lower()


def test_dpftrl_sources_do_not_import_provider_runtime() -> None:
    violations = [
        path for path in (DPFTRL_DIR / "src").rglob("*.py") if _imports_provider(path)
    ]
    assert not violations, (
        "opaque-dpftrl must dispatch through opaque-engine, not import a provider:\n"
        + "\n".join(
            f"  - {path.relative_to(PACKAGES_DIR.parent)}" for path in violations
        )
    )


def test_dpftrl_tests_do_not_import_provider_runtime() -> None:
    violations = [
        path for path in (DPFTRL_DIR / "tests").rglob("*.py") if _imports_provider(path)
    ]
    assert not violations, (
        "DP-FTRL execution tests belong in the wheel that owns their provider:\n"
        + "\n".join(
            f"  - {path.relative_to(PACKAGES_DIR.parent)}" for path in violations
        )
    )


def test_dpftrl_metadata_does_not_require_a_provider() -> None:
    with (DPFTRL_DIR / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]
    dependencies = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        dependencies.extend(extra)

    provider_dependencies = {
        _dependency_name(dependency)
        for dependency in dependencies
        if _dependency_name(dependency)
        in {"torch", "opaque-torch", "jax", "opaque-jax", "mlx", "opaque-mlx"}
    }
    assert not provider_dependencies, (
        "opaque-dpftrl must not require a provider: "
        + ", ".join(sorted(provider_dependencies))
    )
