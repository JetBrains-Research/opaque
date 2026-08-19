"""``opaque-dpsgd`` must not depend on a provider runtime.

DP-SGD mechanisms dispatch native-array work through ``opaque-engine``. A
provider import or dependency here would prevent using the package with a
different provider alone.
"""

from __future__ import annotations

import ast
import pathlib
import re
import tomllib

PACKAGES_DIR = pathlib.Path(__file__).resolve().parents[3] / "packages"
DPSGD_DIR = PACKAGES_DIR / "opaque-dpsgd"
_PROVIDER_IMPORTS = (
    "torch",
    "jax",
    "mlx",
    "opaque.torch",
    "opaque.jax",
    "opaque.mlx",
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


def test_dpsgd_sources_do_not_import_provider_runtime() -> None:
    violations = [
        path for path in (DPSGD_DIR / "src").rglob("*.py") if _imports_provider(path)
    ]

    assert not violations, (
        "opaque-dpsgd must dispatch through opaque-engine, not import a provider:\n"
        + "\n".join(
            f"  - {path.relative_to(PACKAGES_DIR.parent)}" for path in violations
        )
    )


def test_dpsgd_tests_do_not_import_provider_runtime() -> None:
    violations = [
        path for path in (DPSGD_DIR / "tests").rglob("*.py") if _imports_provider(path)
    ]

    assert not violations, (
        "DP-SGD execution tests belong in the wheel that owns their provider:\n"
        + "\n".join(
            f"  - {path.relative_to(PACKAGES_DIR.parent)}" for path in violations
        )
    )


def test_dpsgd_metadata_does_not_require_a_provider() -> None:
    with (DPSGD_DIR / "pyproject.toml").open("rb") as f:
        project = tomllib.load(f)["project"]
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
        "opaque-dpsgd must not require a provider: "
        + ", ".join(sorted(provider_dependencies))
    )
