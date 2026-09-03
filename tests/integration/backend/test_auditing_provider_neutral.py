"""``opaque-auditing`` must not depend on a provider runtime.

The one-run estimator core is plain numpy/scipy, and the attack scorers
dispatch native-array work through ``opaque-engine``. A provider import or
dependency here would prevent auditing models on a different provider.
"""

from __future__ import annotations

import ast
import pathlib
import re
import tomllib

from tests._support.package_metadata import assert_portable_backend_test_matrix

PACKAGES_DIR = pathlib.Path(__file__).resolve().parents[3] / "packages"
AUDITING_DIR = PACKAGES_DIR / "opaque-auditing"
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


def test_auditing_sources_do_not_import_provider_runtime() -> None:
    violations = [
        path for path in (AUDITING_DIR / "src").rglob("*.py") if _imports_provider(path)
    ]

    assert not violations, (
        "opaque-auditing must dispatch through opaque-engine, not import a provider:\n"
        + "\n".join(
            f"  - {path.relative_to(PACKAGES_DIR.parent)}" for path in violations
        )
    )


def test_auditing_tests_declare_any_backend_they_execute_on() -> None:
    """Auditing's tests may use a backend; the package must then say so.

    Most of this suite is numpy-only and needs no provider. The collation
    helper is the exception -- it stacks native arrays, so covering it means
    running one. ARC-006 permits that for a wheel-local test, on the condition
    that the requirement is *declared* rather than inherited from whatever the
    monorepo developer environment happens to install. Relocating such a test
    would be the wrong fix: it separates the test from the behavior it
    describes. So this gate follows actual usage — silent while the suite stays
    provider-free, and demanding a declaration the moment it isn't.
    """
    users = sorted(
        path
        for path in (AUDITING_DIR / "tests").rglob("*.py")
        if _imports_provider(path)
    )
    if not users:
        return

    with (AUDITING_DIR / "pyproject.toml").open("rb") as f:
        groups = tomllib.load(f).get("dependency-groups", {})
    assert_portable_backend_test_matrix(groups.get("test", []))


def test_auditing_metadata_does_not_require_a_provider() -> None:
    with (AUDITING_DIR / "pyproject.toml").open("rb") as f:
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
        "opaque-auditing must not require a provider: "
        + ", ".join(sorted(provider_dependencies))
    )
