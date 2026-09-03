"""``opaque-optimizers`` must not depend on a provider runtime.

The wheel owns the backend-neutral optimizer rules and dispatches through
opaque-engine. A provider import or runtime dependency here would drag a
framework into every consumer that only wants the rules. Its *tests* may
execute on a backend — that requirement is declared, not inherited.
"""

from __future__ import annotations

import ast
import pathlib
import re
import tomllib

from tests._support.package_metadata import assert_portable_backend_test_matrix

PACKAGES_DIR = pathlib.Path(__file__).resolve().parents[3] / "packages"
OPTIMIZERS_DIR = PACKAGES_DIR / "opaque-optimizers"
_PROVIDERS = ("torch", "jax", "mlx")
# Both spellings of every provider: the framework itself, the public facade,
# and the implementation package behind it. Omitting the internal paths would
# let ``from opaque.api.torch import ...`` introduce exactly the coupling this
# gate exists to prevent.
_PROVIDER_IMPORTS = (
    *_PROVIDERS,
    *(f"opaque.{name}" for name in _PROVIDERS),
    *(f"opaque.api.{name}" for name in _PROVIDERS),
)
# Packages whose ``from <package> import <provider>`` form names the provider
# in the alias rather than the module.
_PROVIDER_PARENTS = ("opaque", "opaque.api")


def _imports_provider(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module in _PROVIDER_PARENTS and any(
                alias.name in _PROVIDERS for alias in node.names
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


def test_optimizers_sources_do_not_import_provider_runtime() -> None:
    violations = [
        path
        for path in (OPTIMIZERS_DIR / "src").rglob("*.py")
        if _imports_provider(path)
    ]

    assert not violations, (
        "opaque-optimizers must dispatch through opaque-engine, not import a provider:\n"
        + "\n".join(
            f"  - {path.relative_to(PACKAGES_DIR.parent)}" for path in violations
        )
    )


def test_optimizers_tests_declare_the_backend_they_execute_on() -> None:
    """Tests may use a backend; the package must say which one.

    The rules are neutral, but their tests compute real numbers and so need a
    backend to run on. ARC-006 allows that — a wheel-local test may rely on
    "its backend requirements" — provided the requirement is declared rather
    than inherited from whatever the monorepo developer environment happens to
    install. Relocating the suite to the provider's wheel is the wrong fix: it
    separates the tests from the behavior they describe.
    """
    tests_dir = OPTIMIZERS_DIR / "tests"
    if not any(tests_dir.rglob("test_*.py")):
        return

    with (OPTIMIZERS_DIR / "pyproject.toml").open("rb") as f:
        groups = tomllib.load(f).get("dependency-groups", {})
    test_dependencies = groups.get("test", [])
    assert_portable_backend_test_matrix(test_dependencies)
    assert "torchopt" in {
        _dependency_name(dependency) for dependency in test_dependencies
    }


def test_optimizers_metadata_does_not_require_a_provider() -> None:
    with (OPTIMIZERS_DIR / "pyproject.toml").open("rb") as f:
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
        "opaque-optimizers must not require a provider: "
        + ", ".join(sorted(provider_dependencies))
    )
