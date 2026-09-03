"""Provider wheels must not pull functional owners back down the graph."""

from __future__ import annotations

import ast
import pathlib
import re
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PACKAGES_DIR = REPO_ROOT / "packages"
PROVIDERS = ("opaque-torch", "opaque-mlx")
FUNCTIONAL_PACKAGES = {
    "opaque-dpsgd",
    "opaque-dpftrl",
    "opaque-auditing",
    "opaque-optimizers",
}
FUNCTIONAL_IMPORTS = tuple(
    f"opaque.{name.removeprefix('opaque-')}" for name in FUNCTIONAL_PACKAGES
) + tuple(f"opaque.api.{name.removeprefix('opaque-')}" for name in FUNCTIONAL_PACKAGES)


def _dependency_name(spec: str) -> str:
    return re.split(r"[<>=!~;\s]", spec, maxsplit=1)[0].split("[", 1)[0].lower()


def _marker(spec: str) -> str:
    _, separator, marker = spec.partition(";")
    return marker.replace('"', "'").replace(" ", "") if separator else ""


def test_umbrella_mlx_extra_is_limited_to_apple_silicon() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]

    assert len(extras["mlx"]) == 1
    assert _dependency_name(extras["mlx"][0]) == "opaque-mlx"
    assert _marker(extras["mlx"][0]) == (
        "platform_system=='Darwin'andplatform_machine=='arm64'"
    )


def _imports_functional_owner(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functional_modules = {
        name.rsplit(".", maxsplit=1)[-1] for name in FUNCTIONAL_IMPORTS
    }
    if any(
        isinstance(node, ast.ImportFrom)
        and node.module in {"opaque", "opaque.api"}
        and any(alias.name in functional_modules for alias in node.names)
        for node in ast.walk(tree)
    ):
        return True
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    return any(
        imported == forbidden or imported.startswith(f"{forbidden}.")
        for imported in imports
        for forbidden in FUNCTIONAL_IMPORTS
    )


def test_provider_metadata_has_no_reverse_functional_edges() -> None:
    violations: list[str] = []
    for provider in PROVIDERS:
        with (PACKAGES_DIR / provider / "pyproject.toml").open("rb") as f:
            metadata = tomllib.load(f)
        project = metadata["project"]
        dependencies = list(project.get("dependencies", []))
        for extra in project.get("optional-dependencies", {}).values():
            dependencies.extend(extra)
        for group in metadata.get("dependency-groups", {}).values():
            dependencies.extend(group)
        dependencies.extend(metadata.get("tool", {}).get("uv", {}).get("sources", {}))

        reverse = sorted(
            {
                _dependency_name(spec)
                for spec in dependencies
                if _dependency_name(spec) in FUNCTIONAL_PACKAGES
            }
        )
        violations.extend(f"{provider}: {name}" for name in reverse)

    assert not violations, "Provider reverse dependency edges:\n" + "\n".join(
        violations
    )


def test_provider_tests_do_not_import_functional_owners() -> None:
    violations = [
        path.relative_to(PACKAGES_DIR.parent)
        for provider in PROVIDERS
        for path in (PACKAGES_DIR / provider / "tests").rglob("*.py")
        if _imports_functional_owner(path)
    ]

    assert not violations, "Provider test imports functional owners:\n" + "\n".join(
        f"  - {path}" for path in violations
    )
