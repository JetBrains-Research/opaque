"""``opaque-base`` stays stdlib-only.

The serialization registry is the seam every other wheel builds on, so any
third-party import here — torch, numpy, optree, or another opaque wheel —
widens the dependency cone of the entire namespace. Sharing the ``opaque``
namespace is not the same as being allowed to import from it: only the modules
opaque-base itself ships are in bounds. Pin both the source imports and the
declared metadata.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import tomllib

BASE_DIR = pathlib.Path(__file__).resolve().parents[3] / "packages" / "opaque-base"


# The only ``opaque`` modules opaque-base ships, and therefore the only ones it
# may import. Reducing an intra-namespace import to its first component would
# exempt every other wheel's facade too — ``from opaque.pytree import ...``
# would read as "opaque" and pass — which is exactly the undeclared dependency
# this gate exists to catch.
_OWNED_ROOTS = ("opaque.api.base", "opaque.serialization")


def _imported_modules(path: pathlib.Path) -> set[str]:
    """Return each import as a module path, keeping ``opaque.*`` fully dotted."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            # ``from opaque.api import base`` names the submodule in the alias.
            modules = [node.module]
            if node.module == "opaque" or node.module.startswith("opaque."):
                modules = [f"{node.module}.{alias.name}" for alias in node.names]
        else:
            continue
        for module in modules:
            out.add(module if module.startswith("opaque") else module.split(".")[0])
    return out


def _is_owned(module: str) -> bool:
    return any(module == root or module.startswith(f"{root}.") for root in _OWNED_ROOTS)


def test_base_sources_import_only_the_stdlib() -> None:
    stdlib = set(sys.stdlib_module_names)
    violations: list[str] = []
    for path in (BASE_DIR / "src").rglob("*.py"):
        for name in sorted(_imported_modules(path)):
            rel = path.relative_to(BASE_DIR.parent.parent)
            if name.startswith("opaque"):
                if not _is_owned(name):
                    violations.append(
                        f"  - {rel}: import {name} (owned by another wheel)"
                    )
                continue
            if name not in stdlib:
                violations.append(f"  - {rel}: import {name}")
    assert not violations, "opaque-base must stay stdlib-only:\n" + "\n".join(
        violations
    )


def test_base_metadata_declares_no_dependencies() -> None:
    with (BASE_DIR / "pyproject.toml").open("rb") as f:
        project = tomllib.load(f)["project"]
    dependencies = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        dependencies.extend(extra)
    assert not dependencies, (
        "opaque-base must not declare runtime dependencies: "
        + ", ".join(sorted(dependencies))
    )
