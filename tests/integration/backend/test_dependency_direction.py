"""Lower-layer wheels do not import from upper-layer wheels.

Specifically:

- ``opaque-accounting`` is torch-free and depends only on
  ``opaque-base`` (post-refactor) — its source must not import from
  ``opaque.api.engine``, ``opaque.api.optimizers``, ``opaque.api.dpsgd``,
  or ``opaque.api.dpftrl``.
- ``opaque-base`` (when introduced) must not import from any
  ``opaque.api.<other>``.
- ``opaque-engine`` (when introduced) must not import from any of
  ``opaque.api.{optimizers,accounting,dpsgd,dpftrl,auditing,patches,
  transformers}``.

The check operates over the file system, not import graphs, so it is
phase-aware: when a wheel does not yet exist it has nothing to scan.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGES_DIR = pathlib.Path(__file__).resolve().parents[3] / "packages"

# wheel_dir → forbidden-import prefixes
FORBIDDEN_IMPORTS: dict[str, tuple[str, ...]] = {
    "opaque-base": (
        "opaque.api.engine",
        "opaque.api.torch",
        "opaque.api.jax",
        "opaque.api.mlx",
        "opaque.api.optimizers",
        "opaque.api.accounting",
        "opaque.api.dpsgd",
        "opaque.api.dpftrl",
        "opaque.api.auditing",
        "opaque.api.patches",
        "opaque.api.transformers",
    ),
    "opaque-accounting": (
        "opaque.api.engine",
        "opaque.api.torch",
        "opaque.api.jax",
        "opaque.api.mlx",
        "opaque.api.optimizers",
        "opaque.api.dpsgd",
        "opaque.api.dpftrl",
        "opaque.api.auditing",
        "opaque.api.patches",
        "opaque.api.transformers",
    ),
    "opaque-engine": (
        "opaque.api.torch",
        "opaque.api.jax",
        "opaque.api.mlx",
        "opaque.api.optimizers",
        "opaque.api.accounting",
        "opaque.api.dpsgd",
        "opaque.api.dpftrl",
        "opaque.api.auditing",
        "opaque.api.patches",
        "opaque.api.transformers",
    ),
    "opaque-torch": (
        "opaque.api.jax",
        "opaque.api.mlx",
        "opaque.api.optimizers",
        "opaque.api.accounting",
        "opaque.api.dpsgd",
        "opaque.api.dpftrl",
        "opaque.api.auditing",
        "opaque.api.patches",
        "opaque.api.transformers",
        "opaque.api.alignment",
    ),
    "opaque-optimizers": (
        "opaque.api.accounting",
        "opaque.api.dpsgd",
        "opaque.api.dpftrl",
        "opaque.api.auditing",
        "opaque.api.patches",
        "opaque.api.transformers",
    ),
    # opaque-alignment is mechanism-agnostic (plan §1.1): it builds only on
    # the substrate wheels (opaque-engine, opaque-base). It must never import
    # a mechanism, optimizer, or trainer wheel. The alignment-specific kernels
    # live inside this wheel, so opaque.api.patches is forbidden in source too
    # (the optional ``patches`` extra is used by examples, not by src).
    "opaque-alignment": (
        "opaque.api.optimizers",
        "opaque.api.accounting",
        "opaque.api.dpsgd",
        "opaque.api.dpftrl",
        "opaque.api.auditing",
        "opaque.api.patches",
        "opaque.api.transformers",
    ),
}


def _imports(path: pathlib.Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_dependency_direction() -> None:
    violations: list[str] = []
    for wheel, forbidden in FORBIDDEN_IMPORTS.items():
        src = PACKAGES_DIR / wheel / "src"
        if not src.exists():
            continue
        for path in src.rglob("*.py"):
            mods = _imports(path)
            for forbid in forbidden:
                bad = sorted(
                    m for m in mods if m == forbid or m.startswith(forbid + ".")
                )
                if bad:
                    rel = path.relative_to(PACKAGES_DIR.parent)
                    violations.extend(
                        f"{rel}: imports {forbidden} (forbidden)" for forbidden in bad
                    )

    assert not violations, "Dependency direction violations:\n" + "\n".join(
        f"  - {v}" for v in violations
    )
