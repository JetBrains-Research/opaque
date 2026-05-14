"""User-facing façade docstrings do not leak the internal ``opaque.api.*`` namespace.

The ``opaque.api.*`` paths are the internal contributor surface — not
documented in the user-facing docs at all. Any docstring on a public
façade module that mentions ``opaque.api.`` (in any sphinx role, in
prose, or in code blocks) is a documentation leak: a reader of
``help(opaque.X)`` shouldn't be told to look at ``opaque.api.X`` to
understand the surface they just imported.

The check uses the same ``FACADE_GLOBS_BY_WHEEL`` mapping as
``test_facade_discipline.py`` so the two stay in sync — when a wheel
adds a new façade module, both checks see it automatically.

Allowed nodes are limited to the docstring content; module body code
naturally needs ``from opaque.api.<X> import …`` lines to wire the
re-exports, and those are excluded.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGES_DIR = pathlib.Path(__file__).resolve().parents[2] / "packages"

# Source-of-truth set of façade files. Kept in sync with
# ``test_facade_discipline.FACADE_GLOBS_BY_WHEEL``; see that file for
# the rationale of why each wheel's entry is on the list.
FACADE_GLOBS_BY_WHEEL: dict[str, tuple[str, ...]] = {
    "opaque-base": ("opaque/serialization/__init__.py",),
    "opaque-engine": (
        "opaque/types.py",
        "opaque/pytree.py",
        "opaque/random/__init__.py",
        "opaque/random/types.py",
        "opaque/distributed/__init__.py",
        "opaque/distributed/collectives.py",
        "opaque/distributed/gradients.py",
        "opaque/functional/__init__.py",
        "opaque/scheduling/__init__.py",
        "opaque/scheduling/types.py",
        "opaque/profiling/__init__.py",
        "opaque/profiling/types.py",
    ),
    "opaque-optimizers": (
        "opaque/optimizers/__init__.py",
        "opaque/optimizers/types.py",
    ),
    "opaque-accounting": ("opaque/accounting/__init__.py",),
    "opaque-dpsgd": (
        "opaque/dpsgd/__init__.py",
        "opaque/dpsgd/clipping/__init__.py",
        "opaque/dpsgd/clipping/types.py",
        "opaque/dpsgd/clipping/fun.py",
        "opaque/dpsgd/noise/__init__.py",
        "opaque/dpsgd/noise/types.py",
        "opaque/dpsgd/sampling/__init__.py",
        "opaque/dpsgd/accounting/__init__.py",
        "opaque/dpsgd/accounting/mechanisms/__init__.py",
        "opaque/dpsgd/accounting/amplification/__init__.py",
    ),
    "opaque-dpftrl": (
        "opaque/dpftrl/__init__.py",
        "opaque/dpftrl/clipping/__init__.py",
        "opaque/dpftrl/clipping/types.py",
        "opaque/dpftrl/clipping/fun.py",
        "opaque/dpftrl/noise/__init__.py",
        "opaque/dpftrl/noise/types.py",
        "opaque/dpftrl/sampling/__init__.py",
        "opaque/dpftrl/accounting/__init__.py",
        "opaque/dpftrl/accounting/mechanisms/__init__.py",
        "opaque/dpftrl/accounting/amplification/__init__.py",
    ),
    "opaque-auditing": (
        "opaque/auditing/__init__.py",
        "opaque/auditing/types.py",
        "opaque/auditing/attacks/__init__.py",
        "opaque/auditing/one_run/__init__.py",
    ),
    "opaque-patches": (
        "opaque/patches/__init__.py",
        "opaque/patches/torch/__init__.py",
        "opaque/patches/kernels/__init__.py",
        "opaque/patches/peft/__init__.py",
        "opaque/patches/transformers/__init__.py",
        "opaque/patches/transformers/runtime/__init__.py",
    ),
    "opaque-transformers": ("opaque/transformers/__init__.py",),
}


def _collect_facade_files(wheel: str, globs: tuple[str, ...]) -> list[pathlib.Path]:
    src_dir = PACKAGES_DIR / wheel / "src"
    if not src_dir.exists():
        return []
    files: list[pathlib.Path] = []
    for pattern in globs:
        files.extend(src_dir.glob(pattern))
    return sorted({p.resolve() for p in files if p.is_file()})


def _module_and_function_docstrings(tree: ast.Module) -> list[tuple[int, str]]:
    """Return ``(lineno, docstring)`` pairs for the module + every
    top-level function / class definition's docstring."""
    out: list[tuple[int, str]] = []
    module_doc = ast.get_docstring(tree)
    if module_doc:
        out.append((1, module_doc))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc:
                out.append((node.lineno, doc))
    return out


def test_facade_docstrings_do_not_leak_internal_namespace() -> None:
    violations: list[str] = []
    for wheel, globs in FACADE_GLOBS_BY_WHEEL.items():
        for path in _collect_facade_files(wheel, globs):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as e:
                violations.append(f"{path}: SyntaxError: {e}")
                continue
            for lineno, docstring in _module_and_function_docstrings(tree):
                if "opaque.api." in docstring:
                    rel = path.relative_to(PACKAGES_DIR.parent)
                    violations.append(
                        f"{rel}:{lineno}: docstring leaks ``opaque.api.*`` "
                        f"to a user-facing surface (this is the public "
                        f"façade — the internal namespace must not appear "
                        f"in user-facing docstrings)."
                    )

    assert not violations, "Façade docstring leaks:\n" + "\n".join(
        f"  - {v}" for v in violations
    )
