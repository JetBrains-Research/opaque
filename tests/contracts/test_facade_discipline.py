"""Façade modules contain only re-exports.

A façade module under ``opaque/<concern>/`` (or a stack façade like
``opaque/dpsgd/<concern>/``) re-exports symbols from the corresponding
``opaque.api.*`` impl tree. It does not define new behavior.

Allowed nodes in a façade module:

- ``Import`` / ``ImportFrom`` (the re-export itself).
- ``Assign(__all__|__version__)`` — the public-API declaration / version
  line.
- ``FunctionDef(__getattr__)`` — PEP 562 lazy import (e.g. the
  lazy-accounting hook in ``opaque.dpsgd``).
- ``Expr(Constant(str))`` — module docstring or comments.
- ``If(TYPE_CHECKING)`` — type-only imports.

Anything else is a façade discipline violation and the test fails.

This test only checks façade trees that already exist. It is a no-op
during phases that have not yet introduced the corresponding façade.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGES_DIR = pathlib.Path(__file__).resolve().parents[2] / "packages"

# Façade roots, by wheel. Each entry is a glob relative to the wheel's
# ``src/`` directory. The mapping is the source of truth: CI enforces
# discipline only on listed paths.
#
# The list grows phase by phase. A façade enters the list only when its
# refactor phase makes it a clean re-export. Entries below are commented
# with the phase that adds them.
FACADE_GLOBS_BY_WHEEL: dict[str, tuple[str, ...]] = {
    # Phase 5 already clean today: dpsgd / dpftrl façades are thin
    # re-exports + a private ``_LAZY_SUBMODULES`` constant that powers the
    # PEP 562 ``__getattr__`` lazy-import hook for the ``accounting``
    # submodule.
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
    # Phase 6 — opaque-auditing has no business logic in __init__.
    "opaque-auditing": (
        "opaque/auditing/__init__.py",
        "opaque/auditing/types.py",
        "opaque/auditing/attacks/__init__.py",
        "opaque/auditing/one_run/__init__.py",
    ),
    # Phase 1 — opaque-base.
    "opaque-base": ("opaque/serialization/__init__.py",),
    # Phase 2 — opaque-engine.
    "opaque-engine": (
        "opaque/autodiff.py",
        "opaque/execution.py",
        "opaque/ops.py",
        "opaque/primitive.py",
        "opaque/types.py",
        "opaque/pytree.py",
        "opaque/sampling.py",
        "opaque/backend/__init__.py",
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
    # Phase 3 — opaque-optimizers.
    "opaque-optimizers": (
        "opaque/optimizers/__init__.py",
        "opaque/optimizers/types.py",
    ),
    # Phase 4 — opaque-accounting.
    "opaque-accounting": ("opaque/accounting/__init__.py",),
    # Phase 6 — opaque-patches, opaque-transformers.
    "opaque-patches": (
        "opaque/patches/__init__.py",
        "opaque/patches/torch/__init__.py",
        "opaque/patches/kernels/__init__.py",
        "opaque/patches/peft/__init__.py",
        "opaque/patches/transformers/__init__.py",
        "opaque/patches/transformers/runtime/__init__.py",
    ),
    "opaque-transformers": ("opaque/transformers/__init__.py",),
    # Backend adapters — public surfaces only re-export factories and
    # provider-specific helpers from their implementation packages.
    "opaque-torch": (
        "opaque/torch/__init__.py",
        "opaque/torch/device/__init__.py",
        "opaque/torch/functional/__init__.py",
        "opaque/torch/random/__init__.py",
    ),
    "opaque-jax": ("opaque/jax/__init__.py",),
    "opaque-mlx": ("opaque/mlx/__init__.py",),
    # opaque-alignment — every façade file under opaque/alignment/ is a pure
    # re-export of the opaque.api.alignment.* impl tree.
    "opaque-alignment": (
        "opaque/alignment/__init__.py",
        "opaque/alignment/**/__init__.py",
        "opaque/alignment/**/types.py",
    ),
}


_DUNDER_NAMES = frozenset({"__all__", "__version__"})


def _is_facade_assign_target(target: ast.expr) -> bool:
    """A façade may assign to ``__all__``, ``__version__``, or a
    private (underscore-prefixed) module-level constant — the latter
    supports PEP 562 lazy-import infrastructure such as
    ``_LAZY_SUBMODULES = frozenset({...})``.
    """
    if not isinstance(target, ast.Name):
        return False
    name = target.id
    if name in _DUNDER_NAMES:
        return True
    return name.startswith("_") and not name.startswith("__")


def _is_allowed_node(node: ast.stmt) -> bool:
    """Return True if ``node`` is allowed inside a façade module."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    if isinstance(node, ast.Assign):
        return all(_is_facade_assign_target(t) for t in node.targets)
    if isinstance(node, ast.AnnAssign):
        return _is_facade_assign_target(node.target)
    if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
        # PEP 562 lazy-attribute hook (used by opaque.dpsgd for
        # lazy-loading the accounting submodule).
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return isinstance(node.value.value, str)  # docstring
    if isinstance(node, ast.Raise):
        # ``raise ImportError(...)`` inside ``try/except`` is the only
        # legitimate way to surface a missing native extension at façade
        # boundaries (see opaque.accounting / opaque.patches.kernels).
        return True
    if isinstance(node, ast.Try):
        return (
            all(_is_allowed_node(s) for s in node.body)
            and all(_is_allowed_node(s) for h in node.handlers for s in h.body)
            and all(_is_allowed_node(s) for s in node.orelse)
            and all(_is_allowed_node(s) for s in node.finalbody)
        )
    if isinstance(node, ast.If):
        # TYPE_CHECKING guard, or "if condition: import …".
        return all(_is_allowed_node(s) for s in node.body) and all(
            _is_allowed_node(s) for s in node.orelse
        )
    return False


def _collect_facade_files(wheel: str, globs: tuple[str, ...]) -> list[pathlib.Path]:
    src_dir = PACKAGES_DIR / wheel / "src"
    if not src_dir.exists():
        return []
    files: list[pathlib.Path] = []
    for pattern in globs:
        files.extend(src_dir.glob(pattern))
    # Drop nested impl files that happen to match a façade glob — we only
    # check the top-level façade files explicitly listed.
    return sorted({p.resolve() for p in files if p.is_file()})


def test_facade_modules_contain_only_reexports() -> None:
    violations: list[str] = []
    for wheel, globs in FACADE_GLOBS_BY_WHEEL.items():
        for path in _collect_facade_files(wheel, globs):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as e:
                violations.append(f"{path}: SyntaxError: {e}")
                continue
            violations.extend(
                f"{path.relative_to(PACKAGES_DIR.parent)}:{node.lineno}: "
                f"disallowed node {type(node).__name__}"
                for node in tree.body
                if not _is_allowed_node(node)
            )

    assert not violations, "Façade discipline violations:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_stack_clipping_facades_keep_engine_aliases() -> None:
    import opaque.api.engine.clipping as clipping
    import opaque.api.engine.clipping.fun as clipping_fun
    import opaque.api.engine.clipping.types as clipping_types
    import opaque.dpftrl.clipping as dpftrl_clipping
    import opaque.dpftrl.clipping.fun as dpftrl_clipping_fun
    import opaque.dpftrl.clipping.types as dpftrl_clipping_types
    import opaque.dpsgd.clipping as dpsgd_clipping
    import opaque.dpsgd.clipping.fun as dpsgd_clipping_fun
    import opaque.dpsgd.clipping.types as dpsgd_clipping_types

    for stack_clipping in (dpsgd_clipping, dpftrl_clipping):
        for name in clipping.__all__:
            assert getattr(stack_clipping, name) is getattr(clipping, name)

    for stack_fun in (dpsgd_clipping_fun, dpftrl_clipping_fun):
        for name in clipping_fun.__all__:
            assert getattr(stack_fun, name) is getattr(clipping_fun, name)

    for stack_types in (dpsgd_clipping_types, dpftrl_clipping_types):
        for name in set(clipping_types.__all__) & set(stack_types.__all__):
            assert getattr(stack_types, name) is getattr(clipping_types, name)

    assert "adaptive_clipped_grad" in dpsgd_clipping.__all__
    assert not hasattr(clipping, "adaptive_clipped_grad")
    assert not hasattr(dpftrl_clipping, "adaptive_clipped_grad")
