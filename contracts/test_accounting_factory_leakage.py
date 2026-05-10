"""``opaque.accounting`` does not re-export per-stack accounting factories.

Cross-cutting accounting (algebra primitives, calibration, composition,
generic mechanisms ``identity`` / ``eps_delta`` / ``nonprivate``) lives at
``opaque.accounting``. Stack-specific factories (``gaussian``, ``adaclip``,
``poisson`` for DP-SGD; ``band_mf``, ``blt``, ``b_min_sep``, … for DP-FTRL)
live at ``opaque.api.accounting.{dpsgd,dpftrl}.*`` and are surfaced only on
``opaque.dpsgd.accounting`` / ``opaque.dpftrl.accounting``.

This test parses the ``__all__`` of every façade module under
``opaque/accounting/`` and the ``__all__`` of every impl module under
``opaque/api/accounting/{dpsgd,dpftrl}/`` and asserts the intersection is
empty.

Skipped when no factory __all__ has been declared yet (pre-phase-5).
"""

from __future__ import annotations

import ast
import pathlib

PACKAGES_DIR = pathlib.Path(__file__).resolve().parents[1] / "packages"

ACCOUNTING_FACADE_GLOBS = ("packages/opaque-accounting/src/opaque/accounting/**/*.py",)

# Per-stack factory roots — populated by phase 5.
PER_STACK_FACTORY_GLOBS = (
    "packages/opaque-dpsgd/src/opaque/api/accounting/dpsgd/**/*.py",
    "packages/opaque-dpftrl/src/opaque/api/accounting/dpftrl/**/*.py",
)


def _extract_all(path: pathlib.Path) -> set[str]:
    """Return the names listed in ``__all__`` for the module at ``path``.

    Returns the empty set if the module does not declare ``__all__`` or if
    the assigned value is something we cannot evaluate at parse time.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    for node in tree.body:
        target_names: list[str] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names = [node.target.id]
            value = node.value
        if "__all__" not in target_names or value is None:
            continue
        try:
            collected = ast.literal_eval(value)
        except ValueError:
            continue
        if isinstance(collected, (list, tuple, set)):
            return {str(n) for n in collected if isinstance(n, str)}
    return set()


def _glob_files(patterns: tuple[str, ...]) -> list[pathlib.Path]:
    repo_root = PACKAGES_DIR.parent
    files: list[pathlib.Path] = []
    for pat in patterns:
        files.extend(repo_root.glob(pat))
    return sorted({p.resolve() for p in files if p.is_file()})


def test_no_per_stack_factory_in_accounting_facade() -> None:
    facade_names: set[str] = set()
    for path in _glob_files(ACCOUNTING_FACADE_GLOBS):
        # Skip impl modules accidentally caught by the glob.
        if "/api/" in path.as_posix():
            continue
        facade_names |= _extract_all(path)

    factory_names: set[str] = set()
    for path in _glob_files(PER_STACK_FACTORY_GLOBS):
        factory_names |= _extract_all(path)

    overlap = facade_names & factory_names
    assert not overlap, (
        "opaque.accounting façade leaked per-stack factory names: "
        f"{sorted(overlap)}. Per-stack factories belong on "
        "opaque.dpsgd.accounting / opaque.dpftrl.accounting only."
    )
