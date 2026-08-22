"""``opaque-accounting`` source and tests do not import torch.

The wheel is a torch-free standalone in the post-refactor tree. Any
``import torch`` (or ``from torch …``) in the package's ``src`` or
``tests`` would silently strengthen the dependency cone via the test
runner's environment. Catch it at the source level.

This test scans only ``opaque-accounting``. Other wheels are torch-using by
design.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGES_DIR = pathlib.Path(__file__).resolve().parents[3] / "packages"

ACCOUNTING_DIR = PACKAGES_DIR / "opaque-accounting"


def _imports_torch(path: pathlib.Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" or alias.name.startswith("torch."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "torch" or (node.module or "").startswith("torch."):
                return True
    return False


def test_accounting_sources_are_torch_free() -> None:
    if not ACCOUNTING_DIR.exists():
        return
    violations: list[pathlib.Path] = []
    for path in ACCOUNTING_DIR.rglob("*.py"):
        # Only check source and tests; skip generated artefacts.
        rel = path.relative_to(ACCOUNTING_DIR)
        first = rel.parts[0] if rel.parts else ""
        if first not in {"src", "tests"}:
            continue
        if _imports_torch(path):
            violations.append(path)

    assert not violations, (
        "opaque-accounting must remain torch-free. "
        "These files import torch:\n"
        + "\n".join(f"  - {p.relative_to(PACKAGES_DIR.parent)}" for p in violations)
    )
