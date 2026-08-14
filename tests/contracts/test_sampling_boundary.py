"""Sampling contracts remain backend-neutral and consistently exported."""

from __future__ import annotations

import ast
import pathlib

import opaque.api.engine.sampling as implementation
import opaque.sampling as facade

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLING_ROOTS = (
    REPO_ROOT
    / "packages"
    / "opaque-dpsgd"
    / "src"
    / "opaque"
    / "api"
    / "dpsgd"
    / "sampling",
    REPO_ROOT
    / "packages"
    / "opaque-dpftrl"
    / "src"
    / "opaque"
    / "api"
    / "dpftrl"
    / "sampling",
)


def _imports_torch_sampler(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "torch.utils.data"
        and any(alias.name == "Sampler" for alias in node.names)
        for node in ast.walk(tree)
    )


def test_algorithm_sampling_modules_do_not_import_torch_sampler() -> None:
    violations = [
        str(path.relative_to(REPO_ROOT))
        for root in SAMPLING_ROOTS
        for path in root.glob("*.py")
        if _imports_torch_sampler(path)
    ]

    assert not violations, "Torch Sampler imports found:\n" + "\n".join(violations)


def test_sampling_facade_matches_implementation_exports() -> None:
    assert facade.__all__ == implementation.__all__
    for name in implementation.__all__:
        assert getattr(facade, name) is getattr(implementation, name)
