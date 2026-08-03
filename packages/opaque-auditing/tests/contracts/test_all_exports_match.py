"""Export-contract checks for the auditing attacks façades."""

from __future__ import annotations

import importlib
import pkgutil
import types

import pytest

import opaque.api.auditing
import opaque.auditing

_ROOTS = (opaque.auditing, opaque.api.auditing)


def _iter_checked_modules(root: types.ModuleType):
    """Yield the root and every sub-package / public module beneath it."""
    yield root.__name__
    if not hasattr(root, "__path__"):
        return
    for info in pkgutil.walk_packages(root.__path__, root.__name__ + "."):
        leaf = info.name.rsplit(".", 1)[-1]
        if leaf.startswith("_"):
            continue
        yield info.name


def _module_names() -> list[str]:
    names: list[str] = []
    for root in _ROOTS:
        names.extend(_iter_checked_modules(root))
    return sorted(set(names))


@pytest.mark.parametrize("module_name", _module_names())
def test_module_exports_match(module_name: str) -> None:
    mod = importlib.import_module(module_name)

    assert hasattr(mod, "__all__"), f"{module_name} is missing an explicit __all__"
    declared = set(mod.__all__)

    missing = {name for name in declared if not hasattr(mod, name)}
    assert not missing, (
        f"{module_name}: in __all__ but not importable: {sorted(missing)}"
    )

    is_package = hasattr(mod, "__path__")
    is_facade = module_name.startswith("opaque.auditing")
    if not (is_package or is_facade):
        return

    def _is_package_surface(value: object) -> bool:
        origin = getattr(value, "__module__", "") or ""
        return origin == module_name or origin.startswith(
            ("opaque.auditing", "opaque.api.auditing")
        )

    leaked = {
        name
        for name in dir(mod)
        if not name.startswith("_")
        and name not in declared
        and not isinstance(getattr(mod, name), types.ModuleType)
        and _is_package_surface(getattr(mod, name))
    }
    assert not leaked, (
        f"{module_name}: public names not in __all__ "
        f"(explicit-__all__ discipline violation): "
        f"{sorted(leaked)}"
    )
