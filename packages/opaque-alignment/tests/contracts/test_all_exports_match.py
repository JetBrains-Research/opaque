"""AGENTS.md rule 6 enforcement for the alignment namespaces.

For every package (``__init__.py``) and every public module (a ``.py`` file
whose name does not start with ``_``) under ``opaque.alignment.*`` and
``opaque.api.alignment.*``:

- it declares an explicit ``__all__``;
- every name in ``__all__`` resolves as an attribute; and
- no public, non-submodule attribute leaks outside ``__all__``.

Submodule attributes (bound as a side effect of importing a subpackage) and
private (underscore-prefixed) impl modules are exempt — rule 6 governs
``__init__.py`` files and the public façade surface, not private helpers.
"""

from __future__ import annotations

import importlib
import pkgutil
import types

import pytest

import opaque.alignment
import opaque.api.alignment

_ROOTS = (opaque.alignment, opaque.api.alignment)


def _iter_checked_modules(root: types.ModuleType):
    """Yield the root and every sub-package / public module beneath it."""
    yield root.__name__
    if not hasattr(root, "__path__"):
        return
    for info in pkgutil.walk_packages(root.__path__, root.__name__ + "."):
        leaf = info.name.rsplit(".", 1)[-1]
        if leaf.startswith("_"):
            # Private impl modules are not part of the public-API contract.
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

    leaked = {
        name
        for name in dir(mod)
        if not name.startswith("_")
        and name not in declared
        and not isinstance(getattr(mod, name), types.ModuleType)
    }
    assert not leaked, (
        f"{module_name}: public names not in __all__ (rule 6 violation): "
        f"{sorted(leaked)}"
    )
