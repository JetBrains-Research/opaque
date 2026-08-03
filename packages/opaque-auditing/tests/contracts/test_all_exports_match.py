"""Export-contract checks for the auditing attacks façades."""

from __future__ import annotations

import importlib
import types

import pytest

_MODULES = (
    "opaque.api.auditing.attacks",
    "opaque.auditing.attacks",
    "opaque.api.auditing",
    "opaque.auditing",
)


@pytest.mark.parametrize("module_name", _MODULES)
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
        f"{module_name}: public names not in __all__ (rule 6 violation): "
        f"{sorted(leaked)}"
    )


def test_attacks_facade_exports_match_impl() -> None:
    impl = importlib.import_module("opaque.api.auditing.attacks")
    facade = importlib.import_module("opaque.auditing.attacks")
    assert set(facade.__all__) == set(impl.__all__)
    assert "gradient_scores" in facade.__all__
