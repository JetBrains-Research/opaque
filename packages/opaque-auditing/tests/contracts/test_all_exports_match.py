"""Export-contract checks for the auditing façades."""

from __future__ import annotations

import importlib
import pkgutil
import types

import pytest

import opaque.api.auditing
import opaque.auditing

_ROOTS = (opaque.auditing, opaque.api.auditing)

# Names a façade may declare that its impl counterpart does not: the wheel
# version line lives on the public façade only.
_FACADE_ONLY = frozenset({"__version__"})


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


def _suffixes(root: types.ModuleType) -> set[str]:
    """Public module paths beneath ``root``, relative to it (``""`` is root)."""
    prefix = root.__name__
    return {name[len(prefix) :].lstrip(".") for name in _iter_checked_modules(root)}


def _paired_suffixes() -> list[str]:
    return sorted(_suffixes(opaque.auditing) | _suffixes(opaque.api.auditing))


@pytest.mark.parametrize("suffix", _paired_suffixes())
def test_facade_and_impl_exports_agree(suffix: str) -> None:
    """Every public façade module mirrors its impl counterpart's ``__all__``.

    Checked in both directions, so an impl module that never grew a façade
    fails just as loudly as a façade that drifted behind its impl. Guards the
    divergence behind #375, where ``gradient_scores`` reached
    ``opaque.api.auditing.attacks`` but not the ``opaque.auditing.attacks``
    façade the reference docs point users at.
    """
    dotted = f".{suffix}" if suffix else ""
    facade_name = f"{opaque.auditing.__name__}{dotted}"
    impl_name = f"{opaque.api.auditing.__name__}{dotted}"

    try:
        facade = importlib.import_module(facade_name)
    except ModuleNotFoundError:
        pytest.fail(f"{impl_name} has no {facade_name} façade")
    try:
        impl = importlib.import_module(impl_name)
    except ModuleNotFoundError:
        pytest.fail(f"{facade_name} façade has no {impl_name} counterpart")

    facade_all = set(facade.__all__) - _FACADE_ONLY
    impl_all = set(impl.__all__) - _FACADE_ONLY

    assert facade_all == impl_all, (
        f"{facade_name} and {impl_name} disagree on __all__: "
        f"façade-only={sorted(facade_all - impl_all)}, "
        f"impl-only={sorted(impl_all - facade_all)}"
    )
