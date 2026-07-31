"""MRO resolution, inert declarations, and the unknown-leaf raise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from opaque.serialization import (
    from_state_dict,
    is_inert,
    register_inert_type,
    register_serializer,
    resolve_serializer,
    state_dict,
)


@dataclass
class _Leaf:
    """Registered leaf carrying one value."""

    value: int


class _LeafSubclass(_Leaf):
    """Subclass with no declaration of its own."""


class _InertLeaf(_Leaf):
    """Subclass declared inert even though its base is registered."""


class _Vendor:
    """Opaque non-container: neither registered nor structural."""


class _VendorSpec:
    """Opaque non-container declared inert."""

    def __init__(self, layout: str) -> None:
        self.layout = layout


def _leaf_save(obj: _Leaf) -> dict[str, Any]:
    return {"": obj.value}


def _leaf_load(template: _Leaf, sd: dict[str, Any]) -> _Leaf:
    return type(template)(value=sd.get("", template.value))


register_serializer(_Leaf, _leaf_save, _leaf_load)
register_inert_type(_InertLeaf)
register_inert_type(_VendorSpec)


@dataclass
class _Holder:
    leaf: Any


def test_subclass_resolves_to_registered_base_handler() -> None:
    flat = state_dict(_Holder(leaf=_LeafSubclass(value=7)))
    assert flat == {"leaf": 7}

    restored = from_state_dict(_Holder(leaf=_LeafSubclass(value=0)), flat)
    assert restored.leaf == _LeafSubclass(value=7)
    assert type(restored.leaf) is _LeafSubclass


def test_nearest_declaration_wins_over_registered_base() -> None:
    assert resolve_serializer(_LeafSubclass) is not None
    assert resolve_serializer(_InertLeaf) is None
    assert is_inert(_InertLeaf)

    inert = _InertLeaf(value=3)
    assert state_dict(_Holder(leaf=inert)) == {}
    assert from_state_dict(_Holder(leaf=inert), {"leaf": 99}).leaf is inert


def test_inert_leaf_is_omitted_and_restored_from_template() -> None:
    spec = _VendorSpec(layout="abc")
    assert state_dict({"spec": spec, "n": 2}) == {"n": 2}

    restored = from_state_dict({"spec": spec, "n": 0}, {"n": 2})
    assert restored["spec"] is spec
    assert restored["n"] == 2


def test_unregistered_leaf_raises_on_save_with_path() -> None:
    with pytest.raises(TypeError, match=r"state_dict\(\) cannot handle _Vendor"):
        state_dict({"outer": {"inner": _Vendor()}})

    with pytest.raises(TypeError, match=r"outer.inner"):
        state_dict({"outer": {"inner": _Vendor()}})


def test_unregistered_leaf_raises_on_load() -> None:
    with pytest.raises(TypeError, match=r"from_state_dict\(\) cannot handle _Vendor"):
        from_state_dict(_Holder(leaf=_Vendor()), {})


def test_unregistered_root_leaf_raises_without_path() -> None:
    with pytest.raises(TypeError) as excinfo:
        state_dict(_Vendor())
    assert " at " not in str(excinfo.value)


def test_error_names_both_registration_routes() -> None:
    with pytest.raises(TypeError) as excinfo:
        state_dict(_Vendor())
    message = str(excinfo.value)
    assert "register_serializer()" in message
    assert "register_inert_type()" in message


def test_inert_declaration_conflicts_with_registered_serializer() -> None:
    with pytest.raises(ValueError, match="already has a registered serializer"):
        register_inert_type(_Leaf)


def test_late_registration_invalidates_resolution_cache() -> None:
    class _Late:
        pass

    with pytest.raises(TypeError):
        state_dict({"late": _Late()})

    register_serializer(_Late, lambda _o: {"": "late"}, lambda t, _sd: t)
    try:
        assert state_dict({"late": _Late()}) == {"late": "late"}
    finally:
        from opaque.api.base.serialization import _registry

        del _registry._REGISTRY[_Late]
        _registry._RESOLVED.clear()


def test_registering_a_serializer_clears_an_inert_declaration() -> None:
    class _Flip:
        pass

    register_inert_type(_Flip)
    assert is_inert(_Flip)

    register_serializer(_Flip, lambda _o: {"": 1}, lambda t, _sd: t)
    try:
        assert not is_inert(_Flip)
        assert state_dict({"flip": _Flip()}) == {"flip": 1}
    finally:
        from opaque.api.base.serialization import _registry

        del _registry._REGISTRY[_Flip]
        _registry._RESOLVED.clear()
