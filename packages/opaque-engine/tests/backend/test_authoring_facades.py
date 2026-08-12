"""Public authoring façades expose the canonical dispatch objects."""

from __future__ import annotations

import opaque.autodiff as autodiff
import opaque.ops as ops
import opaque.primitive as primitive
from opaque.api.engine import autodiff as api_autodiff
from opaque.api.engine import ops as api_ops
from opaque.api.engine import primitive as api_primitive


def test_primitive_facade_has_an_explicit_complete_export_contract() -> None:
    assert set(primitive.__all__) == {
        "CORE_PROFILE_VERSION",
        "CORE_PRIMITIVES",
        "CoreProfile",
        "DuplicatePrimitiveRegistrationError",
        "IncompleteBackendError",
        "InvalidPrimitiveRegistrationError",
        "LazyImplementation",
        "Primitive",
        "PrimitiveError",
        "PrimitiveTier",
        "UnsupportedPrimitiveError",
        "core_profile",
        "declare_core_primitives",
        "lazy_implementation",
        "registered_backends",
        "registered_primitives",
        "supports",
        "validate_core_primitives",
    }
    assert all(
        getattr(primitive, name) is getattr(api_primitive, name)
        for name in primitive.__all__
    )


def test_ops_facade_has_an_explicit_complete_export_contract() -> None:
    assert set(ops.__all__) == set(api_ops.__all__)
    assert all(getattr(ops, name) is getattr(api_ops, name) for name in ops.__all__)


def test_autodiff_facade_has_an_explicit_complete_export_contract() -> None:
    assert autodiff.__all__ == ["grad_and_value", "vmap"]
    assert autodiff.grad_and_value is api_autodiff.grad_and_value
    assert autodiff.vmap is api_autodiff.vmap
