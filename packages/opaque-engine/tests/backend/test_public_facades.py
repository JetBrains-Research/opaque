"""Public façades re-export the canonical dispatch objects, unchanged."""

from __future__ import annotations

import inspect

import pytest

import opaque.autodiff as autodiff
import opaque.ops as ops
import opaque.primitive as primitive
from opaque.api.engine import autodiff as api_autodiff
from opaque.api.engine import ops as api_ops
from opaque.api.engine import primitive as api_primitive
from opaque.api.engine import runtime
from opaque.api.engine.backend import _registry, clear_backend


def test_primitive_facade_exports_the_extension_surface_only() -> None:
    """The public facade is for declaring your own operations.

    The portable-core machinery is deliberately absent: declaring a CORE
    primitive from outside Opaque makes every shipped provider incomplete
    and bricks ``set_backend`` for the process.  It stays reachable at the
    impl path for provider authors inside this repository.
    """
    assert set(primitive.__all__) == {
        "BackendProvider",
        "DuplicatePrimitiveRegistrationError",
        "IncompleteBackendError",
        "InvalidPrimitiveRegistrationError",
        "Primitive",
        "PrimitiveError",
        "PrimitiveTier",
        "UnsupportedPrimitiveError",
        "primitive",
        "registered_backends",
        "supports",
    }
    assert all(
        getattr(primitive, name) is getattr(api_primitive, name)
        for name in primitive.__all__
    )
    for provider_only in (
        "CORE_PRIMITIVES",
        "CORE_PROFILE_VERSION",
        "CoreProfile",
        "core_profile",
        "declare_core_primitives",
        "validate_core_primitives",
    ):
        assert provider_only not in primitive.__all__
        assert hasattr(api_primitive, provider_only)


def test_ops_facade_has_an_explicit_complete_export_contract() -> None:
    assert set(ops.__all__) == set(api_ops.__all__)
    assert all(getattr(ops, name) is getattr(api_ops, name) for name in ops.__all__)


def test_autodiff_facade_has_an_explicit_complete_export_contract() -> None:
    assert autodiff.__all__ == ["grad_and_value", "vmap"]
    assert autodiff.grad_and_value is api_autodiff.grad_and_value
    assert autodiff.vmap is api_autodiff.vmap


def test_vmap_exposes_only_portable_vectorization_arguments() -> None:
    assert tuple(inspect.signature(autodiff.vmap).parameters) == (
        "fn",
        "in_axes",
        "out_axes",
        "randomness",
    )
    assert tuple(inspect.signature(api_autodiff._vmap_transform).parameters) == (
        "fn",
        "in_axes",
        "out_axes",
        "randomness",
    )


def test_support_checks_do_not_depend_on_import_order() -> None:
    """Naming a backend answers about the install, not the import history.

    "Check before you use" is what a researcher writes at module scope, before
    anything has activated a provider. Answering from the registry alone made
    that answer False then and True later in the same process.
    """
    torch_provider = pytest.importorskip("opaque.api.torch")
    assert torch_provider is not None

    api_primitive._DISCOVERY_ATTEMPTED.discard(_registry.KnownBackend.TORCH)
    clear_backend()

    assert runtime.supports_profile(runtime.RuntimeProfile.DISTRIBUTED, "torch")
    # Checking support must not select a backend for subsequent dispatch.
    assert _registry.active_backend() is None
    assert not runtime.supports_profile(runtime.RuntimeProfile.DISTRIBUTED, "nosuch")


def test_support_check_without_an_active_backend_is_false_not_an_error() -> None:
    """``backend=None`` asks about the current context, which may be empty."""
    clear_backend()
    assert not api_ops.float64.supports()
    assert not runtime.supports_profile(runtime.RuntimeProfile.DISTRIBUTED)


def test_core_tier_outside_opaque_reports_its_own_cause() -> None:
    """The message must not blame the provider for the caller's declaration.

    A ``CORE`` declaration appends to the global profile every provider has to
    satisfy, so one made in user code makes ``set_backend`` fail for the rest
    of the process — including for code that never touches the extension.
    """
    error = api_primitive.IncompleteBackendError(
        "torch", ("mylab.extras.selective_log_softmax",), 5
    )
    message = str(error)
    assert "mylab.extras.selective_log_softmax" in message
    assert "declared as a CORE primitive outside Opaque" in message
    assert "PrimitiveTier.OPTIONAL" in message

    # A genuinely incomplete provider still reads as one.
    engine_error = api_primitive.IncompleteBackendError(
        "torch", ("opaque.ops.sqrt",), 5
    )
    assert "does not satisfy portable core profile" in str(engine_error)
