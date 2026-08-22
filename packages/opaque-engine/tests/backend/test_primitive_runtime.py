"""Tests for backend-name primitive dispatch and context-local activation."""

from __future__ import annotations

import asyncio
import inspect
import threading

import pytest

import opaque.primitive as facade
from opaque.api.engine import ops
from opaque.api.engine import primitive as primitive_module
from opaque.api.engine.backend import (
    BackendNotSelectedError,
    active_backend,
    clear_backend,
    set_backend,
    use_backend,
)
from opaque.api.engine.primitive import (
    CORE_PRIMITIVES,
    CORE_PROFILE_VERSION,
    BackendProvider,
    DuplicatePrimitiveRegistrationError,
    IncompleteBackendError,
    InvalidPrimitiveRegistrationError,
    Primitive,
    PrimitiveTier,
    UnsupportedPrimitiveError,
    core_profile,
    lazy_implementation,
    primitive,
    supports,
    validate_core_primitives,
)


class _Backend:
    def __init__(self, name: str) -> None:
        self.name = name


def _complete_backend(name: str) -> _Backend:
    backend = _Backend(name)
    for primitive in CORE_PRIMITIVES:  # noqa: F402
        if not primitive.supports(name):
            primitive.register(name, lambda *args, **kwargs: None)
    return backend


def test_primitive_resolves_by_active_backend_name_and_reports_missing() -> None:
    primitive = Primitive("test.runtime.dispatch")
    primitive.register("first", lambda value: f"first:{value}")
    primitive.register("second", lambda value: f"second:{value}")

    with use_backend(_complete_backend("first")):
        assert primitive("value") == "first:value"
        assert supports(primitive)

    with pytest.raises(
        UnsupportedPrimitiveError, match=r"test\.runtime\.dispatch.*missing"
    ):
        primitive.resolve("missing")
    assert not supports(primitive, "missing")


def test_registration_is_canonical_and_replacement_is_explicit() -> None:
    primitive = Primitive("test.runtime.registration")
    assert Primitive("test.runtime.registration") is primitive

    primitive.register("test", lambda: "original")
    with pytest.raises(DuplicatePrimitiveRegistrationError):
        primitive.register("test", lambda: "replacement")

    primitive.register("test", lambda: "replacement", replace=True)
    assert primitive.resolve("test")() == "replacement"
    assert primitive.registered_backends() == ("test",)


def test_primitive_decorator_preserves_function_metadata_and_signature() -> None:
    def declaration(value: object, *, scale: int = 1) -> object:
        """Scale a backend-native value."""

        raise NotImplementedError

    operation = primitive(tier=PrimitiveTier.OPTIONAL)(declaration)

    assert isinstance(operation, Primitive)
    assert operation.__name__ == declaration.__name__
    assert operation.__qualname__ == declaration.__qualname__
    assert operation.__doc__ == declaration.__doc__
    assert operation.__wrapped__ is declaration
    assert inspect.signature(operation) == inspect.signature(declaration)
    assert operation.name.endswith(".declaration")


def test_provider_implementation_decorator_registers_by_primitive_object() -> None:
    operation = Primitive("test.runtime.provider-decorator")
    provider = BackendProvider("decorated-provider")

    @provider.implements(operation)
    def implementation(value: str) -> str:
        return f"decorated:{value}"

    assert operation.resolve(provider) is implementation
    assert operation.resolve("decorated-provider")("value") == "decorated:value"
    assert provider.name == "decorated-provider"

    with pytest.raises(DuplicatePrimitiveRegistrationError):

        @provider.implements(operation)
        def duplicate(value: str) -> str:
            return value


def test_primitive_call_requires_or_infers_a_backend_before_resolution() -> None:
    operation = Primitive("test.runtime.ensure-before-resolution")
    clear_backend()

    with pytest.raises(BackendNotSelectedError):
        operation(object())


def test_lazy_implementation_is_cached_and_support_does_not_import() -> None:
    primitive = Primitive("test.runtime.lazy")
    target = lazy_implementation("operator:neg")
    primitive.register("test", target)

    assert primitive.supports("test")
    assert not target.is_resolved
    assert primitive.resolve("test")(3) == -3
    assert target.is_resolved
    assert primitive.resolve("test") is target.resolve()


def test_core_tier_adds_a_primitive_to_the_versioned_profile(monkeypatch) -> None:
    monkeypatch.setattr(primitive_module, "CORE_PRIMITIVES", [])
    primitive = Primitive("test.runtime.core-tier", tier="core")

    profile = primitive_module.core_profile()
    assert profile.version == CORE_PROFILE_VERSION
    assert profile.primitives == (primitive,)


def test_activation_validates_the_current_portable_core_profile(monkeypatch) -> None:
    core_primitive = Primitive("test.runtime.required")
    target = lazy_implementation("operator:neg")
    core_primitive.register("complete", target)
    monkeypatch.setattr(primitive_module, "CORE_PRIMITIVES", (core_primitive,))

    with pytest.raises(
        IncompleteBackendError, match=r"incomplete.*test\.runtime\.required"
    ):
        set_backend(_Backend("incomplete"))

    complete = _Backend("complete")
    with use_backend(complete):
        assert active_backend() is complete
    assert not target.is_resolved


def test_dpsgd_math_primitives_are_required_and_report_missing_capabilities() -> None:
    required = (ops.exp, ops.erf, ops.erfinv, ops.finfo_eps)
    assert all(operation in CORE_PRIMITIVES for operation in required)

    provider = BackendProvider("missing-dpsgd-math")
    for operation in CORE_PRIMITIVES:
        if operation not in required:
            provider.implements(operation)(lambda *args, **kwargs: None)

    with pytest.raises(IncompleteBackendError) as error:
        validate_core_primitives(provider)

    assert error.value.missing_primitives == tuple(
        operation.name for operation in required
    )


def test_use_backend_is_nested_exception_safe_and_context_local() -> None:
    outer = _complete_backend("outer")
    inner = _complete_backend("inner")
    previous = active_backend()

    with use_backend(outer):
        assert active_backend() is outer

        def raise_inside_inner_context() -> None:
            with use_backend(inner):
                assert active_backend() is inner
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            raise_inside_inner_context()
        assert active_backend() is outer
    assert active_backend() is previous

    seen: list[object] = []

    def worker() -> None:
        seen.append(active_backend())
        with use_backend(inner):
            seen.append(active_backend())

    with use_backend(outer):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert active_backend() is outer

    assert seen[1] is inner
    assert seen[0] is not outer


def test_async_contexts_do_not_overwrite_each_other() -> None:
    async def resolve(name: str) -> str:
        backend = _complete_backend(name)
        with use_backend(backend):
            await asyncio.sleep(0)
            return active_backend().name

    async def run() -> list[str]:
        return await asyncio.gather(resolve("one"), resolve("two"))

    assert asyncio.run(run()) == ["one", "two"]


def test_primitive_facade_is_reexport_only() -> None:
    """The facade adapts nothing; it re-exports the extension surface.

    The portable-core machinery stays behind the impl path — a CORE
    declaration from outside Opaque makes every shipped provider incomplete —
    so it is checked there rather than through the facade.
    """
    assert facade.Primitive is Primitive
    assert facade.primitive is primitive
    assert facade.BackendProvider is BackendProvider
    assert core_profile().version == CORE_PROFILE_VERSION
    assert tuple(CORE_PRIMITIVES) == core_profile().primitives


def test_neutral_primitive_answers_an_unselected_context_from_its_declaration() -> None:
    """The case that keeps capability probes out of calling code.

    Callers of an operation with a correct backend-free answer should make
    the call, not select a backend to ask whether they may.
    """

    @primitive(name="test.runtime.neutral-unselected", neutral=True)
    def probe(value: object) -> str:
        return f"neutral:{value}"

    clear_backend()

    assert probe("x") == "neutral:x"
    # Answering neutrally is not a selection: nothing was activated to say
    # "no provider owns this".
    assert active_backend() is None


def test_neutral_primitive_prefers_the_active_backend_implementation() -> None:
    @primitive(name="test.runtime.neutral-registered", neutral=True)
    def probe(value: object) -> str:
        return f"neutral:{value}"

    probe.register("neutral-registered", lambda value: f"provider:{value}")

    with use_backend(_complete_backend("neutral-registered")):
        assert probe("x") == "provider:x"


def test_neutral_primitive_covers_a_backend_that_registered_nothing() -> None:
    """Support and resolution stay about registrations; the call degrades."""

    @primitive(name="test.runtime.neutral-unregistered", neutral=True)
    def probe(value: object) -> str:
        return f"neutral:{value}"

    backend = _complete_backend("neutral-unregistered")

    with use_backend(backend):
        assert not probe.supports()
        assert probe("x") == "neutral:x"
        with pytest.raises(UnsupportedPrimitiveError) as error:
            probe.resolve()

    assert error.value.primitive_name == "test.runtime.neutral-unregistered"


def test_a_primitive_without_a_neutral_default_still_fails_closed() -> None:
    """Neutrality is opt-in: an unimplemented capability must still raise."""

    @primitive(name="test.runtime.neutral-absent")
    def probe(value: object) -> str:
        raise NotImplementedError

    backend = _complete_backend("neutral-absent")

    with use_backend(backend), pytest.raises(UnsupportedPrimitiveError):
        probe("x")

    clear_backend()
    with pytest.raises(BackendNotSelectedError):
        probe("x")


def test_neutral_default_is_part_of_a_primitive_identity() -> None:
    Primitive("test.runtime.neutral-identity", neutral=True)

    with pytest.raises(InvalidPrimitiveRegistrationError, match="backend-neutral"):
        Primitive("test.runtime.neutral-identity")
