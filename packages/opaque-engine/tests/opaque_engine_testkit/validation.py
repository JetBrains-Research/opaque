"""Validation for inheritable portable backend conformance contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from opaque.api.engine.primitive import core_profile

_TestMethod = TypeVar("_TestMethod", bound=Callable[..., Any])
_COVERS_ATTRIBUTE = "__opaque_covered_primitives__"
_EXCEPTION_ATTRIBUTE = "__opaque_semantic_exception__"


class BackendContractValidationError(AssertionError):
    """Raised when a provider contract weakens portable conformance."""


def covers(*primitive_names: str) -> Callable[[_TestMethod], _TestMethod]:
    """Record the canonical primitives exercised by one contract method."""
    coverage = frozenset(primitive_names)

    def decorate(method: _TestMethod) -> _TestMethod:
        setattr(method, _COVERS_ATTRIBUTE, coverage)
        return method

    return decorate


def semantic_exception(
    reason: str, *, covers: tuple[str, ...] | frozenset[str]
) -> Callable[[_TestMethod], _TestMethod]:
    """Declare a justified provider-specific replacement of an inherited test."""
    if not reason.strip():
        raise ValueError("A semantic exception requires a non-empty reason.")

    def decorate(method: _TestMethod) -> _TestMethod:
        setattr(method, _COVERS_ATTRIBUTE, frozenset(covers))
        setattr(method, _EXCEPTION_ATTRIBUTE, reason)
        return method

    return decorate


def primitive_coverage(method: Callable[..., Any]) -> frozenset[str] | None:
    """Return recorded primitive coverage, if the method declares it."""
    coverage = getattr(method, _COVERS_ATTRIBUTE, None)
    return None if coverage is None else frozenset(coverage)


def contract_test_methods(contract: type) -> dict[str, Callable[..., Any]]:
    """Return effective inherited portable ``test_*`` methods."""
    return {
        name: getattr(contract, name)
        for name in dir(contract)
        if name.startswith("test_") and callable(getattr(contract, name))
    }


def validate_backend_contract(contract: type) -> None:
    """Reject incomplete core mappings and weakening provider overrides."""
    from opaque_engine_testkit.contract import BackendContractTests

    if not issubclass(contract, BackendContractTests):
        raise TypeError(f"{contract.__name__} is not a BackendContractTests subclass.")

    profile = core_profile()
    expected_version = contract.core_profile_version
    if profile.version != expected_version:
        raise BackendContractValidationError(
            f"{contract.__name__} declares core profile {expected_version}, "
            f"but the engine profile is {profile.version}."
        )

    base_methods = contract_test_methods(BackendContractTests)
    for name, base_method in base_methods.items():
        if name not in contract.__dict__:
            continue
        replacement = contract.__dict__[name]
        if not callable(replacement) or not getattr(
            replacement, _EXCEPTION_ATTRIBUTE, None
        ):
            raise BackendContractValidationError(
                f"{contract.__name__}.{name} overrides inherited coverage without "
                "@semantic_exception(...)."
            )
        base_coverage = primitive_coverage(base_method) or frozenset()
        replacement_coverage = primitive_coverage(replacement) or frozenset()
        removed = sorted(base_coverage.difference(replacement_coverage))
        if removed:
            raise BackendContractValidationError(
                f"{contract.__name__}.{name} weakens inherited coverage: {removed}"
            )

    covered = frozenset().union(
        *(
            primitive_coverage(method) or frozenset()
            for method in contract_test_methods(contract).values()
        )
    )
    required = {primitive.name for primitive in profile.primitives}
    missing = sorted(required.difference(covered))
    extra = sorted(covered.difference(required))
    if missing or extra:
        raise BackendContractValidationError(
            f"{contract.__name__} core coverage missing={missing}, extra={extra}"
        )
