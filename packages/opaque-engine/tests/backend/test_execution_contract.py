"""Provider-neutral execution-transform contract tests."""

from __future__ import annotations

import inspect
from enum import StrEnum
from typing import Any

import pytest

import opaque.execution as execution_facade
from opaque.api.engine import execution, ops
from opaque.api.engine.autodiff import _deferred_transform
from opaque.api.engine.backend import (
    BackendNotSelectedError,
    clear_backend,
    use_backend,
)
from opaque.api.engine.primitive import (
    CORE_PRIMITIVES,
    BackendProvider,
    PrimitiveTier,
    UnsupportedPrimitiveError,
    core_profile,
    primitive,
)
from opaque.api.engine.pytree import tree_map


class _Backend:
    def __init__(self, name: str) -> None:
        self.name = name


class _Array:
    """Native-array stand-in with no framework-shaped dimension methods."""

    def __init__(self, value_shape: tuple[int, ...]) -> None:
        self.shape = value_shape


def _python_tree_map(fn, tree):
    if isinstance(tree, dict):
        return {key: _python_tree_map(fn, value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_python_tree_map(fn, value) for value in tree]
    if isinstance(tree, tuple):
        return type(tree)(_python_tree_map(fn, value) for value in tree)
    return fn(tree)


def _complete_backend(name: str) -> _Backend:
    provider = BackendProvider(name)
    implementations = {
        ops.is_array: lambda value: isinstance(value, _Array),
        ops.shape: lambda value: value.shape,
        ops.expand_dims: lambda value, axis: _Array(
            (*value.shape[:axis], 1, *value.shape[axis:])
        ),
        ops.squeeze: lambda value, axis=None: _Array(
            tuple(
                size
                for index, size in enumerate(value.shape)
                if size != 1 or (axis is not None and index != axis)
            )
        ),
        tree_map: _python_tree_map,
    }
    for operation in CORE_PRIMITIVES:
        provider.implements(operation)(
            implementations.get(operation, lambda *args, **kwargs: None)
        )
    return _Backend(name)


@primitive(tier=PrimitiveTier.OPTIONAL, name="opaque.execution._test_mock_transform")
def _mock_transform(fn: Any) -> Any:
    """Mock optional transform used only for deferred-cache tests."""
    raise NotImplementedError


@pytest.fixture(autouse=True)
def _unselected_backend():
    clear_backend()
    yield
    clear_backend()


def test_execution_profile_enum_values_and_discovery() -> None:
    assert issubclass(execution.ExecutionProfile, StrEnum)
    assert [profile.value for profile in execution.ExecutionProfile] == [
        "compilation",
        "checkpointing",
        "saved_activations",
    ]
    assert execution.EXECUTION_PROFILE_VERSION == 1

    snapshot = execution.ExecutionProfileSnapshot(
        version=execution.EXECUTION_PROFILE_VERSION,
        primitives=execution.profile_primitives(execution.ExecutionProfile.COMPILATION),
    )
    assert snapshot.version == 1
    assert snapshot.primitives == (execution._compile_transform,)

    for profile in execution.ExecutionProfile:
        assert all(
            operation.tier is PrimitiveTier.OPTIONAL
            for operation in execution.profile_primitives(profile)
        )


def test_execution_primitives_are_not_in_core_profile() -> None:
    execution_primitives = {
        execution._compile_transform,
        execution._checkpoint_transform,
        execution._optimize_saved_activations_transform,
    }
    assert execution_primitives.isdisjoint(set(core_profile().primitives))
    assert all(
        operation.tier is PrimitiveTier.OPTIONAL for operation in execution_primitives
    )


def test_execution_primitive_signatures_are_provider_neutral() -> None:
    for wrapper in (
        execution.compile,
        execution.checkpoint,
        execution.optimize_saved_activations,
    ):
        signature = inspect.signature(wrapper)
        assert tuple(signature.parameters) == ("fn",)


def test_public_wrappers_defer_resolution_until_call() -> None:
    def identity(x):
        return x

    for wrapper in (
        execution.compile,
        execution.checkpoint,
        execution.optimize_saved_activations,
    ):
        transformed = wrapper(identity)
        assert callable(transformed)
        with pytest.raises(BackendNotSelectedError):
            transformed()


def test_unsupported_backend_raises_on_call() -> None:
    backend = _complete_backend("execution-unsupported")

    with use_backend(backend), pytest.raises(UnsupportedPrimitiveError) as error:
        execution.compile(lambda x: x)()

    assert error.value.primitive_name == execution._compile_transform.name
    assert error.value.backend_name == backend.name


def test_profile_support_reflects_registered_backends() -> None:
    registered = _complete_backend("execution-registered")
    unregistered = _complete_backend("execution-unregistered")

    provider = BackendProvider(registered.name)
    provider.implements(execution._compile_transform)(lambda fn: fn)

    assert execution.ExecutionProfile.COMPILATION.supports(registered.name)
    assert not execution.ExecutionProfile.COMPILATION.supports(unregistered.name)


def test_deferred_transform_caches_per_backend_and_switches() -> None:
    factory_calls: list[str] = []

    def make_impl(name: str):
        def impl(fn):
            factory_calls.append(name)
            return lambda *args, **kwargs: f"{name}:{fn(*args, **kwargs)}"

        return impl

    backend_a = _complete_backend("deferred-a")
    backend_b = _complete_backend("deferred-b")
    provider_a = BackendProvider(backend_a.name)
    provider_b = BackendProvider(backend_b.name)
    provider_a.implements(_mock_transform)(make_impl("a"))
    provider_b.implements(_mock_transform)(make_impl("b"))

    def increment(x: int) -> int:
        return x + 1

    executable = _deferred_transform(_mock_transform, increment)

    with use_backend(backend_a):
        assert executable(1) == "a:2"
        assert executable(2) == "a:3"

    with use_backend(backend_b):
        assert executable(10) == "b:11"
        assert executable(20) == "b:21"

    with use_backend(backend_a):
        assert executable(0) == "a:1"

    assert factory_calls == ["a", "b"]


def test_execution_facade_exports_match_engine_module() -> None:
    assert execution_facade.__all__ == execution.__all__
    for name in execution.__all__:
        assert getattr(execution_facade, name) is getattr(execution, name)
