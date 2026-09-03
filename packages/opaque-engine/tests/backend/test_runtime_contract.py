"""Provider-neutral runtime contract tests."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest

from opaque.api.engine import ops, runtime
from opaque.api.engine.backend import (
    BackendNotSelectedError,
    clear_backend,
    use_backend,
)
from opaque.api.engine.functional import with_batch_dim
from opaque.api.engine.primitive import (
    CORE_PRIMITIVES,
    BackendProvider,
    PrimitiveTier,
    UnsupportedPrimitiveError,
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


@pytest.fixture(autouse=True)
def _unselected_backend():
    clear_backend()
    yield
    clear_backend()


def test_runtime_value_types_are_normalized_and_truthful() -> None:
    assert tuple(runtime.ReduceOp) == (
        runtime.ReduceOp.SUM,
        runtime.ReduceOp.MEAN,
        runtime.ReduceOp.MIN,
        runtime.ReduceOp.MAX,
        runtime.ReduceOp.PRODUCT,
    )
    assert [operation.value for operation in runtime.ReduceOp] == [
        "sum",
        "mean",
        "min",
        "max",
        "product",
    ]

    stats = runtime.MemoryStats()
    assert stats == runtime.MemoryStats(
        active_bytes=None,
        cached_bytes=None,
        peak_active_bytes=None,
        capacity_bytes=None,
    )
    with pytest.raises(FrozenInstanceError):
        stats.active_bytes = 0


def test_runtime_primitive_signatures_are_provider_neutral() -> None:
    reduce_signature = inspect.signature(runtime.distributed_all_reduce)
    assert tuple(reduce_signature.parameters) == ("value", "op")
    assert reduce_signature.parameters["op"].default is runtime.ReduceOp.SUM
    assert reduce_signature.return_annotation in (object, "object")

    gather_signature = inspect.signature(runtime.distributed_all_gather)
    assert tuple(gather_signature.parameters) == ("value", "axis")
    assert gather_signature.parameters["axis"].kind is inspect.Parameter.KEYWORD_ONLY
    assert gather_signature.parameters["axis"].default == 0

    barrier_signature = inspect.signature(runtime.distributed_barrier)
    assert barrier_signature.parameters["name"].default is None

    for operation in (runtime.synchronize, runtime.memory_stats):
        signature = inspect.signature(operation)
        assert tuple(signature.parameters) == ("device",)
        assert signature.parameters["device"].default is None


def test_named_runtime_profiles_are_versioned_and_machine_checkable() -> None:
    assert runtime.RUNTIME_PROFILE_VERSION == 2
    assert runtime.profile_primitives(runtime.RuntimeProfile.DISTRIBUTED) == (
        runtime.distributed_initialized,
        runtime.distributed_all_reduce,
        runtime.distributed_all_gather,
        runtime.distributed_all_gather_object,
        runtime.distributed_rank,
        runtime.distributed_world_size,
        runtime.distributed_barrier,
    )
    assert runtime.profile_primitives("observability") == (
        runtime.synchronize,
        runtime.memory_stats,
    )
    assert all(
        operation.tier is PrimitiveTier.OPTIONAL
        for profile in runtime.RuntimeProfile
        for operation in runtime.profile_primitives(profile)
    )


def test_profile_support_is_derived_from_registered_primitives() -> None:
    complete = BackendProvider("runtime-profile-complete")
    for operation in runtime.profile_primitives(runtime.RuntimeProfile.DISTRIBUTED):
        complete.implements(operation)(lambda *args, **kwargs: None)

    incomplete = BackendProvider("runtime-profile-incomplete")
    for operation in runtime.profile_primitives(runtime.RuntimeProfile.DISTRIBUTED)[
        :-1
    ]:
        incomplete.implements(operation)(lambda *args, **kwargs: None)

    assert runtime.supports_profile(runtime.RuntimeProfile.DISTRIBUTED, complete.name)
    assert not runtime.supports_profile(
        runtime.RuntimeProfile.DISTRIBUTED, incomplete.name
    )
    assert not runtime.supports_profile(
        runtime.RuntimeProfile.OBSERVABILITY, complete.name
    )


def test_reduction_returns_a_new_value_without_mutating_the_input() -> None:
    backend = _complete_backend("runtime-return-semantics")
    provider = BackendProvider(backend)

    @provider.implements(runtime.distributed_all_reduce)
    def all_reduce(value, op=runtime.ReduceOp.SUM):
        assert op is runtime.ReduceOp.SUM
        return [*value, 3]

    value = [1, 2]
    with use_backend(backend):
        reduced = runtime.distributed_all_reduce(value)

    assert reduced == [1, 2, 3]
    assert reduced is not value
    assert value == [1, 2]


def test_unsupported_runtime_capability_fails_explicitly() -> None:
    backend = _complete_backend("runtime-unsupported")

    with use_backend(backend), pytest.raises(UnsupportedPrimitiveError) as error:
        runtime.memory_stats()

    assert error.value.primitive_name == runtime.memory_stats.name
    assert error.value.backend_name == backend.name

    with pytest.raises(BackendNotSelectedError):
        runtime.distributed_rank()


@pytest.mark.parametrize(
    ("call", "primitive_name"),
    [
        (lambda: runtime.distributed_rank(), "opaque.runtime.distributed.rank"),
        (
            lambda: runtime.distributed_all_gather_object(None),
            "opaque.runtime.distributed.all_gather_object",
        ),
        (lambda: runtime.synchronize(), "opaque.runtime.observability.synchronize"),
        (lambda: runtime.memory_stats(), "opaque.runtime.observability.memory_stats"),
        (
            lambda: runtime.trace_scope("runtime-contract"),
            "opaque.runtime.observability.trace_scope",
        ),
    ],
)
def test_optional_runtime_capabilities_fail_at_the_public_call_site(
    call, primitive_name: str
) -> None:
    backend = _complete_backend(
        f"runtime-core-only-{primitive_name.rsplit('.', 1)[-1]}"
    )

    with use_backend(backend), pytest.raises(UnsupportedPrimitiveError) as error:
        call()

    assert error.value.primitive_name == primitive_name
    assert error.value.backend_name == backend.name


def test_torch_shaped_declarations_are_absent_from_shared_runtime() -> None:
    removed = {
        "device_capabilities",
        "device_fused_kernels_available",
        "device_sdpa_autocast_under_vmap_broken",
        "distributed_all_reduce_",
        "distributed_dataset_subset",
        "distributed_gather_for_metrics",
        "distributed_is_initialized",
        "distributed_reduce_scalar",
        "functional_make_functional",
        "profiling_empty_cache",
        "profiling_memory_stats",
        "profiling_normalize_device",
        "profiling_reset_peak_memory",
        "profiling_synchronize",
        "profiling_trace_scope",
    }
    assert removed.isdisjoint(runtime.__all__)
    assert all(not hasattr(runtime, name) for name in removed)


def test_with_batch_dim_uses_provider_ops_and_does_not_mutate_outputs() -> None:
    backend = _complete_backend("runtime-axis-ops")
    returned_outputs = []

    def fn(value):
        assert value.shape == (1, 3)
        output = {"array": _Array((1, 2)), "metadata": "kept"}
        returned_outputs.append(output)
        return output

    wrapped = with_batch_dim(fn, batch_argnums=0, min_ndim=2)
    with use_backend(backend):
        result = wrapped(_Array((3,)))

    assert result["array"].shape == (2,)
    assert result["metadata"] == "kept"
    assert result is not returned_outputs[0]
    assert returned_outputs[0]["array"].shape == (1, 2)
