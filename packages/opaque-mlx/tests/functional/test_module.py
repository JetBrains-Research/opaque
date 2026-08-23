"""Tests for MLX module functionalization."""

from __future__ import annotations

import threading

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from opaque import autodiff, execution, ops
from opaque.mlx.functional import make_functional


class _NestedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.layer(inputs)


class _FailingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.array([1.0])
        self._raise_error = True

    def __call__(self, inputs: mx.array) -> mx.array:
        if self._raise_error:
            raise RuntimeError("forward failed")
        return self.weight * inputs


class _ReentrantModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.array([1.0])
        self._nested_function = None
        self._nested_params = None
        self._binding_after_nested_call = None

    def __call__(self, inputs: mx.array) -> mx.array:
        if self._nested_function is not None:
            nested_function = self._nested_function
            self._nested_function = None
            nested_function(self._nested_params, inputs)
            self._binding_after_nested_call = self.weight
        return self.weight * inputs


class _BlockingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.array([1.0])
        self._entered = threading.Event()
        self._release = threading.Event()
        self._bound_weights: list[mx.array] = []
        self._calls_lock = threading.Lock()

    def __call__(self, inputs: mx.array) -> mx.array:
        with self._calls_lock:
            self._bound_weights.append(self.weight)
            is_first_call = len(self._bound_weights) == 1
        if is_first_call:
            self._entered.set()
            if not self._release.wait(timeout=1.0):
                raise TimeoutError("timed out waiting for the second call")
        return self.weight * inputs


def test_make_functional_uses_explicit_parameters_without_mutating_module() -> None:
    module = nn.Linear(2, 1)
    original = module.parameters()
    functional_module, parameters = make_functional(module)
    inputs = mx.array([[1.0, 2.0]])

    assert parameters["weight"] is original["weight"]
    assert parameters["bias"] is original["bias"]

    np.testing.assert_array_equal(
        ops.to_host(functional_module(parameters, inputs)), ops.to_host(module(inputs))
    )

    replacement = {"weight": mx.ones((1, 2)), "bias": mx.zeros((1,))}
    np.testing.assert_array_equal(
        ops.to_host(functional_module(replacement, inputs)), [[3.0]]
    )
    restored = module.parameters()
    assert restored["weight"] is original["weight"]
    assert restored["bias"] is original["bias"]


def test_make_functional_supports_gradients_and_trainable_partition() -> None:
    module = nn.Linear(2, 1)
    functional_module, parameters = make_functional(module)
    inputs = mx.array([[1.0, 2.0]])

    gradients, value = autodiff.grad_and_value(
        lambda explicit_params: ops.sum(functional_module(explicit_params, inputs))
    )(parameters)
    assert value.shape == ()
    assert gradients["weight"].shape == parameters["weight"].shape
    assert gradients["bias"].shape == parameters["bias"].shape

    module.freeze(keys=["bias"])
    _, trainable, frozen = make_functional(module, partition_trainable=True)
    assert set(trainable) == {"weight"}
    assert set(frozen) == {"bias"}


def test_make_functional_merges_nested_frozen_parameters() -> None:
    module = _NestedLinear()
    module.update({"layer": {"weight": mx.zeros((1, 2)), "bias": mx.array([5.0])}})
    module.layer.freeze(keys=["bias"])
    functional_module, trainable, frozen = make_functional(
        module, partition_trainable=True
    )

    assert set(trainable["layer"]) == {"weight"}
    assert set(frozen["layer"]) == {"bias"}

    output = functional_module(
        {"layer": {"weight": mx.ones((1, 2))}}, mx.array([[1.0, 2.0]])
    )

    np.testing.assert_array_equal(ops.to_host(output), [[8.0]])
    np.testing.assert_array_equal(
        ops.to_host(module.parameters()["layer"]["weight"]), [[0.0, 0.0]]
    )


def test_make_functional_restores_bindings_after_forward_error() -> None:
    module = _FailingModule()
    original = module.parameters()
    functional_module, _ = make_functional(module)

    with pytest.raises(RuntimeError, match="forward failed"):
        functional_module({"weight": mx.array([2.0])}, mx.array([3.0]))

    assert module.parameters()["weight"] is original["weight"]


def test_make_functional_restores_parent_bindings_after_nested_call() -> None:
    module = _ReentrantModule()
    original = module.parameters()
    outer_function, _ = make_functional(module)
    inner_function, _ = make_functional(module)
    outer_params = {"weight": mx.array([2.0])}
    inner_params = {"weight": mx.array([3.0])}
    module._nested_function = inner_function
    module._nested_params = inner_params

    output = outer_function(outer_params, mx.array([4.0]))

    np.testing.assert_array_equal(ops.to_host(output), [8.0])
    assert module._binding_after_nested_call is outer_params["weight"]
    assert module.parameters()["weight"] is original["weight"]


def test_make_functional_serializes_multiple_adapters_for_one_module() -> None:
    module = _BlockingModule()
    first_function, _ = make_functional(module)
    second_function, _ = make_functional(module)
    first_params = {"weight": mx.array([2.0])}
    second_params = {"weight": mx.array([3.0])}
    first_done = threading.Event()
    second_done = threading.Event()
    failures: list[Exception] = []

    def invoke(function, params, done: threading.Event) -> None:
        try:
            function(params, mx.array([1.0]))
        except Exception as error:
            failures.append(error)
        finally:
            done.set()

    first = threading.Thread(
        target=invoke, args=(first_function, first_params, first_done)
    )
    first.start()
    assert module._entered.wait(timeout=1.0)

    second = threading.Thread(
        target=invoke, args=(second_function, second_params, second_done)
    )
    second.start()
    assert not second_done.wait(timeout=0.1)
    assert len(module._bound_weights) == 1

    module._release.set()
    assert first_done.wait(timeout=1.0)
    assert second_done.wait(timeout=1.0)
    first.join()
    second.join()

    assert not failures
    assert len(module._bound_weights) == 2
    assert module._bound_weights[0] is first_params["weight"]
    assert module._bound_weights[1] is second_params["weight"]


def test_make_functional_gradients_use_explicit_parameters_after_restoration() -> None:
    module = nn.Linear(2, 1)
    original = module.parameters()
    functional_module, _ = make_functional(module)
    explicit_params = {
        "weight": mx.array([[3.0, 4.0]]),
        "bias": mx.array([5.0]),
    }
    inputs = mx.array([[1.0, 2.0]])

    gradients, value = autodiff.grad_and_value(
        lambda params: ops.sum(functional_module(params, inputs))
    )(explicit_params)

    assert ops.scalar_item(value) == pytest.approx(16.0)
    np.testing.assert_array_equal(ops.to_host(gradients["weight"]), [[1.0, 2.0]])
    np.testing.assert_array_equal(ops.to_host(gradients["bias"]), [1.0])
    assert module.parameters()["weight"] is original["weight"]
    assert module.parameters()["bias"] is original["bias"]


def test_make_functional_disable_autograd_tracking_stops_initial_parameters() -> None:
    def detached_parameter_sum(weight: mx.array) -> mx.array:
        module = nn.Linear(1, 1)
        module.update({"weight": weight, "bias": mx.zeros((1,))})
        _, parameters = make_functional(module, disable_autograd_tracking=True)
        return ops.sum(parameters["weight"])

    gradients, _ = autodiff.grad_and_value(detached_parameter_sum)(mx.array([[2.0]]))

    np.testing.assert_array_equal(ops.to_host(gradients), [[0.0]])


def test_make_functional_supports_eager_vmap_and_compile() -> None:
    module = nn.Linear(2, 1)
    original = module.parameters()
    functional_module, _ = make_functional(module)
    explicit_params = {
        "weight": mx.ones((1, 2)),
        "bias": mx.zeros((1,)),
    }

    vmapped = autodiff.vmap(lambda inputs: functional_module(explicit_params, inputs))
    np.testing.assert_array_equal(
        ops.to_host(vmapped(mx.array([[1.0, 2.0], [3.0, 4.0]]))), [[3.0], [7.0]]
    )
    assert module.parameters()["weight"] is original["weight"]

    compiled = execution.compile(
        lambda params, inputs: functional_module(params, inputs)
    )
    np.testing.assert_array_equal(
        ops.to_host(compiled(explicit_params, mx.array([[1.0, 2.0]]))), [[3.0]]
    )
    assert module.parameters()["weight"] is original["weight"]
