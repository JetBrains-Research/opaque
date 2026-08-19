"""Unit tests for generic RNG engine (JAX-style semantics)."""

import numpy as np
import pytest
import torch

from opaque import random
from opaque.api.engine.backend import active_backend, clear_backend, use_backend
from opaque.random import fold_in, key, split
from opaque.torch import torch_backend
from opaque.torch.random import generator_from_key

_SEED_BOUNDARIES = (0, 1, 2**63 - 1, 2**63, 2**64 - 2, 2**64 - 1)


def test_key_requires_int_seed():
    with pytest.raises(TypeError, match="seed must be int"):
        key("42")


def test_split_is_deterministic_for_same_key():
    k1 = key(123)
    k2 = key(123)
    c11, c12 = split(k1, 2)
    c21, c22 = split(k2, 2)
    assert c11.seed == c21.seed
    assert c12.seed == c22.seed


def test_split_children_are_distinct():
    k = key(123)
    c1, c2, c3 = split(k, 3)
    assert len({c1.seed, c2.seed, c3.seed}) == 3


def test_fold_in_domain_separates():
    k = key(123)
    a = fold_in(k, "noise")
    b = fold_in(k, "sampling")
    assert a.seed != b.seed


def test_generator_from_key_is_reproducible():
    k = key(999)
    g1 = generator_from_key(k)
    g2 = generator_from_key(k)
    x1 = torch.randn(16, generator=g1)
    x2 = torch.randn(16, generator=g2)
    assert torch.allclose(x1, x2)


def test_fold_in_variadic_equals_sequential():
    """fold_in(k, a, b) must equal fold_in(fold_in(k, a), b)."""
    k = key(42)
    chained = fold_in(fold_in(k, 7), 3)
    variadic = fold_in(k, 7, 3)
    assert chained.seed == variadic.seed


def test_fold_in_variadic_three_values():
    """fold_in(k, a, b, c) == fold_in(fold_in(fold_in(k, a), b), c)."""
    k = key(0)
    chained = fold_in(fold_in(fold_in(k, 1), 2), 3)
    variadic = fold_in(k, 1, 2, 3)
    assert chained.seed == variadic.seed


def test_fold_in_no_data_raises():
    """fold_in(k) with no data arguments must raise ValueError."""
    k = key(42)
    with pytest.raises(ValueError, match="at least one data argument"):
        fold_in(k)


@pytest.mark.parametrize("seed", _SEED_BOUNDARIES)
def test_keyed_normal_replays_at_engine_seed_boundaries(seed: int):
    rng_key = key(seed)

    first = random.normal(rng_key, (64,), dtype=torch.float32)
    second = random.normal(rng_key, (64,), dtype=torch.float32)

    assert torch.equal(first, second)


@pytest.mark.parametrize(
    ("first_seed", "second_seed"),
    [(0, 2**63 - 1), (0, 2**64 - 2), (1, 2**64 - 1)],
)
def test_keyed_normal_distinguishes_engine_seed_boundaries(
    first_seed: int, second_seed: int
):
    first = random.normal(key(first_seed), (64,), dtype=torch.float32)
    second = random.normal(key(second_seed), (64,), dtype=torch.float32)

    assert not torch.equal(first, second)


def test_keyed_normal_ignores_global_rng_draws():
    rng_key = key(41)
    expected = random.normal(rng_key, (64,), dtype=torch.float32)

    torch.manual_seed(17)
    torch.randn(128)
    np.random.seed(17)
    np.random.normal(size=128)
    actual = random.normal(rng_key, (64,), dtype=torch.float32)

    assert torch.equal(expected, actual)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_keyed_normal_honors_shape_dtype_and_placement(
    all_devices: torch.device, dtype: torch.dtype
):
    like = torch.empty(0, dtype=torch.float32, device=all_devices)

    sample = random.normal(key(5), (2, 3), dtype=dtype, like=like)

    assert sample.shape == (2, 3)
    assert sample.dtype is dtype
    assert sample.device == like.device


def test_keyed_normal_registration_activates_torch_provider():
    from opaque.api.torch.backend import _core as provider

    clear_backend()
    backend = torch_backend()

    assert random.normal.resolve(backend) is provider.normal
    with use_backend(backend):
        assert active_backend() is backend
        sample = random.normal(key(5), (1,), dtype=torch.float32)

    assert isinstance(sample, torch.Tensor)
