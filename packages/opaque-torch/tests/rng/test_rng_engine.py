"""Unit tests for generic RNG engine (JAX-style semantics)."""

import pytest
import torch

from opaque.random import fold_in, key, split
from opaque.torch.random import generator_from_key


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
