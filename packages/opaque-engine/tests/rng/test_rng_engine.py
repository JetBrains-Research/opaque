"""Compatibility tests for Opaque's explicit RNG key derivation."""

from __future__ import annotations

import dataclasses

import pytest

from opaque.api.engine.random import _helpers
from opaque.random import fold_in, key, random_key, split
from opaque.random.types import RngKey


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (0, 0),
        (-1, 2**64 - 1),
        (2**64, 0),
        (2**64 + 17, 17),
        (-(2**64) - 17, 2**64 - 17),
    ],
)
def test_key_normalizes_seeds_to_uint64(seed: int, expected: int) -> None:
    assert key(seed).seed == expected
    assert RngKey(seed).seed == expected


@pytest.mark.parametrize("seed", [True, 1.5, "1", None])
def test_key_and_rng_key_reject_non_integer_seeds(seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be int"):
        key(seed)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="seed must be int"):
        RngKey(seed)  # type: ignore[arg-type]


def test_rng_key_is_frozen_and_preserves_implementation_label() -> None:
    rng_key = RngKey(42, impl="test_implementation")

    with pytest.raises(dataclasses.FrozenInstanceError):
        rng_key.seed = 7  # type: ignore[misc]

    assert fold_in(rng_key, 0).impl == "test_implementation"


def test_fold_in_preserves_compatible_derivation_values() -> None:
    rng_key = key(42)

    assert fold_in(rng_key, 0).seed == 6663219682538052210
    assert fold_in(rng_key, 1).seed == 2142789487919306769
    assert fold_in(rng_key, "noise", 3).seed == 6871992306301461613
    assert tuple(child.seed for child in split(rng_key, 3)) == (
        6663219682538052210,
        2142789487919306769,
        7729968134091776446,
    )


def test_fold_in_is_deterministic_sequential_and_domain_separated() -> None:
    rng_key = key(42)

    assert fold_in(rng_key, 1) == fold_in(rng_key, 1)
    assert fold_in(rng_key, "1").seed == 2828878459550303588
    assert fold_in(rng_key, 1) != fold_in(rng_key, "1")
    assert fold_in(rng_key, "stream", 3) == fold_in(fold_in(rng_key, "stream"), 3)
    assert rng_key == key(42)


def test_fold_in_supports_arbitrarily_large_integer_metadata() -> None:
    rng_key = key(42)
    large = 1 << 200

    assert fold_in(rng_key, large) == fold_in(rng_key, large)
    assert fold_in(rng_key, large) != fold_in(rng_key, -large)


@pytest.mark.parametrize("data", [True, 1.5, None, object()])
def test_fold_in_rejects_non_domain_data(data: object) -> None:
    with pytest.raises(TypeError, match="data must be int or str"):
        fold_in(key(42), data)  # type: ignore[arg-type]


def test_fold_in_requires_a_key_and_at_least_one_domain_value() -> None:
    with pytest.raises(TypeError, match="rng_key must be RngKey"):
        fold_in(42, 0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="at least one data argument"):
        fold_in(key(42))


@pytest.mark.parametrize("num", [True, 1.5, "2"])
def test_split_rejects_non_integer_counts(num: object) -> None:
    with pytest.raises(TypeError, match="num must be int"):
        split(key(42), num)  # type: ignore[arg-type]


@pytest.mark.parametrize("num", [0, -1])
def test_split_requires_a_positive_count(num: int) -> None:
    with pytest.raises(ValueError, match="num must be >= 1"):
        split(key(42), num)


def test_random_key_uses_system_entropy_and_returns_a_canonical_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_bits: list[int] = []

    def randbits(num_bits: int) -> int:
        requested_bits.append(num_bits)
        return 2**64 - 1

    monkeypatch.setattr(_helpers.secrets, "randbits", randbits)

    assert random_key() == key(2**64 - 1)
    assert requested_bits == [64]
