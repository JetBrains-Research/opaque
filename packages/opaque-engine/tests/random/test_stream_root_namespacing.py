"""Opaque's own stream roots stay outside the caller's key space.

``fold_in`` folds integers and strings down disjoint hashing paths, and
``split(key, n)`` is defined as ``fold_in(key, i) for i in range(n)``. So the
integers are the caller's: steps, ranks, leaf and group indices, and every key
``split`` hands back. Any stream root Opaque derives for itself must therefore
be a namespaced string — an integer root would be the *same key* a caller gets
from an ordinary derivation, and two mechanisms sharing a base key would draw
identical noise with nothing to signal it.
"""

from __future__ import annotations

import pytest

from opaque.api.engine.noise_allocation import (
    PAIRED_FIRST_STREAM_FOLD,
    PAIRED_SECOND_STREAM_FOLD,
)
from opaque.random import fold_in, key, split

# Every stream root the monorepo derives for itself, collected here so a new
# one cannot be added as a bare integer without this test noticing.  The
# published convention and this list must agree — see the tag table in
# ``docs/reference/rng.md``.
_INTERNAL_STREAM_ROOTS: tuple[tuple[str, str], ...] = (
    ("engine.paired.first", PAIRED_FIRST_STREAM_FOLD),
    ("engine.paired.second", PAIRED_SECOND_STREAM_FOLD),
)


def _dpftrl_roots() -> tuple[tuple[str, str], ...]:
    """Import lazily so this engine test does not require opaque-dpftrl."""
    second_moment = pytest.importorskip("opaque.api.dpftrl.noise._second_moment")
    mf_engine = pytest.importorskip("opaque.api.dpftrl.noise._engine")
    return (
        ("dpftrl.second_moment.first", second_moment.SECOND_MOMENT_FIRST_STREAM_FOLD),
        ("dpftrl.second_moment.second", second_moment.SECOND_MOMENT_SECOND_STREAM_FOLD),
        ("dpftrl.mf_gaussian", mf_engine.MF_GAUSSIAN_STREAM_FOLD),
    )


def _dpsgd_roots() -> tuple[tuple[str, str], ...]:
    """Import lazily so this engine test does not require opaque-dpsgd."""
    gaussian = pytest.importorskip("opaque.api.dpsgd.noise._gaussian")
    adaptive = pytest.importorskip("opaque.api.dpsgd.clipping._adaptive")
    return (
        ("dpsgd.gaussian", gaussian.GAUSSIAN_STREAM_FOLD),
        ("dpsgd.adaptive_clipping", adaptive.ADAPTIVE_CLIPPING_STREAM_FOLD),
    )


def _auditing_roots() -> tuple[tuple[str, str], ...]:
    """Import lazily so this engine test does not require opaque-auditing."""
    coin_flip = pytest.importorskip("opaque.api.auditing._coin_flip")
    return (
        ("auditing.canary_selection", coin_flip._CANARY_SELECTION_DOMAIN),
        ("auditing.coin_flip", coin_flip._COIN_FLIP_DOMAIN),
    )


def _all_roots() -> tuple[tuple[str, str], ...]:
    return _INTERNAL_STREAM_ROOTS + _dpsgd_roots() + _dpftrl_roots() + _auditing_roots()


def test_stream_roots_are_strings_not_integers() -> None:
    for name, tag in _all_roots():
        assert isinstance(tag, str), f"{name} must be a namespaced string, got {tag!r}"


def test_caller_integer_derivations_cannot_reach_a_stream_root() -> None:
    """The failure this guards: `split(base)` aliasing an internal root.

    Before stream roots were namespaced, ``split(base, 2)`` returned exactly
    the paired-stream and second-moment roots, so a caller's second key was
    byte-identical to a key Opaque was already drawing from.
    """
    base = key(42)

    reachable = {child.seed for child in split(base, 16)}
    reachable |= {fold_in(base, index).seed for index in range(512)}
    reachable |= {fold_in(base, -index).seed for index in range(1, 64)}

    for name, tag in _all_roots():
        assert fold_in(base, tag).seed not in reachable, (
            f"{name} is reachable by folding an integer into the same base key"
        )


def test_stream_roots_are_mutually_distinct() -> None:
    base = key(7)
    seeds = {name: fold_in(base, tag).seed for name, tag in _all_roots()}
    assert len(set(seeds.values())) == len(seeds), seeds


def test_integer_and_string_tags_occupy_disjoint_spaces() -> None:
    """A string tag is never reachable by folding the integer that prints alike."""
    base = key(1234)
    for value in (0, 1, 2, 17):
        assert fold_in(base, value).seed != fold_in(base, str(value)).seed
