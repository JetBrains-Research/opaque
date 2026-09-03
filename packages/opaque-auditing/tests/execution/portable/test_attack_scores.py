"""Portable loss and gradient attack-score behavior."""

from __future__ import annotations

import numpy as np
import pytest

import opaque.auditing as auditing
from opaque import ops
from opaque.random import key


def _dataset(backend_case) -> list[tuple[object, object]]:
    return [
        (
            backend_case.array(float(index), dtype=backend_case.dtype("float32")),
            backend_case.array(float(2 * index), dtype=backend_case.dtype("float32")),
        )
        for index in range(32)
    ]


def _squared_error(params: object, value: object, target: object) -> object:
    return ops.square(ops.subtract(ops.multiply(params, value), target))


def _score_dataset(backend_case) -> list[tuple[object, object]]:
    return [
        (
            backend_case.array(float(index), dtype=backend_case.dtype("float32")),
            backend_case.array(float(2 * index), dtype=backend_case.dtype("float32")),
        )
        for index in range(1, 25)
    ]


@pytest.mark.parametrize("score_fn", [auditing.loss_scores, auditing.gradient_scores])
def test_perfect_parameters_produce_zero_portable_attack_scores(
    backend_case, score_fn
) -> None:
    dataset = _dataset(backend_case)
    coin_flip = auditing.coin_flip(dataset, num_canaries=16, key=key(23))

    scores = score_fn(
        _squared_error,
        backend_case.array(2.0, dtype=backend_case.dtype("float32")),
        batch_argnums=(1, 2),
        coin_flip=coin_flip,
        dataset=dataset,
        batch_size=4,
    )

    assert scores.scores.shape == (coin_flip.num_canaries,)
    np.testing.assert_allclose(scores.scores, 0.0, atol=1e-5)


@pytest.mark.parametrize("score_fn", [auditing.loss_scores, auditing.gradient_scores])
def test_reference_scores_preserve_canary_alignment_portably(
    backend_case, score_fn
) -> None:
    dataset = _dataset(backend_case)
    coin_flip = auditing.coin_flip(dataset, num_canaries=16, key=key(31))
    baseline_params = backend_case.array(0.0, dtype=backend_case.dtype("float32"))
    fitted_params = backend_case.array(2.0, dtype=backend_case.dtype("float32"))
    baseline = score_fn(
        _squared_error,
        baseline_params,
        batch_argnums=(1, 2),
        coin_flip=coin_flip,
        dataset=dataset,
        batch_size=5,
    )

    scores = score_fn(
        _squared_error,
        fitted_params,
        batch_argnums=(1, 2),
        coin_flip=coin_flip,
        dataset=dataset,
        batch_size=3,
        reference_scores=baseline,
    )

    np.testing.assert_array_equal(scores.canary_indices, baseline.canary_indices)
    assert np.all(scores.scores >= baseline.scores - 1e-5)


@pytest.mark.parametrize(
    ("score_fn", "expected_score"),
    [
        (auditing.loss_scores, lambda index: -4.0 * index**2),
        (auditing.gradient_scores, lambda index: -16.0 * index**4),
    ],
)
def test_attack_scores_match_the_closed_form_for_each_canary(
    backend_case, score_fn, expected_score
) -> None:
    dataset = _score_dataset(backend_case)
    coin_flip = auditing.coin_flip(dataset, num_canaries=12, key=key(41))
    scores = score_fn(
        _squared_error,
        backend_case.array(0.0, dtype=backend_case.dtype("float32")),
        batch_argnums=(1, 2),
        coin_flip=coin_flip,
        dataset=dataset,
        batch_size=5,
    )

    expected = [expected_score(index + 1) for index in scores.canary_indices]
    np.testing.assert_allclose(scores.scores, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("score_fn", [auditing.loss_scores, auditing.gradient_scores])
def test_attack_score_batch_size_and_reference_reduction_are_portable(
    backend_case, score_fn
) -> None:
    dataset = _score_dataset(backend_case)
    coin_flip = auditing.coin_flip(dataset, num_canaries=12, key=key(53))
    baseline = score_fn(
        _squared_error,
        backend_case.array(0.0, dtype=backend_case.dtype("float32")),
        batch_argnums=(1, 2),
        coin_flip=coin_flip,
        dataset=dataset,
        batch_size=2,
    )
    current = score_fn(
        _squared_error,
        backend_case.array(1.0, dtype=backend_case.dtype("float32")),
        batch_argnums=(1, 2),
        coin_flip=coin_flip,
        dataset=dataset,
        batch_size=7,
    )
    reduction = score_fn(
        _squared_error,
        backend_case.array(1.0, dtype=backend_case.dtype("float32")),
        batch_argnums=(1, 2),
        coin_flip=coin_flip,
        dataset=dataset,
        batch_size=3,
        reference_scores=baseline,
    )

    np.testing.assert_array_equal(current.canary_indices, baseline.canary_indices)
    np.testing.assert_allclose(reduction.scores, current.scores - baseline.scores)
    assert np.all(reduction.scores > 0.0)


@pytest.mark.parametrize("score_fn", [auditing.loss_scores, auditing.gradient_scores])
def test_attack_scores_reject_invalid_portable_batch_arguments(
    backend_case, score_fn
) -> None:
    dataset = _score_dataset(backend_case)
    coin_flip = auditing.coin_flip(dataset, num_canaries=4, key=key(61))

    with pytest.raises(ValueError, match="must be sorted"):
        score_fn(
            _squared_error,
            backend_case.array(1.0, dtype=backend_case.dtype("float32")),
            batch_argnums=(2, 1),
            coin_flip=coin_flip,
            dataset=dataset,
        )
