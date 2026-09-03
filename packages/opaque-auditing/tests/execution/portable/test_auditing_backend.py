"""Portable empirical auditing workflows."""

from __future__ import annotations

import numpy as np
import pytest

import opaque.auditing as auditing
from opaque import ops
from opaque.random import key


def _dataset(backend_case) -> list[object]:
    return [
        backend_case.array(float(index), dtype=backend_case.dtype("float32"))
        for index in range(64)
    ]


def _loss(params: object, value: object) -> object:
    return ops.square(params[0] - value)


@pytest.mark.parametrize("score_fn", [auditing.loss_scores, auditing.gradient_scores])
def test_attack_scores_feed_coin_flip_and_one_run(score_fn, backend_case) -> None:
    dataset = _dataset(backend_case)
    coin_flip = auditing.coin_flip(dataset, num_canaries=32, key=key(67))
    scores = score_fn(
        _loss,
        backend_case.array([0.0], dtype=backend_case.dtype("float32")),
        batch_argnums=(1,),
        coin_flip=coin_flip,
        dataset=dataset,
        batch_size=4,
    )

    assert scores.scores.shape == (coin_flip.num_canaries,)
    assert np.all(np.isfinite(scores.scores))
    estimate = auditing.one_run(scores, coin_flip=coin_flip)
    assert estimate.n_in == len(coin_flip.in_indices)
    assert estimate.n_out == len(coin_flip.out_indices)
    assert 0.0 <= estimate.attack_auc() <= 1.0
