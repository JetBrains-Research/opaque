"""MLX-backed empirical auditing workflows."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import opaque.auditing as auditing
from opaque.random import key


def _dataset() -> list[mx.array]:
    return [mx.array(float(index), dtype=mx.float32) for index in range(64)]


def _loss(params: mx.array, value: mx.array) -> mx.array:
    return mx.square(params[0] - value)


@pytest.mark.parametrize("score_fn", [auditing.loss_scores, auditing.gradient_scores])
def test_mlx_attack_scores_feed_coin_flip_and_one_run(score_fn) -> None:
    dataset = _dataset()
    coin_flip = auditing.coin_flip(dataset, num_canaries=32, key=key(67))
    scores = score_fn(
        _loss,
        mx.array([0.0], dtype=mx.float32),
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
