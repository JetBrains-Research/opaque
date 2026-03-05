"""Empirical privacy auditing for differential privacy.

Canary-based auditing with pluggable attacks and estimation methods.

Quick start (one-run auditing, Steinke et al. 2023)::

    import opaque.auditing as auditing
    from opaque.random import key

    cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
    train_data = dataset.select(cf.train_indices(len(dataset)))

    # ... DP-SGD training loop ...

    scores = auditing.loss_scores(loss_fn, params,
                                   coin_flip=cf, dataset=dataset,
                                   batch_argnums=(1,))
    estimate = auditing.one_run(scores, coin_flip=cf)
    print(estimate.summary(delta=1e-5))

References:
    - Steinke, Nasr, Jagielski (2023), https://arxiv.org/abs/2305.08846
    - Carlini et al. (2022), https://arxiv.org/abs/2112.03570
"""

from opaque.auditing.attacks import loss_scores
from opaque.auditing.coin_flip import CoinFlip, coin_flip
from opaque.auditing.one_run import OneRunEstimate, one_run

__all__ = [
    "CoinFlip",
    "OneRunEstimate",
    "coin_flip",
    "loss_scores",
    "one_run",
]
