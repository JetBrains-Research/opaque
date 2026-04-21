"""Empirical privacy auditing for differential privacy.

Canary-based auditing with pluggable attacks and estimation methods.

Quick start (one-run auditing, Steinke et al. 2023)::

    import opaque.auditing as auditing
    from opaque.core.random import key
    from torch.utils.data import DataLoader, Subset

    cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
    train_data = dataset.select(cf.train_indices(len(dataset)))

    # ... DP-SGD training loop ...

    canary_loader = DataLoader(
        Subset(dataset, cf.canary_indices.tolist()),
        batch_size=32, collate_fn=canary_collate,
    )
    scores = auditing.loss_scores(loss_fn, params,
                                   batch_argnums=(1,),
                                   dataloader=canary_loader)
    estimate = auditing.one_run(scores, coin_flip=cf)
    print(f"ε (empirical): {estimate.epsilon_at(delta=1e-5):.4f}")

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
