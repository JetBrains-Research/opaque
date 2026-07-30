"""Empirical privacy auditing façade — canary-based with pluggable attacks.

Quick start::

    import opaque.auditing as auditing
    from opaque.random import key
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
    print(f"ε (empirical): {estimate.eps_delta().epsilon_at(delta=1e-5):.4f}")

Result data classes (``CoinFlip``, ``OneRunEstimate``) live in
:mod:`opaque.auditing.types`.

References:
    - Xiang, Chen, Kerkouche (2025), https://arxiv.org/abs/2509.08704
    - Steinke, Nasr, Jagielski (2023), https://arxiv.org/abs/2305.08846
    - Carlini et al. (2022), https://arxiv.org/abs/2112.03570
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from opaque.api.auditing import coin_flip, gradient_scores, loss_scores, one_run

try:
    __version__ = _pkg_version("opaque-auditing")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "coin_flip",
    "gradient_scores",
    "loss_scores",
    "one_run",
]
