"""Empirical privacy auditing for differential privacy.

End-to-end one-run auditing (Steinke et al. 2023)::

    import opaque.auditing as auditing

    # Start: set up canaries and coin flips
    experiment = auditing.setup(dataset, num_canaries=1000, seed=42)
    train_loader = DataLoader(experiment.subset(dataset), ...)

    # ... train model ...

    # End: score canaries and compute epsilon
    audit = auditing.evaluate(experiment, loss_fn, params, dataset)
    audit.epsilon_at(delta=1e-5)
    print(audit.summary())

Or construct an :class:`AuditResult` directly from pre-computed scores::

    audit = AuditResult(in_scores, out_scores)
    audit.epsilon_at(delta=1e-5)

References:
    - Steinke, Nasr, Jagielski (2023), https://arxiv.org/abs/2305.08846
    - Carlini et al. (2022), https://arxiv.org/abs/2112.03570
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from opaque.auditing.audit import AuditResult, CoinFlipExperiment
from opaque.auditing.bootstrap import BootstrapParams
from opaque.auditing.scoring import score_by_loss

__all__ = [
    "AuditResult",
    "BootstrapParams",
    "CoinFlipExperiment",
    "evaluate",
    "score_by_loss",
    "setup",
]


def setup(
    dataset: Any,
    *,
    num_canaries: int,
    seed: int | None = None,
) -> CoinFlipExperiment:
    """Set up a one-run privacy audit experiment.

    Randomly selects ``num_canaries`` examples from the dataset and flips
    a fair coin for each to decide inclusion/exclusion. This is the entry
    point for one-run auditing (Steinke et al. 2023).

    Args:
        dataset: A PyTorch-style dataset with ``len()``.
        num_canaries: Number of canary examples to designate.
        seed: Random seed for reproducible canary selection and coin flips.

    Returns:
        A :class:`CoinFlipExperiment` managing the canary assignment.

    Example::

        import opaque.auditing as auditing

        experiment = auditing.setup(dataset, num_canaries=1000, seed=42)
        train_data = experiment.subset(dataset)
    """
    dataset_size = len(dataset)
    if num_canaries > dataset_size:
        raise ValueError(
            f"num_canaries ({num_canaries}) exceeds dataset size ({dataset_size})"
        )

    rng = np.random.default_rng(seed)
    canary_indices = rng.choice(dataset_size, size=num_canaries, replace=False)
    coin_seed = int(rng.integers(0, 2**31))
    return CoinFlipExperiment(canary_indices, seed=coin_seed)


def evaluate(
    experiment: CoinFlipExperiment,
    loss_fn: Callable,
    params: Any,
    dataset: Any,
    *,
    batch_size: int = 256,
) -> AuditResult:
    """Score canaries and produce audit results in one call.

    Computes per-example negative loss as membership score (via
    ``torch.func.vmap``), then splits scores by coin flip.

    Args:
        experiment: The :class:`CoinFlipExperiment` from :func:`setup`.
        loss_fn: Loss function with signature ``loss_fn(params, x, y) -> scalar``.
            Must be vmap-compatible (same requirement as ``clipped_grad``).
        params: Trained model parameters.
        dataset: The full dataset (same one passed to :func:`setup`).
        batch_size: Batch size for scoring. Default: 256.

    Returns:
        :class:`AuditResult` with ``epsilon_at()`` defaulting to the
        ``'one_run'`` method.

    Example::

        audit = auditing.evaluate(experiment, loss_fn, params, dataset)
        audit.epsilon_at(delta=1e-5)
        print(audit.summary())
    """
    scores = score_by_loss(
        loss_fn,
        params,
        dataset,
        experiment._canary_indices,
        batch_size=batch_size,
    )
    return experiment.audit(scores)
