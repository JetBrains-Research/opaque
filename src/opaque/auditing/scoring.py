"""Membership scoring utilities for privacy auditing.

Computes per-example membership scores using the same ``torch.func.vmap``
pattern as :func:`opaque.clipped_grad`. Higher scores indicate higher
likelihood of being a training member.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["score_by_loss"]


def score_by_loss(
    loss_fn: Callable,
    params: Any,
    dataset: Any,
    indices: np.ndarray | None = None,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute membership scores as negative per-example loss.

    Higher score = lower loss = more likely a training member.
    Uses ``torch.func.vmap`` for per-example loss computation, consistent
    with opaque's :func:`~opaque.clipped_grad` pattern.

    Args:
        loss_fn: Loss function with signature ``loss_fn(params, x, y) -> scalar``.
            Must be vmap-compatible (same requirement as ``clipped_grad``).
        params: Model parameters (same as used in training).
        dataset: A PyTorch-style dataset returning ``(x, y, ...)`` tuples.
        indices: If provided, only score these dataset indices. Typically
            ``experiment._canary_indices``.
        batch_size: Batch size for scoring. Default: 256.

    Returns:
        Array of membership scores, shape ``(n,)`` where n is the number
        of scored examples. Scores are negated losses (higher = more likely
        member).
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    if indices is not None:
        subset = Subset(dataset, np.asarray(indices).tolist())
    else:
        subset = dataset

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    per_example_fn = torch.func.vmap(loss_fn, in_dims=(None, 0, 0))

    all_scores = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            losses = per_example_fn(params, x, y)
            all_scores.append(-losses.detach().cpu().numpy())

    return np.concatenate(all_scores)
