"""Coin-flip partitioning for canary-based privacy auditing.

Shared infrastructure used by all auditing approaches (OneRun, Nasr, etc.).
Each canary is independently included or excluded from training with
probability 0.5 (a fair coin flip).

Reference:
    Steinke, Nasr, Jagielski. "Privacy Auditing with One (1) Training
    Run." NeurIPS 2023. https://arxiv.org/abs/2305.08846
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import Subset

from opaque.random import RngKey, fold_in

__all__ = ["CoinFlip", "coin_flip"]


class CoinFlip:
    """Coin-flip partitioning for canary-based privacy auditing.

    Each canary is independently included or excluded from training
    with probability 0.5 (a fair coin flip). This class only handles
    the partition — it does not know about scoring or epsilon estimation.

    Attributes:
        num_canaries: Total number of canary examples.
        canary_indices: All canary dataset indices.
        in_indices: Canary indices included in training (coin = heads).
        out_indices: Canary indices excluded from training (coin = tails).
    """

    def __init__(
        self,
        canary_indices: np.ndarray,
        *,
        key: RngKey,
    ) -> None:
        canary_indices = np.asarray(canary_indices)
        if canary_indices.ndim != 1 or canary_indices.size == 0:
            raise ValueError("canary_indices must be a non-empty 1-D array")

        rng = np.random.default_rng(key.seed)
        in_mask = rng.random(len(canary_indices)) < 0.5

        self.num_canaries = len(canary_indices)
        self.canary_indices = canary_indices
        self._in_mask = in_mask
        self.in_indices = canary_indices[in_mask]
        self.out_indices = canary_indices[~in_mask]

    def __repr__(self) -> str:
        return (
            f"CoinFlip(num_canaries={self.num_canaries}, "
            f"n_in={len(self.in_indices)}, n_out={len(self.out_indices)})"
        )

    def train_indices(self, dataset_size: int) -> list[int]:
        """Dataset indices to use for training.

        Returns all indices in ``range(dataset_size)`` except the excluded
        canaries (coin = tails).

        Args:
            dataset_size: Total number of examples in the full dataset.

        Returns:
            Sorted list of training indices.
        """
        excluded = set(self.out_indices.tolist())
        return [i for i in range(dataset_size) if i not in excluded]

    def canary_subset(self, dataset: Any) -> Subset:
        """Return a ``Subset`` containing only canary examples.

        Args:
            dataset: The full dataset (before filtering out held-out canaries).

        Returns:
            ``torch.utils.data.Subset`` over canary indices.
        """
        return Subset(dataset, self.canary_indices.tolist())

    def train_subset(self, dataset: Any) -> Subset:
        """Return a ``Subset`` containing all training examples.

        Includes all non-canary examples plus included canaries (coin = heads).
        Excludes held-out canaries (coin = tails).

        Args:
            dataset: The full dataset.

        Returns:
            ``torch.utils.data.Subset`` over training indices.
        """
        return Subset(dataset, self.train_indices(len(dataset)))

    def split_scores(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split per-canary scores into in-group and out-group.

        Args:
            scores: Membership scores, shape ``(num_canaries,)``, in the
                same order as ``canary_indices``.

        Returns:
            ``(in_scores, out_scores)`` tuple.
        """
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (self.num_canaries,):
            raise ValueError(
                f"Expected {self.num_canaries} scores, got shape {scores.shape}"
            )
        return scores[self._in_mask], scores[~self._in_mask]


def coin_flip(
    dataset: Any,
    *,
    num_canaries: int,
    key: RngKey,
) -> CoinFlip:
    """Create a coin-flip partition for canary-based auditing.

    Randomly selects ``num_canaries`` examples from the dataset and flips
    a fair coin for each to decide inclusion/exclusion.

    Args:
        dataset: Any dataset with ``len()`` (HuggingFace or PyTorch).
        num_canaries: Number of canary examples to designate.
        key: RNG key for reproducible canary selection and coin flips.

    Returns:
        A :class:`CoinFlip` with the canary partition.

    Example::

        cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
        train_data = dataset.select(cf.train_indices(len(dataset)))
    """
    dataset_size = len(dataset)
    if num_canaries > dataset_size:
        raise ValueError(
            f"num_canaries ({num_canaries}) exceeds dataset size ({dataset_size})"
        )

    rng = np.random.default_rng(key.seed)
    canary_indices = rng.choice(dataset_size, size=num_canaries, replace=False)
    coin_key = fold_in(key, 1)
    return CoinFlip(canary_indices, key=coin_key)
