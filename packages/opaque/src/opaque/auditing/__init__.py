"""Empirical privacy auditing for differential privacy.

One-run auditing with HuggingFace integration (Steinke et al. 2023).

Quick start::

    import opaque.auditing as auditing
    from opaque.random import key

    audit_state = auditing.setup(
        dataset, num_canaries=1000, key=key(42),
        batch_argnums=(1,),
        collate_fn=data_collator,
        batch_unpack=lambda b: (b["input_ids"].to(device),),
    )
    train_data = dataset.select(audit_state.train_indices)

    # ... DP-SGD training loop ...

    result = audit_state.evaluate(per_example_loss_fn, trained_params)
    print(result.summary(delta=1e-5))

Or with a pre-built :class:`CoinFlip`::

    cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
    audit_state = auditing.setup(dataset, coin_flip=cf, batch_argnums=(1,))

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

from opaque.auditing.audit import AuditResult, CoinFlip, OneRunEstimator
from opaque.auditing.scoring import score
from opaque.random import RngKey, fold_in

__all__ = [
    "AuditResult",
    "CoinFlip",
    "OneRunEstimator",
    "coin_flip",
    "score",
    "setup",
]


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
    return _make_coin_flip(dataset, num_canaries=num_canaries, key=key)


def _make_coin_flip(
    dataset: Any,
    *,
    num_canaries: int,
    key: RngKey,
) -> CoinFlip:
    """Internal: create CoinFlip without name shadowing issues."""
    dataset_size = len(dataset)
    if num_canaries > dataset_size:
        raise ValueError(
            f"num_canaries ({num_canaries}) exceeds dataset size ({dataset_size})"
        )

    rng = np.random.default_rng(key.seed)
    canary_indices = rng.choice(dataset_size, size=num_canaries, replace=False)
    coin_key = fold_in(key, 1)  # Derive separate key for coin flips
    return CoinFlip(canary_indices, key=coin_key)


def setup(
    dataset: Any,
    *,
    num_canaries: int | None = None,
    key: RngKey | None = None,
    coin_flip: CoinFlip | None = None,
    batch_argnums: tuple[int, ...] | None = None,
    collate_fn: Callable | None = None,
    batch_unpack: Callable | None = None,
    batch_size: int = 256,
) -> OneRunEstimator:
    """Set up a one-run privacy audit.

    Creates (or accepts) a coin-flip partition and wraps it with the
    dataset and scoring configuration. The returned
    :class:`OneRunEstimator` is the main handle for the audit — call
    :meth:`~OneRunEstimator.evaluate` after training.

    Either provide ``num_canaries`` + ``key`` to create a partition
    automatically, or provide a pre-built ``coin_flip``.

    Args:
        dataset: Any dataset with ``len()`` (HuggingFace or PyTorch).
        num_canaries: Number of canary examples to designate.
            Required unless ``coin_flip`` is provided.
        key: RNG key for reproducible canary selection and coin flips.
            Required unless ``coin_flip`` is provided.
        coin_flip: A pre-built :class:`CoinFlip` partition. If provided,
            ``num_canaries`` and ``key`` are ignored.
        batch_argnums: Which ``loss_fn`` args come from dataset batches
            (same convention as ``clipped_grad``).
        collate_fn: DataLoader collate function.
        batch_unpack: Extracts tensors from a collated batch.
        batch_size: Scoring batch size. Default: 256.

    Returns:
        A :class:`OneRunEstimator` ready for
        :meth:`~OneRunEstimator.evaluate`.

    Example::

        # Create partition automatically
        audit_state = auditing.setup(
            dataset, num_canaries=1000, key=key(42),
            batch_argnums=(1,),
        )
        train_data = dataset.select(audit_state.train_indices)

        # Or with a pre-built CoinFlip
        cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
        audit_state = auditing.setup(dataset, coin_flip=cf, batch_argnums=(1,))
    """
    if coin_flip is not None:
        cf = coin_flip
    elif num_canaries is not None and key is not None:
        cf = _make_coin_flip(dataset, num_canaries=num_canaries, key=key)
    else:
        raise TypeError("Either provide coin_flip or both num_canaries and key")

    return OneRunEstimator(
        cf,
        dataset=dataset,
        batch_argnums=batch_argnums,
        collate_fn=collate_fn,
        batch_unpack=batch_unpack,
        batch_size=batch_size,
    )
