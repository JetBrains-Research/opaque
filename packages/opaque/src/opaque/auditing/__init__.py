"""Empirical privacy auditing for differential privacy.

One-run auditing with HuggingFace integration (Steinke et al. 2023).

Two-step API — partition first, then wrap with an estimator::

    import opaque.auditing as auditing
    from opaque.random import key

    # Partition: coin-flip canary assignment
    coin_flip = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))

    # Estimator: one-run method + scoring config
    audit_state = auditing.one_run(
        coin_flip, dataset=dataset,
        batch_argnums=(1,),
        collate_fn=data_collator,
        batch_unpack=lambda b: (b["input_ids"].to(device),),
    )
    train_data = dataset.select(audit_state.train_indices)

    # ... DP-SGD training loop ...

    audit = auditing.evaluate(per_example_loss_fn, trained_params, state=audit_state)
    print(audit.summary(delta=1e-5))

Or use the convenience :func:`setup` (combines both steps)::

    audit_state = auditing.setup(
        dataset, num_canaries=1000, key=key(42),
        batch_argnums=(1,), collate_fn=data_collator,
        batch_unpack=lambda b: (b["input_ids"].to(device),),
    )
    train_data = dataset.select(audit_state.train_indices)
    # ... train ...
    audit = auditing.evaluate(loss_fn, trained_params, state=audit_state)

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
    "evaluate",
    "one_run",
    "score",
    "setup",
]

# Sentinel for "not provided" (distinct from None which is a valid value)
_UNSET = object()


def coin_flip(
    dataset: Any,
    *,
    num_canaries: int,
    key: RngKey,
) -> CoinFlip:
    """Create a coin-flip partition for canary-based auditing.

    Randomly selects ``num_canaries`` examples from the dataset and flips
    a fair coin for each to decide inclusion/exclusion. This only handles
    the partition — wrap the result with :func:`one_run` to add scoring
    config and estimation.

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
    coin_key = fold_in(key, 1)  # Derive separate key for coin flips
    return CoinFlip(canary_indices, key=coin_key)


def one_run(
    partition: CoinFlip,
    *,
    dataset: Any,
    batch_argnums: tuple[int, ...] | None = None,
    collate_fn: Callable | None = None,
    batch_unpack: Callable | None = None,
    batch_size: int = 256,
) -> OneRunEstimator:
    """Create a one-run estimator from a coin-flip partition.

    Wraps a :class:`CoinFlip` with the dataset and scoring configuration
    so that :func:`evaluate` needs only the loss function and trained
    parameters.

    Args:
        partition: A :class:`CoinFlip` from :func:`coin_flip`.
        dataset: The full dataset (same one passed to :func:`coin_flip`).
        batch_argnums: Which ``loss_fn`` args come from dataset batches
            (same convention as ``clipped_grad``).
        collate_fn: DataLoader collate function.
        batch_unpack: Extracts tensors from a collated batch.
        batch_size: Scoring batch size. Default: 256.

    Returns:
        A :class:`OneRunEstimator` ready for :func:`evaluate`.

    Example::

        cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
        audit_state = auditing.one_run(
            cf, dataset=dataset, batch_argnums=(1,),
        )
        train_data = dataset.select(audit_state.train_indices)
    """
    return OneRunEstimator(
        partition,
        dataset=dataset,
        batch_argnums=batch_argnums,
        collate_fn=collate_fn,
        batch_unpack=batch_unpack,
        batch_size=batch_size,
    )


def setup(
    dataset: Any,
    *,
    num_canaries: int,
    key: RngKey,
    batch_argnums: tuple[int, ...] | None = None,
    collate_fn: Callable | None = None,
    batch_unpack: Callable | None = None,
    batch_size: int = 256,
) -> OneRunEstimator:
    """Set up a one-run privacy audit in one call.

    Convenience function that combines :func:`coin_flip` and
    :func:`one_run`.

    Args:
        dataset: Any dataset with ``len()`` (HuggingFace or PyTorch).
        num_canaries: Number of canary examples to designate.
        key: RNG key for reproducible canary selection and coin flips.
        batch_argnums: Which ``loss_fn`` args come from dataset batches.
        collate_fn: DataLoader collate function.
        batch_unpack: Extracts tensors from a collated batch.
        batch_size: Scoring batch size. Default: 256.

    Returns:
        A :class:`OneRunEstimator` containing the canary partition and
        scoring configuration.

    Example::

        audit_state = auditing.setup(
            dataset, num_canaries=1000, key=key(42),
            batch_argnums=(1,),
            collate_fn=data_collator,
            batch_unpack=lambda b: (b["input_ids"].to(device),),
        )
        train_data = dataset.select(audit_state.train_indices)
    """
    cf = coin_flip(dataset, num_canaries=num_canaries, key=key)
    return one_run(
        cf,
        dataset=dataset,
        batch_argnums=batch_argnums,
        collate_fn=collate_fn,
        batch_unpack=batch_unpack,
        batch_size=batch_size,
    )


def evaluate(
    loss_fn: Callable,
    *args: Any,
    state: OneRunEstimator,
    batch_argnums: tuple[int, ...] | None = _UNSET,
    dataset: Any = _UNSET,
    collate_fn: Callable | None = _UNSET,
    batch_unpack: Callable | None = _UNSET,
    batch_size: int | None = None,
) -> AuditResult:
    """Score canaries and produce audit results in one call.

    When scoring config was provided at setup time, only ``loss_fn``
    and trained parameters are needed::

        audit = auditing.evaluate(loss_fn, trained_params, state=audit_state)

    Any parameter passed here overrides the value stored in ``state``.

    Args:
        loss_fn: Per-example loss function (vmap-compatible).
        *args: Non-batched arguments to ``loss_fn`` (e.g., model params).
        state: The :class:`OneRunEstimator` from :func:`setup` or
            :func:`one_run`.
        batch_argnums: Which ``loss_fn`` args come from dataset batches.
        dataset: The full dataset.
        collate_fn: DataLoader collate function.
        batch_unpack: Batch extraction function.
        batch_size: Scoring batch size.

    Returns:
        :class:`AuditResult` with ``epsilon_at()`` defaulting to the
        ``'one_run'`` method.
    """
    # Resolve parameters: explicit > stored in state > error
    resolved_dataset = _resolve(dataset, state, "_dataset", "dataset")
    resolved_argnums = _resolve(batch_argnums, state, "_batch_argnums", "batch_argnums")
    resolved_collate = _resolve_optional(collate_fn, state, "_collate_fn")
    resolved_unpack = _resolve_optional(batch_unpack, state, "_batch_unpack")
    resolved_batch_size = (
        batch_size if batch_size is not None else getattr(state, "_batch_size", 256)
    )

    cf = state.coin_flip
    scores = score(
        loss_fn,
        *args,
        batch_argnums=resolved_argnums,
        dataset=resolved_dataset,
        indices=cf.canary_indices,
        collate_fn=resolved_collate,
        batch_unpack=resolved_unpack,
        batch_size=resolved_batch_size,
    )
    return state.audit(scores)


def _resolve(value: Any, state: OneRunEstimator, attr: str, name: str) -> Any:
    """Resolve a parameter: explicit value > stored on state > error."""
    if value is not _UNSET:
        return value
    stored = getattr(state, attr, None)
    if stored is not None:
        return stored
    raise TypeError(f"'{name}' must be provided either to setup() or evaluate()")


def _resolve_optional(value: Any, state: OneRunEstimator, attr: str) -> Any | None:
    """Resolve an optional parameter: explicit value > stored > None."""
    if value is not _UNSET:
        return value
    return getattr(state, attr, None)
