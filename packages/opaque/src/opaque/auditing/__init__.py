"""Empirical privacy auditing for differential privacy.

One-run auditing with HuggingFace integration (Steinke et al. 2023).

Recommended pattern — configure scoring at setup, evaluate in one line::

    import opaque.auditing as auditing
    from opaque.random import key

    # Setup: specify dataset + scoring config once
    experiment = auditing.setup(
        dataset, num_canaries=1000, key=key(42),
        batch_argnums=(1,),
        collate_fn=data_collator,
        batch_unpack=lambda b: (b["input_ids"].to(device),),
    )
    train_data = dataset.select(experiment.train_indices(len(dataset)))

    # ... DP-SGD training loop ...

    # Evaluate: just pass loss_fn and trained params
    audit = auditing.evaluate(experiment, per_example_loss_fn, trained_params)
    print(audit.summary(delta=1e-5))

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
from opaque.auditing.scoring import score
from opaque.random import RngKey, fold_in

__all__ = [
    "AuditResult",
    "CoinFlipExperiment",
    "evaluate",
    "score",
    "setup",
]

# Sentinel for "not provided" (distinct from None which is a valid value)
_UNSET = object()


def setup(
    dataset: Any,
    *,
    num_canaries: int,
    key: RngKey,
    batch_argnums: tuple[int, ...] | None = None,
    collate_fn: Callable | None = None,
    batch_unpack: Callable | None = None,
    batch_size: int = 256,
) -> CoinFlipExperiment:
    """Set up a one-run privacy audit experiment.

    Randomly selects ``num_canaries`` examples from the dataset and flips
    a fair coin for each to decide inclusion/exclusion. Optionally stores
    the dataset and scoring configuration so that :func:`evaluate` requires
    only the loss function and trained parameters.

    Args:
        dataset: Any dataset with ``len()`` (HuggingFace or PyTorch).
        num_canaries: Number of canary examples to designate.
        key: RNG key for reproducible canary selection and coin flips.
        batch_argnums: Which ``loss_fn`` args come from dataset batches
            (same convention as ``clipped_grad``). Stored for
            :func:`evaluate`.
        collate_fn: DataLoader collate function (e.g.,
            ``DataCollatorForLanguageModeling``). Stored for
            :func:`evaluate`.
        batch_unpack: Extracts tensors from a collated batch (e.g.,
            ``lambda b: (b["input_ids"].to(device),)``). Stored for
            :func:`evaluate`.
        batch_size: Scoring batch size. Default: 256.

    Returns:
        A :class:`CoinFlipExperiment` managing the canary assignment.

    Example (recommended — configure scoring here)::

        experiment = auditing.setup(
            dataset, num_canaries=1000, key=key(42),
            batch_argnums=(1,),
            collate_fn=data_collator,
            batch_unpack=lambda b: (b["input_ids"].to(device),),
        )
        train_data = dataset.select(experiment.train_indices(len(dataset)))

    Example (minimal — configure scoring at evaluate time)::

        experiment = auditing.setup(dataset, num_canaries=1000, key=key(42))
        train_data = experiment.subset(dataset)
    """
    dataset_size = len(dataset)
    if num_canaries > dataset_size:
        raise ValueError(
            f"num_canaries ({num_canaries}) exceeds dataset size ({dataset_size})"
        )

    rng = np.random.default_rng(key.seed)
    canary_indices = rng.choice(dataset_size, size=num_canaries, replace=False)
    coin_key = fold_in(key, 1)  # Derive separate key for coin flips
    experiment = CoinFlipExperiment(canary_indices, key=coin_key)

    # Store dataset and scoring config for evaluate()
    experiment._dataset = dataset
    experiment._batch_argnums = batch_argnums
    experiment._collate_fn = collate_fn
    experiment._batch_unpack = batch_unpack
    experiment._batch_size = batch_size

    return experiment


def evaluate(
    experiment: CoinFlipExperiment,
    loss_fn: Callable,
    *args: Any,
    batch_argnums: tuple[int, ...] | None = _UNSET,
    dataset: Any = _UNSET,
    collate_fn: Callable | None = _UNSET,
    batch_unpack: Callable | None = _UNSET,
    batch_size: int | None = None,
) -> AuditResult:
    """Score canaries and produce audit results in one call.

    When scoring config was provided to :func:`setup`, only ``loss_fn``
    and trained parameters are needed::

        audit = auditing.evaluate(experiment, loss_fn, trained_params)

    Any parameter passed here overrides the value stored at setup time.

    Args:
        experiment: The :class:`CoinFlipExperiment` from :func:`setup`.
        loss_fn: Per-example loss function, same as used with
            :func:`~opaque.clipped_grad`. Must be vmap-compatible.
        *args: Non-batched arguments to ``loss_fn`` (e.g., model parameters).
        batch_argnums: Which ``loss_fn`` args come from dataset batches.
            Falls back to the value from :func:`setup` if not provided.
        dataset: The full dataset. Falls back to the dataset from
            :func:`setup` if not provided.
        collate_fn: DataLoader collate function. Falls back to :func:`setup`.
        batch_unpack: Batch extraction function. Falls back to :func:`setup`.
        batch_size: Scoring batch size. Falls back to :func:`setup`.

    Returns:
        :class:`AuditResult` with ``epsilon_at()`` defaulting to the
        ``'one_run'`` method.
    """
    # Resolve parameters: explicit > stored at setup > error
    resolved_dataset = _resolve(dataset, experiment, "_dataset", "dataset")
    resolved_argnums = _resolve(
        batch_argnums, experiment, "_batch_argnums", "batch_argnums"
    )
    resolved_collate = _resolve_optional(collate_fn, experiment, "_collate_fn")
    resolved_unpack = _resolve_optional(batch_unpack, experiment, "_batch_unpack")
    resolved_batch_size = (
        batch_size
        if batch_size is not None
        else getattr(experiment, "_batch_size", 256)
    )

    scores = score(
        loss_fn,
        *args,
        batch_argnums=resolved_argnums,
        dataset=resolved_dataset,
        indices=experiment.canary_indices,
        collate_fn=resolved_collate,
        batch_unpack=resolved_unpack,
        batch_size=resolved_batch_size,
    )
    return experiment.audit(scores)


def _resolve(value: Any, experiment: CoinFlipExperiment, attr: str, name: str) -> Any:
    """Resolve a parameter: explicit value > stored on experiment > error."""
    if value is not _UNSET:
        return value
    stored = getattr(experiment, attr, None)
    if stored is not None:
        return stored
    raise TypeError(f"'{name}' must be provided either to setup() or evaluate()")


def _resolve_optional(
    value: Any, experiment: CoinFlipExperiment, attr: str
) -> Any | None:
    """Resolve an optional parameter: explicit value > stored > None."""
    if value is not _UNSET:
        return value
    return getattr(experiment, attr, None)
