"""Empirical privacy auditing for differential privacy.

One-run auditing with HuggingFace integration (Steinke et al. 2023).

Recommended pattern — configure scoring at setup, evaluate in one line::

    import opaque.auditing as auditing
    from opaque.random import key

    # Setup: specify dataset + scoring config once
    audit_state = auditing.setup(
        dataset, num_canaries=1000, key=key(42),
        batch_argnums=(1,),
        collate_fn=data_collator,
        batch_unpack=lambda b: (b["input_ids"].to(device),),
    )
    train_data = dataset.select(audit_state.train_indices)

    # ... DP-SGD training loop ...

    # Evaluate: just pass loss_fn and trained params
    audit = auditing.evaluate(per_example_loss_fn, trained_params, state=audit_state)
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

from opaque.auditing.audit import AuditResult, AuditState, CoinFlipExperiment
from opaque.auditing.scoring import score
from opaque.random import RngKey, fold_in

__all__ = [
    "AuditResult",
    "AuditState",
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
) -> AuditState:
    """Set up a one-run privacy audit experiment.

    Randomly selects ``num_canaries`` examples from the dataset and flips
    a fair coin for each to decide inclusion/exclusion. Returns an opaque
    :class:`AuditState` that bundles the coin-flip partition with the
    dataset and scoring configuration.

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
        An :class:`AuditState` containing the canary assignment and
        scoring configuration.

    Example (recommended — configure scoring here)::

        audit_state = auditing.setup(
            dataset, num_canaries=1000, key=key(42),
            batch_argnums=(1,),
            collate_fn=data_collator,
            batch_unpack=lambda b: (b["input_ids"].to(device),),
        )
        train_data = dataset.select(audit_state.train_indices)

    Example (minimal — configure scoring at evaluate time)::

        audit_state = auditing.setup(dataset, num_canaries=1000, key=key(42))
        train_data = dataset.select(audit_state.train_indices)
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

    return AuditState(
        experiment,
        dataset=dataset,
        batch_argnums=batch_argnums,
        collate_fn=collate_fn,
        batch_unpack=batch_unpack,
        batch_size=batch_size,
    )


def evaluate(
    loss_fn: Callable,
    *args: Any,
    state: AuditState,
    batch_argnums: tuple[int, ...] | None = _UNSET,
    dataset: Any = _UNSET,
    collate_fn: Callable | None = _UNSET,
    batch_unpack: Callable | None = _UNSET,
    batch_size: int | None = None,
) -> AuditResult:
    """Score canaries and produce audit results in one call.

    When scoring config was provided to :func:`setup`, only ``loss_fn``
    and trained parameters are needed::

        audit = auditing.evaluate(loss_fn, trained_params, state=audit_state)

    Any parameter passed here overrides the value stored at setup time.

    Args:
        loss_fn: Per-example loss function, same as used with
            :func:`~opaque.clipped_grad`. Must be vmap-compatible.
        *args: Non-batched arguments to ``loss_fn`` (e.g., model parameters).
        state: The :class:`AuditState` from :func:`setup`.
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
    # Resolve parameters: explicit > stored in state > error
    resolved_dataset = _resolve(dataset, state, "_dataset", "dataset")
    resolved_argnums = _resolve(batch_argnums, state, "_batch_argnums", "batch_argnums")
    resolved_collate = _resolve_optional(collate_fn, state, "_collate_fn")
    resolved_unpack = _resolve_optional(batch_unpack, state, "_batch_unpack")
    resolved_batch_size = (
        batch_size if batch_size is not None else getattr(state, "_batch_size", 256)
    )

    experiment = state._experiment
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


def _resolve(value: Any, state: AuditState, attr: str, name: str) -> Any:
    """Resolve a parameter: explicit value > stored on state > error."""
    if value is not _UNSET:
        return value
    stored = getattr(state, attr, None)
    if stored is not None:
        return stored
    raise TypeError(f"'{name}' must be provided either to setup() or evaluate()")


def _resolve_optional(value: Any, state: AuditState, attr: str) -> Any | None:
    """Resolve an optional parameter: explicit value > stored > None."""
    if value is not _UNSET:
        return value
    return getattr(state, attr, None)
