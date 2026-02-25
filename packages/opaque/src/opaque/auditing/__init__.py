"""Empirical privacy auditing for differential privacy.

One-run auditing with HuggingFace integration (Steinke et al. 2023)::

    import opaque.auditing as auditing
    from opaque.random import key

    # 1. Setup: designate canaries and flip coins
    experiment = auditing.setup(dataset, num_canaries=1000, key=key(42))

    # 2. Train on subset (excludes held-out canaries)
    train_data = dataset.select(experiment.train_indices(len(dataset)))
    # ... DP-SGD training loop ...

    # 3. Score canaries and compute epsilon
    audit = auditing.evaluate(
        experiment,
        per_example_loss_fn,
        trainable_params,
        batch_argnums=(1,),
        dataset=dataset,
        collate_fn=data_collator,
        batch_unpack=lambda b: (b["input_ids"].to(device),),
    )
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
from opaque.auditing.scoring import score
from opaque.random import RngKey, fold_in

__all__ = [
    "AuditResult",
    "CoinFlipExperiment",
    "evaluate",
    "score",
    "setup",
]


def setup(
    dataset: Any,
    *,
    num_canaries: int,
    key: RngKey,
) -> CoinFlipExperiment:
    """Set up a one-run privacy audit experiment.

    Randomly selects ``num_canaries`` examples from the dataset and flips
    a fair coin for each to decide inclusion/exclusion. This is the entry
    point for one-run auditing (Steinke et al. 2023).

    Args:
        dataset: Any dataset with ``len()`` (HuggingFace or PyTorch).
        num_canaries: Number of canary examples to designate.
        key: RNG key for reproducible canary selection and coin flips.

    Returns:
        A :class:`CoinFlipExperiment` managing the canary assignment.

    Example::

        import opaque.auditing as auditing
        from opaque.random import key

        experiment = auditing.setup(dataset, num_canaries=1000, key=key(42))

        # HuggingFace datasets:
        train_data = dataset.select(experiment.train_indices(len(dataset)))

        # PyTorch datasets:
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
    return CoinFlipExperiment(canary_indices, key=coin_key)


def evaluate(
    experiment: CoinFlipExperiment,
    loss_fn: Callable,
    *args: Any,
    batch_argnums: tuple[int, ...],
    dataset: Any,
    collate_fn: Callable | None = None,
    batch_unpack: Callable | None = None,
    batch_size: int = 256,
) -> AuditResult:
    """Score canaries and produce audit results in one call.

    Computes per-example negative loss as membership score (via
    ``torch.func.vmap``), then splits scores by coin flip. Uses the
    same ``batch_argnums`` convention as :func:`~opaque.clipped_grad`.

    Args:
        experiment: The :class:`CoinFlipExperiment` from :func:`setup`.
        loss_fn: Per-example loss function, same as used with
            :func:`~opaque.clipped_grad`. Must be vmap-compatible.
        *args: Non-batched arguments to ``loss_fn`` (e.g., model parameters).
        batch_argnums: Indices of ``loss_fn`` positional arguments that come
            from dataset batches (same convention as ``clipped_grad``).
        dataset: The full dataset (same one passed to :func:`setup`).
        collate_fn: Collate function for the DataLoader (e.g.,
            ``DataCollatorForLanguageModeling``). Default: PyTorch default.
        batch_unpack: Callable mapping a DataLoader batch to a tuple of
            tensors, one per ``batch_argnums`` position. For example,
            ``lambda b: (b["input_ids"].to(device),)`` for HF batches.
            If ``None``, batches are unpacked as tuples/lists.
        batch_size: Batch size for scoring. Default: 256.

    Returns:
        :class:`AuditResult` with ``epsilon_at()`` defaulting to the
        ``'one_run'`` method.

    Example (HuggingFace pattern)::

        audit = auditing.evaluate(
            experiment,
            per_example_loss_fn,
            trainable_params,
            batch_argnums=(1,),
            dataset=full_dataset,
            collate_fn=data_collator,
            batch_unpack=lambda b: (b["input_ids"].to(device),),
        )

    Example (PyTorch (x, y) pattern)::

        audit = auditing.evaluate(
            experiment, loss_fn, params,
            batch_argnums=(1, 2),
            dataset=dataset,
        )
    """
    scores = score(
        loss_fn,
        *args,
        batch_argnums=batch_argnums,
        dataset=dataset,
        indices=experiment.canary_indices,
        collate_fn=collate_fn,
        batch_unpack=batch_unpack,
        batch_size=batch_size,
    )
    return experiment.audit(scores)
