"""Loss-based membership scoring for privacy auditing.

Computes per-example membership scores using the same ``torch.func.vmap``
and ``batch_argnums`` pattern as :func:`opaque.clipped_grad`. Higher scores
indicate higher likelihood of being a training member.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from opaque.api.auditing.attacks._helpers import (
    _bind_scores,
    _extract_batch_tensors,
    _iter_scoring_batches,
    _merge_args,
    _resolve_scoring_mode,
    _validate_batch_argnums,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.api.auditing._coin_flip import CanaryScores, CoinFlip

__all__ = ["loss_scores"]


def loss_scores(
    loss_fn: Callable,
    *args: Any,
    batch_argnums: tuple[int, ...],
    dataloader: Any | None = None,
    reference_scores: np.ndarray | CanaryScores | None = None,
    coin_flip: CoinFlip | None = None,
    dataset: Any | None = None,
    batch_size: int | None = None,
    collate_fn: Callable | None = None,
) -> np.ndarray | CanaryScores:
    """Compute membership scores as negative per-example loss.

    Higher score = lower loss = more likely a training member.
    Uses ``torch.func.vmap`` for per-example loss computation, following
    the same ``batch_argnums`` convention as :func:`~opaque.clipped_grad`.

    Two scoring modes:

    - **Verified** (``coin_flip=`` + ``dataset=``): the scorer builds its
      own loader over the partition's canaries and pairs every score with
      the dataset index of the example that produced it.  Returns
      :class:`~opaque.auditing.types.CanaryScores`, which
      :func:`~opaque.auditing.one_run` requires — the score-to-membership
      pairing is joined by identifier, so it cannot be misaligned by
      loader order.  ``batch_size`` and ``collate_fn`` configure the
      internal loader; ``collate_fn`` receives the raw canary examples
      and must return a batch for ``loss_fn`` without reordering examples
      within the batch.
    - **Legacy** (``dataloader=``): scores an arbitrary loader and
      returns a bare array with no identifiers.  Use for custom attack
      pipelines; to audit such scores, attest their identifiers
      explicitly via ``canary_scores(values, canary_indices=...)``.

    When ``reference_scores`` are provided, the returned scores are
    ``current - reference``, which equals ``loss(w0) - loss(wt)``
    (the loss reduction from initial to current model). This matches
    Algorithm 3 of Steinke et al. (2023).  In verified mode the reference
    must itself be a ``CanaryScores``; it is aligned by identifier before
    subtraction.

    Args:
        loss_fn: Per-example loss function, same as used with
            :func:`~opaque.clipped_grad`. Must be vmap-compatible.
        *args: Non-batched arguments to ``loss_fn`` (e.g., model parameters).
        batch_argnums: Indices of ``loss_fn`` positional arguments that come
            from dataset batches. Must be sorted, unique, non-negative.
        dataloader: An iterable of batches (typically a ``DataLoader``)
            for legacy scoring. Must yield tensors or tuples of tensors
            matching ``batch_argnums``. Mutually exclusive with
            ``coin_flip``/``dataset``.
        reference_scores: Baseline scores from an untrained model, shape
            ``(n,)``. When provided, returned scores are the loss reduction
            ``scores - reference_scores``. Typically obtained by calling
            ``loss_scores`` on the untrained model before training.
        coin_flip: The audit partition to score against (verified mode).
        dataset: The full dataset the partition was created from
            (verified mode); canaries are selected internally.
        batch_size: Batch size of the internal loader (verified mode
            only). Defaults to 32.
        collate_fn: Collation for the internal loader (verified mode
            only). Defaults to ``torch.utils.data.default_collate``.

    Returns:
        Membership scores, shape ``(n,)``: negated losses (higher = more
        likely member), optionally adjusted by reference scores.  A
        :class:`~opaque.auditing.types.CanaryScores` in verified mode,
        else a bare ``np.ndarray``.

    Raises:
        ValueError: If the scoring mode arguments are inconsistent, or a
            legacy ``dataloader`` shuffles (RandomSampler-family sampler).
        TypeError: If ``reference_scores`` verification does not match
            the scoring mode.

    Example (verified HuggingFace pattern)::

        def canary_collate(examples):
            batch = data_collator(examples)
            return (batch["input_ids"].to(device),)

        ref = auditing.loss_scores(
            loss_fn, initial_params,
            batch_argnums=(1,),
            coin_flip=cf, dataset=dataset,
            batch_size=32, collate_fn=canary_collate,
        )
        scores = auditing.loss_scores(
            loss_fn, trained_params,
            batch_argnums=(1,),
            coin_flip=cf, dataset=dataset,
            batch_size=32, collate_fn=canary_collate,
            reference_scores=ref,
        )
        estimate = auditing.one_run(scores, coin_flip=cf)

    Example (legacy PyTorch ``(x, y)`` pattern)::

        loader = DataLoader(dataset, batch_size=256)
        scores = auditing.loss_scores(
            loss_fn, params,
            batch_argnums=(1, 2), dataloader=loader,
        )
    """
    import torch

    _validate_batch_argnums(batch_argnums, len(args))
    loader, verified = _resolve_scoring_mode(
        dataloader=dataloader,
        dataset=dataset,
        coin_flip=coin_flip,
        batch_size=batch_size,
        collate_fn=collate_fn,
        reference_scores=reference_scores,
    )

    # Build in_dims for vmap: None for non-batch args, 0 for batch args
    n_args = len(args) + len(batch_argnums)
    in_dims = tuple(0 if i in batch_argnums else None for i in range(n_args))
    per_example_fn = torch.func.vmap(loss_fn, in_dims=in_dims)

    all_positions: list[int] = []
    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for positions, batch in _iter_scoring_batches(loader, verified):
            if positions is not None:
                all_positions.extend(positions)
            batch_tensors = _extract_batch_tensors(batch, batch_argnums)

            full_args = _merge_args(args, batch_tensors, batch_argnums)

            losses = per_example_fn(*full_args)
            all_scores.append(-losses.detach().cpu().numpy())

    scores = np.concatenate(all_scores) if all_scores else np.empty(0)

    return _bind_scores(
        scores,
        all_positions if verified else None,
        coin_flip=coin_flip,
        reference_scores=reference_scores,
    )
