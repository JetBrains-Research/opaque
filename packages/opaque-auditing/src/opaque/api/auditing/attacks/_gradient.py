"""Gradient-norm membership scoring for privacy auditing (white-box).

Computes per-example membership scores using the squared L2 norm of the
gradient of the loss with respect to model parameters.  Members have
small gradient norms (the model has converged on them), producing higher
scores.  This is the first-order approximation of the Bayes-optimal
membership test statistic (Sablayrolles et al. 2019).

Uses the same ``batch_argnums`` convention as :func:`opaque.clipped_grad`
and :func:`~opaque.api.auditing.attacks._loss.loss_scores`.
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

__all__ = ["gradient_scores"]


def gradient_scores(
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
    """Compute membership scores as negative squared per-example gradient norm.

    Higher score = smaller gradient norm = more likely a training member.
    The model parameters must be the first positional argument to
    ``loss_fn`` (position 0), matching the default ``argnums=0`` of
    :func:`~opaque.clipped_grad`.

    Two scoring modes, matching
    :func:`~opaque.api.auditing.attacks._loss.loss_scores`:

    - **Verified** (``coin_flip=`` + ``dataset=``): scores the
      partition's canaries over an internal identifier-carrying loader
      and returns :class:`~opaque.auditing.types.CanaryScores`, which
      :func:`~opaque.auditing.one_run` requires.  Identifiers are attached
      per batch before collation, so ``collate_fn`` must emit one row per
      example in the order it received them; reordering within a batch
      misaligns the pairing undetectably.
    - **Legacy** (``dataloader=``): scores an arbitrary loader and
      returns a bare array with no identifiers.

    Processes one sample at a time to keep peak GPU memory at 2× model
    size regardless of model or batch size.  ``torch.func.grad`` is a
    functional transform that operates independently of the standard
    autograd context, so the loop runs inside ``torch.no_grad()`` for
    efficiency (same as :func:`~opaque.api.auditing.attacks._loss.loss_scores`).

    When ``reference_scores`` are provided, the returned scores are
    ``current - reference``, which equals
    ``||∇loss(θ₀)||² - ||∇loss(θₜ)||²`` (positive when the current model
    has smaller gradients, i.e. is more converged).  This matches the calibration pattern
    of :func:`~opaque.api.auditing.attacks._loss.loss_scores`.  In
    verified mode the reference must itself be a ``CanaryScores``; it is
    aligned by identifier before subtraction.

    Args:
        loss_fn: Per-example loss function whose first argument is the
            model parameters (a tensor or PyTree).  Must return a scalar.
        *args: Non-batched arguments to ``loss_fn`` (e.g., model parameters).
            The first element is differentiated.
        batch_argnums: Indices of ``loss_fn`` positional arguments that come
            from dataset batches.  Must not include 0 (reserved for params).
            Must be sorted, unique, non-negative.
        dataloader: An iterable of batches (typically a ``DataLoader``)
            for legacy scoring.  Mutually exclusive with
            ``coin_flip``/``dataset``.
        reference_scores: Baseline scores from an untrained model, shape
            ``(n,)``.  When provided, returned scores are
            ``scores - reference_scores``.  Typically obtained by calling
            ``gradient_scores`` on the untrained model before training.
        coin_flip: The audit partition to score against (verified mode).
        dataset: The full dataset the partition was created from
            (verified mode); canaries are selected internally.
        batch_size: Batch size of the internal loader (verified mode
            only). Defaults to 32.
        collate_fn: Collation for the internal loader (verified mode
            only). Defaults to ``torch.utils.data.default_collate``.

    Returns:
        Membership scores, shape ``(n,)``: negated squared gradient norms
        (higher = more likely member), optionally adjusted by reference
        scores.  A :class:`~opaque.auditing.types.CanaryScores` in
        verified mode, else a bare ``np.ndarray``.

    Raises:
        ValueError: If ``0 in batch_argnums`` (params must be at
            position 0), the scoring mode arguments are inconsistent, or
            a legacy ``dataloader`` shuffles (RandomSampler-family
            sampler).
        TypeError: If ``reference_scores`` verification does not match
            the scoring mode.

    Example::

        ref = auditing.gradient_scores(
            loss_fn, initial_params,
            batch_argnums=(1,),
            coin_flip=cf, dataset=dataset, collate_fn=canary_collate,
        )
        scores = auditing.gradient_scores(
            loss_fn, trained_params,
            batch_argnums=(1,),
            coin_flip=cf, dataset=dataset, collate_fn=canary_collate,
            reference_scores=ref,
        )
    """
    import torch

    from opaque.api.engine.pytree import global_norm

    _validate_batch_argnums(batch_argnums, len(args))

    if 0 in batch_argnums:
        raise ValueError(
            "gradient_scores differentiates w.r.t. the first argument "
            "(position 0), which must not be in batch_argnums. "
            f"Got batch_argnums={batch_argnums}."
        )

    loader, verified = _resolve_scoring_mode(
        dataloader=dataloader,
        dataset=dataset,
        coin_flip=coin_flip,
        batch_size=batch_size,
        collate_fn=collate_fn,
        reference_scores=reference_scores,
    )

    grad_fn = torch.func.grad(loss_fn)

    all_positions: list[int] = []
    all_scores: list[float] = []
    with torch.no_grad():
        for positions, batch in _iter_scoring_batches(loader, verified):
            if positions is not None:
                all_positions.extend(positions)
            batch_tensors = _extract_batch_tensors(batch, batch_argnums)
            n_in_batch = batch_tensors[0].shape[0]

            for j in range(n_in_batch):
                single_tensors = tuple(t[j] for t in batch_tensors)
                full_args = _merge_args(args, single_tensors, batch_argnums)

                grad_pytree = grad_fn(*full_args)
                norm = global_norm(grad_pytree)
                all_scores.append(-(norm**2).item())
                del grad_pytree

    scores = np.array(all_scores)

    return _bind_scores(
        scores,
        all_positions if verified else None,
        coin_flip=coin_flip,
        reference_scores=reference_scores,
    )
