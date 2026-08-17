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
    _extract_batch_tensors,
    _merge_args,
    _validate_batch_argnums,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["gradient_scores"]


def gradient_scores(
    loss_fn: Callable,
    *args: Any,
    batch_argnums: tuple[int, ...],
    dataloader: Any,
    reference_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Compute membership scores as negative squared per-example gradient norm.

    Higher score = smaller gradient norm = more likely a training member.
    The model parameters must be the first positional argument to
    ``loss_fn`` (position 0), matching the default ``argnums=0`` of
    :func:`~opaque.clipped_grad`.

    Processes one sample at a time to keep peak GPU memory at 2× model
    size regardless of model or batch size.  ``torch.func.grad`` is a
    functional transform that operates independently of the standard
    autograd context, so the loop runs inside ``torch.no_grad()`` for
    efficiency (same as :func:`~opaque.api.auditing.attacks._loss.loss_scores`).

    When ``reference_scores`` are provided, the returned scores are
    ``current - reference``, which equals
    ``||∇loss(θ₀)||² - ||∇loss(θₜ)||²`` (positive when the current model
    has smaller gradients, i.e. is more converged).  This matches the calibration pattern
    of :func:`~opaque.api.auditing.attacks._loss.loss_scores`.

    Args:
        loss_fn: Per-example loss function whose first argument is the
            model parameters (a tensor or PyTree).  Must return a scalar.
        *args: Non-batched arguments to ``loss_fn`` (e.g., model parameters).
            The first element is differentiated.
        batch_argnums: Indices of ``loss_fn`` positional arguments that come
            from dataset batches.  Must not include 0 (reserved for params).
            Must be sorted, unique, non-negative.
        dataloader: An iterable of batches (typically a ``DataLoader``).
        reference_scores: Baseline scores from an untrained model, shape
            ``(n,)``.  When provided, returned scores are
            ``scores - reference_scores``.  Typically obtained by calling
            ``gradient_scores`` on the untrained model before training.

    Returns:
        Array of membership scores, shape ``(n,)``.  Scores are negated
        squared gradient norms (higher = more likely member), optionally
        adjusted by reference scores.

    Raises:
        ValueError: If ``0 in batch_argnums`` (params must be at position 0).

    Example::

        ref = auditing.gradient_scores(
            loss_fn, initial_params,
            batch_argnums=(1,), dataloader=canary_loader,
        )
        scores = auditing.gradient_scores(
            loss_fn, trained_params,
            batch_argnums=(1,), dataloader=canary_loader,
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

    grad_fn = torch.func.grad(loss_fn)

    all_scores: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch_tensors = _extract_batch_tensors(batch, batch_argnums)
            batch_size = batch_tensors[0].shape[0]

            for j in range(batch_size):
                single_tensors = tuple(t[j] for t in batch_tensors)
                full_args = _merge_args(args, single_tensors, batch_argnums)

                grad_pytree = grad_fn(*full_args)
                norm = global_norm(grad_pytree)
                all_scores.append(-norm.pow(2).item())
                del grad_pytree

    scores = np.array(all_scores)

    if reference_scores is not None:
        reference_scores = np.asarray(reference_scores)
        if reference_scores.shape != scores.shape:
            raise ValueError(
                f"reference_scores shape {reference_scores.shape} does not match "
                f"scores shape {scores.shape}"
            )
        scores = scores - reference_scores

    return scores
