"""Shared helpers for membership inference scoring functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from opaque.api.auditing._coin_flip import CanaryScores

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from opaque.api.auditing._coin_flip import CoinFlip


def _validate_batch_argnums(batch_argnums: tuple[int, ...], n_non_batch: int) -> None:
    """Validate batch_argnums constraints."""
    if not batch_argnums:
        raise ValueError(f"batch_argnums must be non-empty, got {batch_argnums}")
    if any(a < 0 for a in batch_argnums):
        raise ValueError(f"batch_argnums must be non-negative, got {batch_argnums}")
    if len(set(batch_argnums)) != len(batch_argnums):
        raise ValueError(f"batch_argnums must be unique, got {batch_argnums}")
    if tuple(sorted(batch_argnums)) != batch_argnums:
        raise ValueError(f"batch_argnums must be sorted, got {batch_argnums}")
    n_total = n_non_batch + len(batch_argnums)
    if max(batch_argnums) >= n_total:
        raise ValueError(
            f"batch_argnums index {max(batch_argnums)} out of range for "
            f"{n_total} total arguments ({n_non_batch} non-batched + "
            f"{len(batch_argnums)} batched), got {batch_argnums}"
        )


def _merge_args(
    args: tuple[Any, ...],
    batch_tensors: tuple[Any, ...],
    batch_argnums: tuple[int, ...],
) -> list[Any]:
    """Merge non-batched args and batch tensors into a single arg list."""
    n_total = len(args) + len(batch_argnums)
    result: list[Any] = [None] * n_total

    for pos, tensor in zip(batch_argnums, batch_tensors, strict=False):
        result[pos] = tensor

    arg_iter = iter(args)
    for i in range(n_total):
        if i not in batch_argnums:
            result[i] = next(arg_iter)

    return result


def _extract_batch_tensors(
    batch: Any,
    batch_argnums: tuple[int, ...],
) -> tuple[Any, ...]:
    """Extract tensors from a dataloader batch.

    Returns a tuple of tensors matching the length of ``batch_argnums``.
    """
    if isinstance(batch, (list, tuple)):
        return tuple(batch[i] for i in range(len(batch_argnums)))
    return (batch,)


def _check_unshuffled(dataloader: Any) -> None:
    """Raise when a torch ``DataLoader`` would shuffle the scoring order.

    Membership scores are paired positionally with the coin-flip labels, so
    a shuffled loader silently attaches scores to the wrong labels and the
    audit reports no leakage that was never measured.  Detects the torch
    shuffling samplers on both the ``sampler`` and ``batch_sampler.sampler``
    seats.  Applies to legacy (bare ``dataloader=``) scoring only; verified
    scoring joins scores to labels by canary identifier and never depends
    on iteration order.
    """
    import torch.utils.data as tud

    shuffling = (tud.RandomSampler, tud.SubsetRandomSampler)
    sampler = getattr(dataloader, "sampler", None)
    inner = getattr(getattr(dataloader, "batch_sampler", None), "sampler", None)
    if isinstance(sampler, shuffling) or isinstance(inner, shuffling):
        raise ValueError(
            "dataloader is shuffled (RandomSampler/SubsetRandomSampler); "
            "bare scores are paired positionally, so the scoring order "
            "must be reproducible. Construct the DataLoader with "
            "shuffle=False, or switch to verified scoring (coin_flip= + "
            "dataset=), which pairs scores by canary identifier and does "
            "not depend on order"
        )


class _IndexedCanaries:
    """Map-style dataset yielding ``(position, example)`` canary pairs."""

    def __init__(self, dataset: Any, canary_indices: np.ndarray) -> None:
        self._dataset = dataset
        self._canary_indices = canary_indices

    def __len__(self) -> int:
        return len(self._canary_indices)

    def __getitem__(self, position: int) -> tuple[int, Any]:
        return position, self._dataset[int(self._canary_indices[position])]


def _canary_loader(
    dataset: Any,
    coin_flip: CoinFlip,
    batch_size: int,
    collate_fn: Callable | None,
) -> Any:
    """Build the internal DataLoader for verified canary scoring.

    Each batch arrives as ``(positions, collated_examples)``: the canary
    positions ride alongside the examples through collation, so every
    score is paired with the identifier of the example that produced it —
    the pairing never relies on the loader's iteration order.
    """
    import torch.utils.data as tud

    example_collate = tud.default_collate if collate_fn is None else collate_fn

    def indexed_collate(batch: list[tuple[int, Any]]) -> tuple[list[int], Any]:
        positions = [position for position, _ in batch]
        examples = [example for _, example in batch]
        return positions, example_collate(examples)

    return tud.DataLoader(
        _IndexedCanaries(dataset, coin_flip.canary_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=indexed_collate,
    )


def _resolve_scoring_mode(
    *,
    dataloader: Any,
    dataset: Any,
    coin_flip: CoinFlip | None,
    batch_size: int | None,
    collate_fn: Callable | None,
    reference_scores: np.ndarray | CanaryScores | None,
) -> tuple[Any, bool]:
    """Validate scoring arguments and return ``(loader, verified)``.

    Verified mode (``coin_flip=`` + ``dataset=``) builds an internal
    identifier-carrying loader; legacy mode passes ``dataloader`` through
    unchanged and keeps the bare-array contract.
    """
    if coin_flip is not None or dataset is not None:
        if coin_flip is None or dataset is None:
            raise ValueError("verified scoring requires both coin_flip= and dataset=")
        if dataloader is not None:
            raise ValueError(
                "pass either dataloader= or (coin_flip=, dataset=), not "
                "both; verified scoring builds its own loader over the "
                "canaries"
            )
        if reference_scores is not None and not isinstance(
            reference_scores, CanaryScores
        ):
            raise TypeError(
                "verified scoring requires reference_scores with canary "
                "identifiers; compute the reference with coin_flip= and "
                "dataset= as well"
            )
        loader = _canary_loader(
            dataset,
            coin_flip,
            32 if batch_size is None else batch_size,
            collate_fn,
        )
        return loader, True

    if dataloader is None:
        raise ValueError("either dataloader= or (coin_flip=, dataset=) is required")
    if batch_size is not None or collate_fn is not None:
        raise ValueError(
            "batch_size= and collate_fn= apply to verified scoring "
            "(coin_flip= + dataset=); configure them on your own "
            "dataloader otherwise"
        )
    if isinstance(reference_scores, CanaryScores):
        raise TypeError(
            "reference_scores carries canary identifiers but scoring uses "
            "a bare dataloader; score with coin_flip= and dataset= so both "
            "passes are verified (or subtract reference_scores.scores "
            "yourself)"
        )
    _check_unshuffled(dataloader)
    return dataloader, False


def _iter_scoring_batches(
    loader: Any, verified: bool
) -> Iterator[tuple[list[int] | None, Any]]:
    """Yield ``(positions, batch)``; positions is None for legacy loaders."""
    if verified:
        for positions, batch in loader:
            yield [int(position) for position in positions], batch
    else:
        for batch in loader:
            yield None, batch


def _aligned_reference(ids: np.ndarray, reference_scores: CanaryScores) -> np.ndarray:
    """Return reference score values aligned to ``ids`` by identifier."""
    ref_ids = reference_scores.canary_indices
    if ref_ids.size == 0:
        if ids.size:
            raise ValueError(
                "reference_scores are empty but scores are not; compute "
                "the reference over the same coin_flip and dataset"
            )
        return np.empty(0, dtype=float)
    sorter = np.argsort(ref_ids, kind="stable")
    pos = np.searchsorted(ref_ids[sorter], ids)
    pos = np.minimum(pos, ref_ids.size - 1)
    if not np.all(ref_ids[sorter][pos] == ids):
        raise ValueError(
            "reference_scores do not cover the scored canaries; compute "
            "the reference over the same coin_flip and dataset"
        )
    return reference_scores.scores[sorter[pos]]


def _bind_scores(
    scores: np.ndarray,
    positions: list[int] | None,
    *,
    coin_flip: CoinFlip | None,
    reference_scores: np.ndarray | CanaryScores | None,
) -> np.ndarray | CanaryScores:
    """Apply reference calibration and, when verified, attach identifiers."""
    if coin_flip is None:
        if reference_scores is not None:
            reference_scores = np.asarray(reference_scores)
            if reference_scores.shape != scores.shape:
                raise ValueError(
                    f"reference_scores shape {reference_scores.shape} does "
                    f"not match scores shape {scores.shape}"
                )
            scores = scores - reference_scores
        return scores

    ids = coin_flip.canary_indices[np.asarray(positions, dtype=int)]
    if reference_scores is not None:
        scores = scores - _aligned_reference(ids, reference_scores)
    return CanaryScores(scores, canary_indices=ids)
