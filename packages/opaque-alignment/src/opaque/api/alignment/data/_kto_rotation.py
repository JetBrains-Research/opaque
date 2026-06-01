"""KTO completion rotation — KL-estimation dataset transform.

:func:`rotate_kto_completions` is a pure-Python + ``datasets`` port of the
``_get_kl_dataset`` helper from ``trl/kto_trainer.py`` (TRL ≥ 0.18).

**TRL reference:** ``KTOTrainer._get_kl_dataset`` (TRL source
``trl/trainer/kto_trainer.py``, circa TRL 0.18) uses::

    def _get_kl_dataset(self, batch: dict[str, list]) -> dict[str, list]:
        batch["KL_completion"] = batch["completion"][1:] + batch["completion"][:1]
        ...
        return batch

called via ``dataset.map(_get_kl_dataset, batched=True,
batch_size=self.args.per_device_train_batch_size)``.

Within each consecutive block of ``batch_size`` rows the ``completion``
column is **left-rotated by 1** — i.e. the sequence
``[c0, c1, c2, ..., c_{n-1}]`` becomes ``[c1, c2, ..., c_{n-1}, c0]``.
This produces "mismatched" (completion, prompt) pairs for estimating the
KL divergence term in the KTO objective (Equation 8 of Ethayarajh et al.
2023, arXiv:2310.01854).

**Seed / shuffle scheme:** before rotation, the dataset is shuffled with
``dataset.shuffle(seed=seed)`` (``datasets.Dataset.shuffle``).  This
reorders rows globally before batching so that within-batch completion
diversity is higher — reducing pathological cases where all completions in
a batch are identical.  The post-shuffle row order is deterministic for a
given ``seed``; re-calling with the same ``seed`` always produces the same
``KL_completion`` column.

**Non-identity assertion:** if every row in a batch has the *same*
completion string, the rotation is degenerate (identity for that batch).
When ALL completions are identical across the *entire* dataset (the
"all-equal" adversarial case) a ``ValueError`` is raised, because no
rotation of any batch can produce a non-identity mapping.  When only a
subset of batches are degenerate (all-equal within batch but not globally)
the assertion is skipped for those batches — the global assertion fires
only when the whole dataset is degenerate.

This module has **no torch dependency** and works on plain
``datasets.Dataset`` objects containing any column types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

__all__ = ["rotate_kto_completions"]


def rotate_kto_completions(
    dataset: "Dataset",
    batch_size: int,
    seed: int = 0,
) -> "Dataset":
    """Produce the KL-estimation "mismatched" completions for KTO.

    Within each consecutive block of *batch_size* rows the ``completion``
    column is **left-rotated by 1**: ``[c0, c1, c2]`` → ``[c1, c2, c0]``.
    The rotated completions are added as a new ``KL_completion`` column.

    **Shuffle scheme:** ``dataset.shuffle(seed=seed)`` is applied before
    rotation so that each batch contains a diverse mix of completions.  The
    same ``seed`` always yields the same output (deterministic).

    **Non-identity assertion:** raises :class:`ValueError` if every
    completion in the *entire* dataset is identical, because no rotation
    can ever produce distinct KL completions in that case.  Batches that
    are locally degenerate (all-equal *within* the batch) are silently
    skipped — only a fully-degenerate dataset triggers the error.

    Args:
        dataset: A ``datasets.Dataset`` with at least a ``"completion"``
            column (any type — strings, token-ID lists, etc.).
        batch_size: Number of rows per rotation batch.  Mirrors
            ``per_device_train_batch_size`` in TRL's ``KTOTrainer``.
        seed: Integer seed for ``dataset.shuffle``.  Controls row order
            before batching; same seed → identical ``KL_completion`` output.

    Returns:
        A **new** ``datasets.Dataset`` with all original columns preserved
        plus a ``KL_completion`` column containing the rotated completions.
        Row count is unchanged.

    Raises:
        ValueError: If every completion in *dataset* is identical (the
            rotation would be entirely degenerate).

    Examples:
        Basic rotation with batch_size=4 (seed=0 passes through shuffle)::

            >>> from datasets import Dataset
            >>> ds = Dataset.from_dict({
            ...     "prompt": ["p0", "p1", "p2", "p3"],
            ...     "completion": ["c0", "c1", "c2", "c3"],
            ... })
            >>> out = rotate_kto_completions(ds, batch_size=4, seed=0)
            >>> out["KL_completion"]  # left-rotation of shuffled completions
            [...]
    """
    # ------------------------------------------------------------------ #
    # 1. Global degenerate-dataset guard (before shuffle, on original data)
    # ------------------------------------------------------------------ #
    completions: list = dataset["completion"]
    if len(completions) > 1 and len(set(_hashable(c) for c in completions)) == 1:
        raise ValueError(
            "rotate_kto_completions: all completions in the dataset are identical "
            f"({completions[0]!r}). No rotation can produce non-identity KL "
            "completions. Provide a dataset with at least two distinct completions."
        )

    # ------------------------------------------------------------------ #
    # 2. Shuffle for KL diversity
    # ------------------------------------------------------------------ #
    shuffled: "Dataset" = dataset.shuffle(seed=seed)

    # ------------------------------------------------------------------ #
    # 3. Batched left-rotation via dataset.map
    # ------------------------------------------------------------------ #
    def _rotate_batch(batch: dict[str, list]) -> dict[str, list]:
        """Left-rotate the completion column by 1 within the batch."""
        comps = batch["completion"]
        n = len(comps)
        if n == 0:
            rotated: list = []
        else:
            # Left-rotate by 1: [c0, c1, ..., c_{n-1}] → [c1, ..., c_{n-1}, c0]
            rotated = comps[1:] + comps[:1]
        return {"KL_completion": rotated}

    result: "Dataset" = shuffled.map(
        _rotate_batch,
        batched=True,
        batch_size=batch_size,
    )

    # ------------------------------------------------------------------ #
    # 4. Non-identity assertion over the full output
    # ------------------------------------------------------------------ #
    kl_completions: list = result["KL_completion"]
    original_completions: list = result["completion"]

    # Check if at least one row has a non-matching KL completion.
    any_distinct = any(
        _hashable(orig) != _hashable(kl)
        for orig, kl in zip(original_completions, kl_completions)
    )
    if not any_distinct and len(original_completions) > 1:
        # This branch is only reachable if the shuffle + rotation still left
        # every row with the same completion as before — which can only happen
        # when every completion in the shuffled dataset is identical.  The
        # global guard above covers the deterministic case; this is a runtime
        # safety net for edge cases (e.g. seed-driven degenerate shuffle).
        raise ValueError(
            "rotate_kto_completions: rotation produced an identity mapping — "
            "every KL_completion equals its corresponding completion after "
            f"shuffle(seed={seed}). This indicates all completions are identical "
            "in the shuffled order. Provide distinct completions or a different seed."
        )

    return result


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _hashable(value: object) -> object:
    """Return a hashable representation of *value* for set membership tests.

    Converts lists and other unhashable sequences to tuples recursively so
    they can be added to a Python :class:`set`.
    """
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value
