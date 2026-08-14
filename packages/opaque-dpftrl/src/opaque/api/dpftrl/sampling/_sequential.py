"""Sequential batch sampler for fixed-order training.

This sampler iterates through a dataset sequentially in fixed-size
batches, dropping the last incomplete batch.  It is used by the BLT
(Buffered Linear Toeplitz) mechanism, which requires a deterministic
batch order with fixed ``min_sep`` between participations.

Unlike Poisson or BnB sampling there is no randomness — the dataset
should be pre-shuffled once before constructing the sampler.
"""

from collections.abc import Iterator, Mapping
from typing import Any

from opaque.sampling import Sampler


class SequentialBatchSampler(Sampler[list[int]]):
    """Deterministic fixed-size sequential batching.

    Yields contiguous, non-overlapping chunks of ``batch_size`` indices
    ``[0, ..., B-1], [B, ..., 2B-1], ...``.  The last chunk is dropped
    when smaller than ``batch_size``.

    This sampler has no RNG key — it is fully deterministic.
    Call ``dataset.shuffle(seed=...)`` once beforehand to randomise
    which examples land in which batch.

    Args:
        data_source: Dataset to sample from (any object with ``__len__``).
        batch_size: Exact number of examples per batch (must be ≥ 1).

    Example:
        >>> sampler = SequentialBatchSampler(dataset, batch_size=256)
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(self, data_source: object, batch_size: int):
        super().__init__()

        if len(data_source) == 0:
            raise ValueError("data_source must not be empty")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.data_source = data_source
        self._num_samples = len(data_source)
        self._batch_size = batch_size
        self._num_batches = self._num_samples // self._batch_size
        self._consumed = 0

    def __iter__(self) -> Iterator[list[int]]:
        """Yield fixed-size batches of contiguous indices.

        The last incomplete batch (if any) is dropped.  Iteration
        resumes from ``self._consumed`` so a loaded sampler continues
        at its saved cursor.
        """
        for i in range(self._consumed, self._num_batches):
            start = i * self._batch_size
            # Increment before yield so a snapshot taken mid-iter
            # reports the count of batches actually emitted so far.
            self._consumed = i + 1
            yield list(range(start, start + self._batch_size))

    def __len__(self) -> int:
        """Complete batches remaining (``_num_batches - consumed``).

        After a partial run, reflects what ``__iter__`` will yield —
        so ``len(DataLoader(...))`` matches the resumed iteration count.
        """
        return self._num_batches - self._consumed

    @property
    def consumed(self) -> int:
        """Number of batches yielded so far (resume cursor)."""
        return self._consumed

    @property
    def expected_batch_size(self) -> float:
        """Batch size (exact, not statistical)."""
        return float(self._batch_size)


def _state_dict_sequential(s: SequentialBatchSampler) -> dict[str, Any]:
    """Serialise ``SequentialBatchSampler`` state.

    Iteration is fully deterministic with no RNG and no Markov state,
    so the cursor alone fixes the resume point.  ``num_samples`` is
    persisted so the loader can validate the template dataset length —
    a mismatched length would change ``_num_batches`` and thereby what
    indices each cursor position maps to.
    """
    return {
        "consumed": int(s._consumed),
        "num_samples": int(s._num_samples),
        "batch_size": int(s._batch_size),
    }


def _from_state_dict_sequential(
    template: SequentialBatchSampler, sd: Mapping[str, Any]
) -> SequentialBatchSampler:
    """Rebuild ``SequentialBatchSampler`` at the saved cursor.

    Raises ``ValueError`` if the template dataset length differs from
    the snapshot — ``_num_batches = num_samples // batch_size`` drives
    the index ranges each yield emits.
    """
    saved_n = int(sd["num_samples"])
    template_n = len(template.data_source)
    if saved_n != template_n:
        raise ValueError(
            f"SequentialBatchSampler.from_state_dict: template dataset "
            f"length {template_n} does not match snapshot "
            f"num_samples={saved_n}.  Restoring with a differently-sized "
            "dataset would silently expose / skip different indices after "
            "resume."
        )
    sampler = SequentialBatchSampler(
        template.data_source,
        batch_size=int(sd["batch_size"]),
    )
    sampler._consumed = int(sd["consumed"])
    return sampler


def _register_sequential_batch_sampler_serializer() -> None:
    from opaque.serialization import register_serializer

    register_serializer(
        SequentialBatchSampler,
        _state_dict_sequential,
        _from_state_dict_sequential,
    )


_register_sequential_batch_sampler_serializer()
