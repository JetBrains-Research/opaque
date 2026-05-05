"""HF-shim dataloader helpers for DP training.

These wrappers keep a stable batch-sampler object (HF-style `set_epoch`
surface) while recreating immutable Opaque samplers per iteration.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from opaque.dpsgd.sampling import PoissonSampler, TruncatedPoissonSampler
from opaque.random import fold_in, key


class _OpaqueEpochBaseBatchSampler:
    """Stable epoch-aware wrapper around immutable Opaque samplers.

    The wrapper itself is mutable only in the HF shim and exists to provide
    an HF-compatible `set_epoch(epoch)` surface without mutating core Opaque
    samplers.
    """

    def __init__(
        self,
        *,
        dataset: object,
        sample_rate: float,
        num_iterations: int,
        seed: int,
    ) -> None:
        self._dataset = dataset
        self._sample_rate = float(sample_rate)
        self._num_iterations = int(num_iterations)
        self._seed = int(seed)
        self._epoch = 0
        self._active_sampler: Any | None = None
        self._pending_sampler_state: dict[str, Any] | None = None

    def __len__(self) -> int:
        return self._num_iterations

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch != self._epoch:
            self._epoch = epoch
            # Saved sampler state is only valid for the epoch it was created for.
            self._pending_sampler_state = None

    def state_dict(self) -> dict[str, Any]:
        if self._active_sampler is not None:
            return self._active_sampler.state_dict()
        return self._pending_sampler_state or {
            "key": {
                "seed": int(fold_in(key(self._seed), self._epoch).seed),
                "impl": str(key(self._seed).impl),
            },
            "iter_count": 0,
            "sample_rate": float(self._sample_rate),
            "num_iterations": int(self._num_iterations),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._pending_sampler_state = dict(state)
        if self._active_sampler is not None:
            self._active_sampler.load_state_dict(self._pending_sampler_state)
            self._pending_sampler_state = None

    def _make_sampler(self) -> Any:
        raise NotImplementedError

    def __iter__(self) -> Iterator[list[int]]:
        sampler = self._make_sampler()
        if self._pending_sampler_state is not None:
            sampler.load_state_dict(self._pending_sampler_state)
            self._pending_sampler_state = None
        self._active_sampler = sampler
        yield from sampler


class OpaqueEpochPoissonBatchSampler(_OpaqueEpochBaseBatchSampler):
    """Epoch-aware wrapper for :class:`PoissonSampler`."""

    def _make_sampler(self) -> PoissonSampler:
        return PoissonSampler(
            self._dataset,
            sample_rate=self._sample_rate,
            num_iterations=self._num_iterations,
            key=fold_in(key(self._seed), self._epoch),
        )


class OpaqueEpochTruncatedPoissonBatchSampler(_OpaqueEpochBaseBatchSampler):
    """Epoch-aware wrapper for :class:`TruncatedPoissonSampler`."""

    def __init__(
        self,
        *,
        dataset: object,
        sample_rate: float,
        num_iterations: int,
        seed: int,
        max_batch_size: int,
    ) -> None:
        super().__init__(
            dataset=dataset,
            sample_rate=sample_rate,
            num_iterations=num_iterations,
            seed=seed,
        )
        self._max_batch_size = int(max_batch_size)

    def _make_sampler(self) -> TruncatedPoissonSampler:
        return TruncatedPoissonSampler(
            self._dataset,
            sample_rate=self._sample_rate,
            max_batch_size=self._max_batch_size,
            num_iterations=self._num_iterations,
            key=fold_in(key(self._seed), self._epoch),
        )
