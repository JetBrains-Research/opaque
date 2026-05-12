"""HF-shim dataloader helpers for DP training.

These wrappers keep a stable batch-sampler object (HF-style `set_epoch`
surface) while recreating immutable Opaque samplers per iteration.

Under DDP every rank sees the same epoch-keyed stream; combined with
``opaque.distributed.local_shard`` of the dataset, that produces disjoint
per-rank batches at the global Poisson rate.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from opaque.dpsgd.sampling import PoissonSampler
from opaque.random import fold_in, key
from opaque.random.types import RngKey


def _epoch_key(seed: int, epoch: int) -> RngKey:
    """Build the per-iteration RNG key for an epoch (identical on every rank)."""
    return fold_in(key(seed), epoch)


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

    def _key_for_epoch(self) -> RngKey:
        return _epoch_key(self._seed, self._epoch)

    def state_dict(self) -> dict[str, Any]:
        if self._active_sampler is not None and callable(
            getattr(self._active_sampler, "state_dict", None)
        ):
            return self._active_sampler.state_dict()
        ek = self._key_for_epoch()
        return self._pending_sampler_state or {
            "key": {
                "seed": int(ek.seed),
                "impl": str(ek.impl),
            },
            "iter_count": 0,
            "sample_rate": float(self._sample_rate),
            "num_iterations": int(self._num_iterations),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._pending_sampler_state = dict(state)
        if self._active_sampler is not None and callable(
            getattr(self._active_sampler, "load_state_dict", None)
        ):
            self._active_sampler.load_state_dict(self._pending_sampler_state)
            self._pending_sampler_state = None

    def _make_sampler(self) -> Any:
        raise NotImplementedError

    def __iter__(self) -> Iterator[list[int]]:
        sampler = self._make_sampler()
        if self._pending_sampler_state is not None and callable(
            getattr(sampler, "load_state_dict", None)
        ):
            sampler.load_state_dict(self._pending_sampler_state)
            self._pending_sampler_state = None
        self._active_sampler = sampler
        yield from sampler


class OpaqueEpochPoissonBatchSampler(_OpaqueEpochBaseBatchSampler):
    """Epoch-aware wrapper for :class:`PoissonSampler`.

    Optional ``truncated_batch_size`` caps per-step batch size (truncated
    Poisson sampling); use matching ``dpsgd_acc.poisson(..., truncated_batch_size=,
    dataset_size=)`` accounting.
    """

    def __init__(
        self,
        *,
        dataset: object,
        sample_rate: float,
        num_iterations: int,
        seed: int,
        truncated_batch_size: int | None = None,
    ) -> None:
        super().__init__(
            dataset=dataset,
            sample_rate=sample_rate,
            num_iterations=num_iterations,
            seed=seed,
        )
        self._truncated_batch_size = (
            int(truncated_batch_size) if truncated_batch_size is not None else None
        )

    def _make_sampler(self) -> PoissonSampler:
        return PoissonSampler(
            self._dataset,
            sample_rate=self._sample_rate,
            n_steps=self._num_iterations,
            truncated_batch_size=self._truncated_batch_size,
            key=self._key_for_epoch(),
        )
