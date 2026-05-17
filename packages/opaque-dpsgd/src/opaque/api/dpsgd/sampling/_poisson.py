"""Poisson subsampling for DP-SGD training.

Poisson subsampling: each example is independently included with
probability ``sample_rate``.  Optional ``truncated_batch_size`` caps the
realised batch for stable sizes and memory; that is **weaker** for privacy than
plain Poisson at the same ``sample_rate``—pair with
:func:`opaque.dpsgd.accounting.poisson` passing both
``truncated_batch_size`` and ``dataset_size``.

For distributed training, shard the dataset **before** creating the
sampler using ``local_shard()`` and derive a per-rank key with
``fold_in(key, rank)``.
"""

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from torch.utils.data import Sampler

from ._helpers import _plain_poisson_step_indices
from opaque.random.types import RngKey


class PoissonSampler(Sampler):
    """Poisson sampler for privacy amplification (DP-SGD).

    Each example is independently included with probability
    ``sample_rate``.  When ``truncated_batch_size`` is set, batches are
    capped at that size (uniform without replacement from the Poisson
    sample).

    For distributed training, shard the dataset externally and pass a
    per-rank key via ``fold_in(key, rank)``::

        from opaque.distributed import local_shard

        shard = local_shard(dataset, rank=rank, world_size=world_size)
        sampler = PoissonSampler(
            shard, sample_rate=0.01, key=fold_in(key(42), rank)
        )

    Args:
        data_source: Dataset to sample from (any object with ``__len__``).
        sample_rate: Probability of including each example ``∈ (0, 1]``.
        n_steps: Number of batches to yield. ``None`` yields indefinitely.
        truncated_batch_size: Optional cap on per-step batch size (truncated
            Poisson; use matching accounting—privacy is weaker than uncapped
            Poisson at the same ``sample_rate``).
        key: RNG key for reproducibility.

    Example::

        from opaque.random import key
        sampler = PoissonSampler(
            dataset, sample_rate=0.01, n_steps=1000, key=key(42),
        )
        loader = DataLoader(dataset, batch_sampler=sampler)

    Note:
        - Batch sizes are variable (Poisson property).
        - Expected batch size: ``len(data_source) * sample_rate`` (uncapped).
        - Use with ``DataLoader``'s ``batch_sampler`` parameter.
        - Resume via :func:`opaque.serialization.state_dict` /
          :func:`opaque.serialization.from_state_dict`: state captures the
          original ``key`` plus the consumed-step count; load replays the
          generator forward to the resume position.
    """

    def __init__(
        self,
        data_source: object,
        sample_rate: float,
        n_steps: int | None = None,
        truncated_batch_size: int | None = None,
        *,
        key: RngKey,
    ):
        super().__init__()

        if not 0 < sample_rate <= 1:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        if n_steps is not None and n_steps < 1:
            raise ValueError(f"n_steps must be >= 1 or None, got {n_steps}")
        if truncated_batch_size is not None and truncated_batch_size < 1:
            raise ValueError(
                f"truncated_batch_size must be >= 1, got {truncated_batch_size}"
            )

        self.data_source = data_source
        self.sample_rate = sample_rate
        self.n_steps = n_steps
        self.truncated_batch_size = truncated_batch_size

        self._num_samples = len(data_source)
        # Original key kept for state-dict round-trips; the live
        # ``generator`` advances each yield and isn't reconstructable
        # from the key alone after that point.
        self._key = key
        self.generator = np.random.default_rng(key.seed)
        self._consumed = 0

    def _sample_step(self) -> list[int]:
        return _plain_poisson_step_indices(
            self.generator,
            self._num_samples,
            self.sample_rate,
            self.truncated_batch_size,
        )

    def __iter__(self) -> Iterator[list[int]]:
        """Yield variable-size batches as lists of indices.

        Each iteration samples the entire dataset using Poisson subsampling,
        optionally truncated.  ``self._consumed`` increments after each
        yielded batch so state-dict snapshots reflect the live position.
        """
        if self.n_steps is None:
            while True:
                batch = self._sample_step()
                self._consumed += 1
                yield batch
        else:
            for _ in range(self._consumed, self.n_steps):
                batch = self._sample_step()
                self._consumed += 1
                yield batch

    def __len__(self) -> int:
        """Return number of batches.

        Raises:
            TypeError: If n_steps is None (infinite iteration).
        """
        if self.n_steps is None:
            raise TypeError("len() of unsized object (n_steps=None)")
        return self.n_steps

    @property
    def consumed(self) -> int:
        """Number of batches yielded so far (resume cursor)."""
        return self._consumed

    @property
    def expected_batch_size(self) -> float:
        """Expected batch size = num_samples * sample_rate (before truncation)."""
        return self._num_samples * self.sample_rate

    @property
    def batch_size_variance(self) -> float:
        """Variance of batch size for Poisson sampling (before truncation)."""
        return self._num_samples * self.sample_rate * (1 - self.sample_rate)


def _state_dict_poisson(s: PoissonSampler) -> dict[str, Any]:
    """Serialise ``PoissonSampler`` state.

    Saves the original ``key`` (so the generator can be reconstructed)
    plus a ``consumed`` counter (so the loader can replay the generator
    forward to the resume position).  The dataset is *not* serialised —
    it's supplied by the ``template`` argument on load.
    """
    return {
        "key_seed": int(s._key.seed),
        "key_impl": str(s._key.impl),
        "consumed": int(s._consumed),
        "sample_rate": float(s.sample_rate),
        "n_steps": s.n_steps,
        "truncated_batch_size": s.truncated_batch_size,
    }


def _from_state_dict_poisson(
    template: PoissonSampler, sd: Mapping[str, Any]
) -> PoissonSampler:
    """Rebuild ``PoissonSampler`` at the saved cursor.

    The dataset comes from ``template`` (it can't be serialised); every
    other constructor arg is taken from ``sd``.  Replay advances the
    numpy generator by ``consumed`` discarded ``_sample_step`` calls so
    the next yielded batch matches a continuous run.
    """
    sampler = PoissonSampler(
        template.data_source,
        sample_rate=float(sd["sample_rate"]),
        n_steps=sd.get("n_steps"),
        truncated_batch_size=sd.get("truncated_batch_size"),
        key=RngKey(seed=int(sd["key_seed"]), impl=str(sd["key_impl"])),
    )
    consumed = int(sd["consumed"])
    for _ in range(consumed):
        sampler._sample_step()
    sampler._consumed = consumed
    return sampler


def _register_poisson_sampler_serializer() -> None:
    from opaque.serialization import register_serializer

    register_serializer(PoissonSampler, _state_dict_poisson, _from_state_dict_poisson)


_register_poisson_sampler_serializer()
