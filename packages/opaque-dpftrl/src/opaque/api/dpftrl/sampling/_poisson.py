"""Cyclic Poisson sampler for DP-FTRL training (identity and band-MF regimes).

The dataset is split into ``bands`` disjoint groups (see ``partition_type``).
Step ``i`` activates only group ``i % bands``; each example in that group is
included independently with probability ``sample_rate``, so the batch size is
Binomial on that group's size.  That rotating participation pattern pairs with
correlated MF noise (e.g. BandMF).

If ``bands == 1`` there is a single group covering the whole dataset, so every
step is ordinary Poisson subsampling — use this with ``identity_strategy()``
inside ``ftrl_acc.mf_gaussian(nm, identity_strategy())`` and
``ftrl_acc.poisson(...)``.  If ``bands > 1``, set ``bands`` to match the
strategy you wrap in ``ftrl_acc.mf_gaussian(nm, band_mf_strategy(bands=...))``.

Optional ``truncated_batch_size`` caps the realised per-step batch.  Pair with
``ftrl_acc.poisson(..., truncated_batch_size=, dataset_size=)`` so the
accounting matches the runtime cap; that combination is only supported for the
identity strategy (``bands == 1``).

For distributed training, shard the dataset before constructing the
sampler with ``opaque.distributed.local_shard`` and derive a per-rank
key with ``opaque.random.fold_in(key, rank)``.

References:
    - BandMF amplification: https://arxiv.org/abs/2306.08153
    - Cyclic Poisson sampling for matrix mechanisms: https://arxiv.org/abs/2211.06530
"""

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from torch.utils.data import Sampler

from opaque.random.types import RngKey
from opaque.api.dpftrl.sampling._partitions import (
    PartitionType,
    _equal_split_partition,
    _independent_partition,
)


class CyclicPoissonSampler(Sampler):
    """Cyclic Poisson subsampling for DP-FTRL (one active group per step).

    Disjoint example groups are fixed at construction (modulo
    ``partition_type``).  Step ``i`` samples only from group ``i % bands``;
    each eligible example is a Bernoulli(``sample_rate``) draw, so batch size is
    random within that group and the active group advances each step.

    Use ``bands=1`` for identity MF (full dataset, plain Poisson each step,
    ``identity_strategy`` / ``identity_mf`` and ``ftrl_acc.poisson``).  For
    BandMF, set ``bands`` to the band count in ``band_mf_strategy`` / ``BandMf``.

    Args:
        data_source: Dataset to sample from (any object with ``__len__``).
        sample_rate: Per-step inclusion probability ``∈ (0, 1]``.
        bands: Number of groups in the cycle.  ``1`` = identity-style plain
            Poisson on the full dataset every step.
        n_steps: Total number of batches to yield.  Defaults to ``1``.
        partition_type: Partition strategy when ``bands > 1``.
        truncated_batch_size: Optional cap on per-step batch size (truncated
            Poisson; use matching accounting—privacy is weaker than uncapped
            Poisson at the same ``sample_rate``).  Only the ``bands == 1`` /
            ``IdentityMf`` combination is supported in
            :func:`opaque.dpftrl.accounting.poisson`.
        key: RNG key for reproducibility.

    Example::

        from opaque.random import key
        sampler = CyclicPoissonSampler(
            dataset, sample_rate=0.01, bands=4, n_steps=1000, key=key(42),
        )
        loader = DataLoader(dataset, batch_sampler=sampler)

    Note:
        Batch sizes are variable (Poisson).  Expected batch size per step is
        ``|group| * sample_rate`` where ``|group| = |D| / bands``.  Use with
        ``DataLoader``'s ``batch_sampler``.
    """

    def __init__(
        self,
        data_source: object,
        sample_rate: float,
        bands: int = 1,
        n_steps: int = 1,
        partition_type: PartitionType = PartitionType.EQUAL_SPLIT,
        truncated_batch_size: int | None = None,
        *,
        key: RngKey,
    ):
        super().__init__()

        if len(data_source) == 0:
            raise ValueError("data_source must not be empty")
        if not 0 < sample_rate <= 1:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        if bands < 1:
            raise ValueError(f"bands must be >= 1, got {bands}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if truncated_batch_size is not None and truncated_batch_size < 1:
            raise ValueError(
                f"truncated_batch_size must be >= 1, got {truncated_batch_size}"
            )

        self.num_examples = len(data_source)
        self._key = key
        self.generator = np.random.default_rng(key.seed)

        if partition_type == PartitionType.INDEPENDENT:
            partition_fn = _independent_partition
        elif partition_type == PartitionType.EQUAL_SPLIT:
            partition_fn = _equal_split_partition
        else:
            raise ValueError(f"Unsupported partition_type: {partition_type}")

        dtype = np.min_scalar_type(-self.num_examples)
        self.partition = partition_fn(self.num_examples, bands, self.generator, dtype)

        self.data_source = data_source
        self.sample_rate = sample_rate
        self.bands = bands
        self.n_steps = n_steps
        self.partition_type = partition_type
        self.truncated_batch_size = truncated_batch_size
        self._consumed = 0

    def _sample_step(self, step_idx: int) -> list[int]:
        """Sample one batch at the given step index.

        The active group rotates with ``step_idx % bands``; the per-step
        generator advance is the same regardless of whether the result
        is yielded or discarded, so state-dict replay can use this
        helper to fast-forward the generator without yielding.
        """
        group_idx = step_idx % self.bands
        group = self.partition[group_idx]

        sample_size = self.generator.binomial(n=len(group), p=self.sample_rate)
        if self.truncated_batch_size is not None:
            sample_size = min(sample_size, self.truncated_batch_size)

        if sample_size > 0:
            batch = self.generator.choice(
                group, size=sample_size, replace=False, shuffle=False
            )
        else:
            batch = np.array([], dtype=group.dtype)

        return batch.tolist()

    def __iter__(self) -> Iterator[list[int]]:
        """Yield Poisson batches.

        For each step, samples from group ``step % bands`` with inclusion
        probability ``sample_rate`` per example, optionally capped at
        ``truncated_batch_size``.
        """
        for step in range(self._consumed, self.n_steps):
            batch = self._sample_step(step)
            self._consumed += 1
            yield batch

    def __len__(self) -> int:
        return self.n_steps

    @property
    def consumed(self) -> int:
        """Number of batches yielded so far (resume cursor)."""
        return self._consumed

    @property
    def expected_batch_size(self) -> float:
        """Expected batch size: ``|group| * sample_rate`` (before truncation)."""
        avg_group_size = self.num_examples / self.bands
        return avg_group_size * self.sample_rate

    @property
    def batch_size_variance(self) -> float:
        """Variance of batch size (Poisson property; before truncation)."""
        avg_group_size = self.num_examples / self.bands
        return avg_group_size * self.sample_rate * (1 - self.sample_rate)


def _state_dict_cyclic_poisson(s: CyclicPoissonSampler) -> dict[str, Any]:
    """Serialise ``CyclicPoissonSampler`` state.

    Partition is deterministic from ``(key, bands, num_examples,
    partition_type)`` so we don't store it explicitly — load
    reconstructs it from the saved key.
    """
    return {
        "key_seed": int(s._key.seed),
        "key_impl": str(s._key.impl),
        "consumed": int(s._consumed),
        "sample_rate": float(s.sample_rate),
        "bands": int(s.bands),
        "n_steps": int(s.n_steps),
        "partition_type": s.partition_type.name,
        "truncated_batch_size": s.truncated_batch_size,
    }


def _from_state_dict_cyclic_poisson(
    template: CyclicPoissonSampler, sd: Mapping[str, Any]
) -> CyclicPoissonSampler:
    """Rebuild ``CyclicPoissonSampler`` at the saved cursor.

    The dataset comes from ``template``; the partition reconstructs
    deterministically from the saved key + bands; the generator is
    fast-forwarded by replaying ``consumed`` discarded ``_sample_step``
    calls so the next yielded batch matches a continuous run.
    """
    sampler = CyclicPoissonSampler(
        template.data_source,
        sample_rate=float(sd["sample_rate"]),
        bands=int(sd["bands"]),
        n_steps=int(sd["n_steps"]),
        partition_type=PartitionType[sd["partition_type"]],
        truncated_batch_size=sd.get("truncated_batch_size"),
        key=RngKey(seed=int(sd["key_seed"]), impl=str(sd["key_impl"])),
    )
    consumed = int(sd["consumed"])
    for step_idx in range(consumed):
        sampler._sample_step(step_idx)
    sampler._consumed = consumed
    return sampler


def _register_cyclic_poisson_sampler_serializer() -> None:
    from opaque.serialization import register_serializer

    register_serializer(
        CyclicPoissonSampler,
        _state_dict_cyclic_poisson,
        _from_state_dict_cyclic_poisson,
    )


_register_cyclic_poisson_sampler_serializer()
