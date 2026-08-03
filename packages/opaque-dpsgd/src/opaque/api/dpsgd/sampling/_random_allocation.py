"""Random-allocation sampler for DP-SGD.

Each epoch every example independently picks one of ``num_bins`` bins
uniformly at random, and the epoch yields those ``num_bins`` bins as
batches.  The assignment is **redrawn every epoch**.

This is *not* the same scheme as
:class:`opaque.dpftrl.sampling.BallsInBinsSampler`, which draws the bin
assignment once and reuses it for the whole run.  Fixed assignment is
required there by the matrix-mechanism dominating pair, which needs a
known separation between an example's participations.  DP-SGD has no such
constraint — its noise is uncorrelated across steps — and re-randomising
is strictly better: the re-randomised dominating pair has no larger
hockey-stick divergence at any ε, in either direction.

Pair this sampler with
:func:`opaque.dpsgd.accounting.random_allocation`, never with the
DP-FTRL balls-in-bins accountant.

References:
    - Chua et al. (2025), "Balls-and-Bins Sampling for DP-SGD"
    - Feldman & Shenfeld (2026), "Efficient privacy loss accounting for
      subsampling and random allocation"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from torch.utils.data import Sampler

from opaque.random import fold_in
from opaque.random.types import RngKey

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class RandomAllocationSampler(Sampler):
    """1-out-of-``num_bins`` random allocation, redrawn every epoch.

    Each epoch yields exactly ``num_bins`` batches whose union is the whole
    dataset.  Bin sizes are random (``Binomial(N, 1/num_bins)`` marginally),
    so **some batches may be empty** — that is the scheme, not a defect, and
    empty batches are handled downstream by the engine's collate path.
    Compacting them away would change each example's effective participation
    separation and silently break the accounting.

    Args:
        data_source: Dataset to sample from (any object with ``__len__``).
        num_bins: Bins per epoch (``b ≥ 2``).  Typically
            ``dataset_size / desired_batch_size``.
        n_steps: Total number of batches to yield. A final partial epoch yields
            its first ``n_steps % num_bins`` bins. ``None`` yields indefinitely.
        key: RNG key for reproducibility.

    Example:
        >>> from opaque.random import key
        >>> sampler = RandomAllocationSampler(
        ...     dataset, num_bins=100, n_steps=1000, key=key(42)
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        data_source: object,
        num_bins: int,
        n_steps: int | None = None,
        *,
        key: RngKey,
    ):
        super().__init__()

        if len(data_source) == 0:
            raise ValueError("data_source must not be empty")
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        if n_steps is not None and n_steps < 1:
            raise ValueError(f"n_steps must be >= 1 or None, got {n_steps}")

        self.data_source = data_source
        self.num_bins = num_bins
        self.n_steps = n_steps

        self._num_samples = len(data_source)
        self._key = key
        self._consumed = 0

    @property
    def num_epochs(self) -> int | None:
        """Allocation epochs touched by the stream, or ``None`` when unbounded."""
        return None if self.n_steps is None else -(-self.n_steps // self.num_bins)

    @property
    def consumed(self) -> int:
        """Number of batches yielded so far (resume cursor)."""
        return self._consumed

    @property
    def expected_batch_size(self) -> float:
        """Expected batch size = ``num_samples / num_bins``."""
        return self._num_samples / self.num_bins

    def _epoch_bins(self, epoch: int) -> list[list[int]]:
        """Bin assignment for ``epoch``, derived from the key.

        Deriving the assignment from ``fold_in(key, epoch)`` rather than a
        live generator makes resume O(1): any epoch can be reconstructed
        without replaying the ones before it.
        """
        rng = np.random.default_rng(fold_in(self._key, epoch).seed)
        assignment = rng.integers(0, self.num_bins, size=self._num_samples)
        bins: list[list[int]] = [[] for _ in range(self.num_bins)]
        for idx, b in enumerate(assignment):
            bins[b].append(int(idx))
        return bins

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches: ``num_bins`` per epoch, redrawn each epoch.

        Bins are emitted in index order.  Order within an epoch is
        irrelevant to the accounting — the dominating pair is exchangeable
        over bins — and all the randomness comes from the redraw.
        """
        i = self._consumed
        bins: list[list[int]] | None = None
        cur_epoch: int | None = None
        while self.n_steps is None or i < self.n_steps:
            epoch, slot = divmod(i, self.num_bins)
            # Rebuild whenever the epoch changes — including on the first
            # iteration after a mid-epoch resume, where ``slot != 0``.
            if bins is None or epoch != cur_epoch:
                bins = self._epoch_bins(epoch)
                cur_epoch = epoch
            self._consumed = i + 1
            yield bins[slot]
            i += 1

    def __len__(self) -> int:
        """Batches remaining.

        Raises:
            TypeError: If ``n_steps`` is None (infinite iteration).
        """
        if self.n_steps is None:
            raise TypeError("len() of unsized object (n_steps=None)")
        return self.n_steps - self._consumed


def _state_dict_random_allocation(s: RandomAllocationSampler) -> dict[str, Any]:
    """Serialise ``RandomAllocationSampler`` state.

    ``num_bins`` is privacy-load-bearing — it *is* the amplification factor —
    so it round-trips through the snapshot rather than the template.
    """
    return {
        "key_seed": int(s._key.seed),
        "key_impl": str(s._key.impl),
        "consumed": int(s._consumed),
        "num_samples": int(s._num_samples),
        "num_bins": int(s.num_bins),
        "n_steps": s.n_steps,
    }


def _from_state_dict_random_allocation(
    template: RandomAllocationSampler, sd: Mapping[str, Any]
) -> RandomAllocationSampler:
    """Rebuild ``RandomAllocationSampler`` at the saved cursor.

    The dataset and ``n_steps`` come from ``template`` (the user may extend
    the run on resume); ``key`` and ``num_bins`` come from the snapshot.
    Because each epoch's assignment is derived from ``fold_in(key, epoch)``,
    no replay loop is needed — restoring the cursor is enough.

    Raises ``ValueError`` if the template dataset length differs from the
    snapshot: the assignment is drawn over ``num_samples`` indices, so a
    mismatched length would silently emit a different stream.
    """
    saved_n = int(sd["num_samples"])
    template_n = len(template.data_source)
    if saved_n != template_n:
        raise ValueError(
            f"RandomAllocationSampler.from_state_dict: template dataset length "
            f"{template_n} does not match snapshot num_samples={saved_n}.  "
            "Restoring with a differently-sized dataset would silently emit "
            "a different allocation stream."
        )
    sampler = RandomAllocationSampler(
        template.data_source,
        num_bins=int(sd["num_bins"]),
        n_steps=template.n_steps,
        key=RngKey(seed=int(sd["key_seed"]), impl=str(sd["key_impl"])),
    )
    sampler._consumed = int(sd["consumed"])
    return sampler


def _register_random_allocation_sampler_serializer() -> None:
    from opaque.serialization import register_serializer

    register_serializer(
        RandomAllocationSampler,
        _state_dict_random_allocation,
        _from_state_dict_random_allocation,
    )


_register_random_allocation_sampler_serializer()
