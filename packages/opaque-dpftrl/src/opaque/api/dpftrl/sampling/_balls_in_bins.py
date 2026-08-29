"""Balls-in-Bins sampler for DP-SGD training.

In the Balls-in-Bins (BnB) sampling scheme each example independently
and uniformly picks one of ``num_bins`` bins (Definition 3.1 of
Choquette-Choo et al. 2024).  Bin sizes are therefore **random**,
following Binomial(N, 1/num_bins) marginally.

The bin assignment is generated once in ``__init__`` and reused every
epoch (round-robin).  This is required for the dominating-pair privacy
accounting (Lemma 3.2) to be valid: each example must stay in its bin
across all epochs.

For distributed training, shard the dataset **before** creating the
sampler using ``local_shard()`` and derive a per-rank key with
``fold_in(key, rank)``.

This is *not* the same scheme as
:class:`opaque.dpsgd.sampling.KOutOfTSampler` with ``allocation="block"``,
which redraws the
assignment every epoch.  Redrawing is valid only when the noise is
uncorrelated across steps, which is exactly what the matrix mechanism
gives up.  Pair this sampler only with ``dpftrl_acc.balls_in_bins``.

References:
    - Chua et al. (2025), "Balls-and-Bins Sampling for DP-SGD":
      https://arxiv.org/abs/2412.16802
    - Choquette-Choo et al. (2024), "Privacy Amplification for Matrix Mechanisms"
"""

from collections.abc import Iterator, Mapping, Sized
from typing import Any

import numpy as np
from torch.utils.data import Sampler

from opaque.exceptions import ConfigurationError, InputTypeError
from opaque.random.types import RngKey

_MIN_NUM_BINS = 2


class BallsInBinsSampler(Sampler):
    """Balls-in-Bins sampler: random independent bin assignment, fixed across epochs.

    Each example independently chooses one of ``num_bins`` bins uniformly
    at random (true BnB, Definition 3.1 of Choquette-Choo et al. 2024).
    Bin sizes are random — some bins may be larger or smaller than the
    expected size ``len(dataset) / num_bins``, and bins can even be empty.
    Empty bin slots yield empty batches so every epoch retains exactly
    ``num_bins`` optimizer steps.

    The assignment is generated once and reused every epoch — this is
    required for the BnB privacy accounting (Lemma 3.2) to be valid.

    For distributed training, shard the dataset externally and pass a
    per-rank key via ``fold_in(key, rank)``.

    Args:
        data_source: Dataset to sample from (any object with ``__len__``).
        num_bins: Number of bins per epoch (b ≥ 2).  Typically
            ``dataset_size / desired_batch_size``.
        n_steps: Total number of batches to yield.  Must be a positive
            multiple of ``num_bins`` (per-bin participation count is
            ``n_steps // num_bins``).  ``None`` yields indefinitely.
        key: RNG key for reproducibility.

    Example:
        >>> from opaque.random import key
        >>> sampler = BallsInBinsSampler(
        ...     dataset, num_bins=100, n_steps=1000, key=key(42)
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        data_source: Sized,
        num_bins: int,
        n_steps: int | None = None,
        *,
        key: RngKey,
    ):
        super().__init__()

        if len(data_source) == 0:
            raise ConfigurationError(*("data_source must not be empty",))
        if num_bins < _MIN_NUM_BINS:
            raise ConfigurationError(*(f"num_bins must be >= 2, got {num_bins}",))
        if n_steps is not None:
            if n_steps < 1:
                raise ConfigurationError(
                    *(f"n_steps must be >= 1 or None, got {n_steps}",)
                )
            if n_steps % num_bins != 0:
                raise ConfigurationError(
                    *(
                        f"n_steps ({n_steps}) must be a positive multiple of "
                        f"num_bins ({num_bins}); BnB analysis assumes integer epochs.",
                    )
                )

        self.data_source: Sized = data_source
        self.num_bins = num_bins
        self.n_steps = n_steps

        self._num_samples = len(data_source)
        self._key = key

        generator = np.random.default_rng(key.seed)
        # True BnB: each example independently picks a bin.
        assignments = generator.integers(0, num_bins, size=self._num_samples)
        self._bins: list[list[int]] = [[] for _ in range(num_bins)]
        for idx, b in enumerate(assignments):
            self._bins[b].append(idx)
        self._consumed = 0

    @property
    def num_epochs(self) -> int | None:
        """Per-bin participation count: ``n_steps // num_bins``."""
        return None if self.n_steps is None else self.n_steps // self.num_bins

    @property
    def consumed(self) -> int:
        """Number of batches yielded so far (resume cursor)."""
        return self._consumed

    def __iter__(self) -> Iterator[list[int]]:
        """Yield every bin slot, including empty batches, for ``num_epochs``.

        The same bin assignment is reused every epoch — the BnB
        privacy accounting (Lemma 3.2) requires the assignment be fixed across
        the run. Empty slots must remain in the stream: dropping them would
        make the executed step schedule differ from the accountant's
        ``num_bins``-slot epoch.
        """
        bins = self._bins
        if self.n_steps is None:
            i = self._consumed
            while True:
                # Increment before yield so a snapshot taken mid-iter
                # reports the count of batches actually emitted so far.
                self._consumed = i + 1
                yield bins[i % self.num_bins]
                i += 1
        else:
            for i in range(self._consumed, self.n_steps):
                self._consumed = i + 1
                yield bins[i % self.num_bins]

    def __len__(self) -> int:
        """Declared bin slots remaining.

        After a partial run, reflects what ``__iter__`` will yield, including
        empty batches, so ``len(DataLoader(...))`` matches the accountant's
        remaining schedule.

        Raises:
            TypeError: If n_steps is None (infinite iteration).
        """
        if self.n_steps is None:
            raise InputTypeError(*("len() of unsized object (n_steps=None)",))
        return self.n_steps - self._consumed

    @property
    def expected_batch_size(self) -> float:
        """Expected batch size: len(dataset) / num_bins."""
        return self._num_samples / self.num_bins

    @property
    def sample_rate(self) -> float:
        """Effective sampling rate: 1 / num_bins."""
        return 1.0 / self.num_bins


def _state_dict_balls_in_bins(s: BallsInBinsSampler) -> dict[str, Any]:
    """Serialise ``BallsInBinsSampler`` state.

    Bin assignment is deterministic from ``(key, num_bins, num_samples)``;
    ``num_samples`` is persisted so the loader can validate the
    template dataset length before relying on deterministic
    reconstruction.  Round-robin iteration is deterministic given the
    assignment, so no Markov-state replay is needed on load — the
    cursor alone fixes the resume point.
    """
    return {
        "key_seed": int(s._key.seed),
        "key_impl": str(s._key.impl),
        "consumed": int(s._consumed),
        "num_samples": int(s._num_samples),
        "num_bins": int(s.num_bins),
        "n_steps": s.n_steps,
    }


def _from_state_dict_balls_in_bins(
    template: BallsInBinsSampler, sd: Mapping[str, Any]
) -> BallsInBinsSampler:
    """Rebuild ``BallsInBinsSampler`` at the saved cursor.

    The dataset comes from ``template``; the bin assignment is
    reconstructed deterministically by the constructor; the cursor is
    restored so ``__iter__`` resumes at the right round-robin
    position.

    Raises ``ValueError`` if the template dataset length differs from
    the snapshot — the bin assignment depends on ``num_samples`` (each
    example independently picks a bin), so a mismatched length would
    silently produce a different assignment.
    """
    saved_n = int(sd["num_samples"])
    template_n = len(template.data_source)
    if saved_n != template_n:
        raise ConfigurationError(
            *(
                f"BallsInBinsSampler.from_state_dict: template dataset length "
                f"{template_n} does not match snapshot num_samples={saved_n}.  "
                "Restoring with a differently-sized dataset would silently "
                "produce a different bin assignment, voiding the BnB privacy "
                "accounting (Lemma 3.2 requires fixed assignment across the run).",
            )
        )
    sampler = BallsInBinsSampler(
        template.data_source,
        num_bins=int(sd["num_bins"]),
        # Take ``n_steps`` from the template — caller may extend or
        # shorten the run on resume; the cursor below fixes the
        # round-robin resume position.
        n_steps=template.n_steps,
        key=RngKey(seed=int(sd["key_seed"]), impl=str(sd["key_impl"])),
    )
    sampler._consumed = int(sd["consumed"])
    return sampler


def _register_balls_in_bins_sampler_serializer() -> None:
    from opaque.serialization import register_serializer

    register_serializer(
        BallsInBinsSampler,
        _state_dict_balls_in_bins,
        _from_state_dict_balls_in_bins,
    )


_register_balls_in_bins_sampler_serializer()
