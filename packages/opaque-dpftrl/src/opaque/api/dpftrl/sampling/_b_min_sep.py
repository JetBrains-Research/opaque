"""Warm-start b-min-sep subsampling for BandMF (Dong & Ganesh, arXiv:2602.09338).

Algorithm 2 assigns each example an initial state ``s_j`` so the expected batch
size is approximately constant across iterations, then Poisson-samples from
eligible examples (those not in any of the previous ``b-1`` batches).

The per-iteration inclusion probability ``p`` matches the paper (not the
per-example rate ``p_0`` used in cyclic Poisson accounting). For target
``p_0 = E[|B|]/|D|``, use ``p = p_0 / (1 - p_0 * (b - 1))`` when ``b > 1``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from opaque.random.types import RngKey


class BMinSepSampler(Sampler):
    """Warm-start b-min-sep sampler for BandMF privacy amplification.

    Each iteration draws a batch by including each *eligible* example
    independently with probability ``p``. Eligibility excludes any example
    that appeared in one of the previous ``bands - 1`` batches (Algorithm 2).

    Args:
        data_source: Dataset with ``__len__``.
        bands: Minimum-separation parameter ``b`` (same as BandMF bandwidth).
        sampling_prob: Paper's ``p`` in ``(0, 1]`` (per-iteration inclusion
            probability for each eligible example).
        n_steps: Number of batches to yield.
        key: RNG key for reproducibility.

    Note:
        Use ``batch_sampler=...`` in ``DataLoader``. Batch sizes are random
        (Poisson). For the same expected batch size as Poisson with
        per-example rate ``p_0`` and ``b`` bands, set
        ``sampling_prob = p_0 / (1 - p_0 * (bands - 1))`` (``bands==1`` → ``p_0``).
    """

    def __init__(
        self,
        data_source: object,
        bands: int,
        sampling_prob: float,
        n_steps: int,
        *,
        key: RngKey,
    ):
        super().__init__()
        if len(data_source) == 0:
            raise ValueError("data_source must not be empty")
        if bands < 1:
            raise ValueError(f"bands must be >= 1, got {bands}")
        if not 0 < sampling_prob <= 1:
            raise ValueError(f"sampling_prob must be in (0, 1], got {sampling_prob}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        self.num_examples = len(data_source)
        self.data_source = data_source
        self.bands = bands
        self.sampling_prob = sampling_prob
        self.n_steps = n_steps
        self.generator = np.random.default_rng(key.seed)

        # Warm-start initial cooldown (Algorithm 2, lines 1–2).
        self._bars_left = np.zeros(self.num_examples, dtype=np.int32)
        if bands > 1:
            p = sampling_prob
            denom = 1.0 + (bands - 1) * p
            for j in range(self.num_examples):
                if self.generator.random() * denom < 1.0:
                    self._bars_left[j] = 0
                else:
                    # Uniform on {1, …, b-1}
                    self._bars_left[j] = int(self.generator.integers(1, bands))

        self._recent: deque[set[int]] = deque(maxlen=max(0, bands - 1))

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.n_steps):
            exclude: set[int] = set()
            for batch in self._recent:
                exclude.update(batch)

            eligible = [
                idx for idx in range(self.num_examples) if self._bars_left[idx] == 0
            ]
            eligible = [i for i in eligible if i not in exclude]

            n_elig = len(eligible)
            if n_elig == 0:
                batch_indices: list[int] = []
            else:
                sample_size = self.generator.binomial(n=n_elig, p=self.sampling_prob)
                if sample_size > 0:
                    pos = self.generator.choice(
                        n_elig, size=sample_size, replace=False, shuffle=False
                    )
                    batch_indices = [eligible[int(k)] for k in pos]
                else:
                    batch_indices = []

            batch_set = set(batch_indices)

            # Advance cooldown: one step closer to eligibility.
            mask = self._bars_left > 0
            self._bars_left[mask] -= 1

            # Selected examples cannot appear in the next (bands - 1) iterations.
            for idx in batch_indices:
                self._bars_left[idx] = max(self._bars_left[idx], self.bands - 1)

            self._recent.append(batch_set)
            yield batch_indices

    def __len__(self) -> int:
        return self.n_steps

    @property
    def expected_batch_size(self) -> float:
        """Approximate expected batch size once the chain has mixed."""
        p0 = self.sampling_prob / (1.0 + self.sampling_prob * max(0, self.bands - 1))
        return self.num_examples * p0

    @property
    def batch_size_variance(self) -> float:
        """Approximate variance (Poisson on eligible pool ~ full size)."""
        p = self.sampling_prob
        n = float(self.num_examples)
        return n * p * (1.0 - p)
