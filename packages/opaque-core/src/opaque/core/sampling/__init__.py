"""Sampling helpers shared across DP mechanisms.

Core exposes only the mechanism-agnostic collate wrapper :func:`empty_collate`
that handles empty/variable-size batches from any Poisson-style sampler. The
samplers themselves live next to their mechanism:

- :class:`opaque.dpsgd.sampling.PoissonSampler` — generic Poisson batch sampler.
- :class:`opaque.dpsgd.sampling.TruncatedPoissonSampler` — fixed-batch DP-SGD.
- :class:`opaque.dpftrl.sampling.CyclicPoissonSampler`,
  :class:`~opaque.dpftrl.sampling.BMinSepSampler`,
  :class:`~opaque.dpftrl.sampling.BallsInBinsSampler`,
  :class:`~opaque.dpftrl.sampling.SequentialBatchSampler` — DP-FTRL.

For dataset sharding across distributed ranks, use
:func:`opaque.distributed.local_shard`.
"""

from opaque.core.sampling.collate import empty_collate

__all__ = [
    "empty_collate",
]
