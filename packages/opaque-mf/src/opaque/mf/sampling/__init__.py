"""Matrix-factorization participation samplers.

These samplers generate participation patterns required by MF mechanisms
(band-MF, BLT, BSR, lambda-CGD) — b-min-separation, cyclic Poisson,
balls-in-bins, and sequential batches.
"""

from opaque.mf.sampling.b_min_sep import BMinSepSampler
from opaque.mf.sampling.balls_in_bins import BallsInBinsSampler
from opaque.mf.sampling.cyclic_poisson import CyclicPoissonSampler
from opaque.mf.sampling.sequential import SequentialBatchSampler

__all__ = [
    "BMinSepSampler",
    "BallsInBinsSampler",
    "CyclicPoissonSampler",
    "SequentialBatchSampler",
]
