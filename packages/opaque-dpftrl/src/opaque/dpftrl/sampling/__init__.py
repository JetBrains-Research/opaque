"""Matrix-factorization participation samplers.

These samplers generate participation patterns required by MF mechanisms
(band-MF, BLT, BSR, lambda-CGD) — b-min-separation, cyclic Poisson,
balls-in-bins, and sequential batches.
"""

from opaque.dpftrl.sampling._b_min_sep import BMinSepSampler
from opaque.dpftrl.sampling._balls_in_bins import BallsInBinsSampler
from opaque.dpftrl.sampling._cyclic_poisson import CyclicPoissonSampler
from opaque.dpftrl.sampling._sequential import SequentialBatchSampler

__all__ = [
    "BMinSepSampler",
    "BallsInBinsSampler",
    "CyclicPoissonSampler",
    "SequentialBatchSampler",
]
