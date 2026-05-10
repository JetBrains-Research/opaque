"""Matrix-factorization participation samplers for DP-FTRL.

These samplers generate the participation patterns required by MF
mechanisms (band-MF, BLT, BSR, lambda-CGD) — b-min-separation, Poisson
(``bands`` selects cyclic BandMF vs identity baseline), balls-in-bins,
and sequential batches.
"""

from opaque.dpftrl.sampling._b_min_sep import BMinSepSampler
from opaque.dpftrl.sampling._balls_in_bins import BallsInBinsSampler
from opaque.dpftrl.sampling._poisson import PoissonSampler
from opaque.dpftrl.sampling._sequential import SequentialBatchSampler

__all__ = [
    "BMinSepSampler",
    "BallsInBinsSampler",
    "PoissonSampler",
    "SequentialBatchSampler",
]
