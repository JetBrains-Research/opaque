"""DP-FTRL accounting mechanism factories impl (matrix factorization).

For correlated MF mechanisms (BLT, BSR, BISR, λCGD), build the dataclass
via the corresponding ``*Strategy.as_mechanism(noise_multiplier)`` helper
in :mod:`opaque.dpftrl.noise`; the strategy owns the structural data those
mechanisms need for BnB amplification and ``at_step``.
"""

from opaque.api.accounting.dpftrl.mechanisms._band_mf import band_mf
from opaque.api.accounting.dpftrl.mechanisms._identity import identity_mf

__all__ = ["band_mf", "identity_mf"]
