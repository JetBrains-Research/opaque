"""DP-FTRL accounting mechanism factories façade (matrix factorization).

Correlated MF mechanisms (``Blt``, ``Bsr``, ``Bisr``, ``LambdaCgd``) are
constructed via the matching ``*Strategy.as_mechanism(noise_multiplier)``
helper in :mod:`opaque.dpftrl.noise` rather than exposed as factories here.
"""

from opaque.api.accounting.dpftrl.mechanisms import band_mf, identity_mf

__all__ = ["band_mf", "identity_mf"]
