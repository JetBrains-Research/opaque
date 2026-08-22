"""DP-SGD noise mechanisms façade — Gaussian (optionally bounded).

:func:`gaussian_noise` builds the mechanism; :func:`noise_stddev` answers
what standard deviation that mechanism will apply to a given contribution
bound, without needing a step to have run.

State (``GaussianNoiseState``) lives in :mod:`opaque.dpsgd.noise.types`.
"""

from opaque.api.dpsgd.noise import gaussian_noise, noise_stddev
from opaque.dpsgd.noise import types

__all__ = ["gaussian_noise", "noise_stddev", "types"]
