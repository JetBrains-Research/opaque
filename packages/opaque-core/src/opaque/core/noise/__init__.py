"""Generic noise-mechanism base class.

Only the ``NoiseMechanism`` abstract interface lives here. Concrete
DP-SGD mechanisms (Gaussian, truncated Gaussian, per-group) live in
``opaque.dpsgd.noise``. Matrix-factorization mechanisms live in
``opaque.dpftrl.noise``.
"""

from opaque.core.noise.types import NoiseState

__all__ = ["NoiseState"]
