"""DP-compatible optimizers.

- :func:`dp_adam` — Adam optimizer for use with JME noise
  (:func:`~opaque.noise.mf.mf_noise_jme`).  Compatible with the
  ``torchopt`` ``GradientTransformation`` protocol (``init`` / ``update``
  / ``apply_updates``).
"""

from opaque.optimizers.adam import dp_adam, DPAdamState

__all__ = ["dp_adam", "DPAdamState"]
