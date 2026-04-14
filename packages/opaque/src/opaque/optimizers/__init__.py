"""DP-compatible optimizers.

- :func:`jme_adam` — Adam optimizer paired with JME noise
  (:func:`~opaque.noise.mf.jme_noise`).  Compatible with the
  ``torchopt`` ``GradientTransformation`` protocol (``init`` / ``update``
  / ``apply_updates``).
"""

from opaque.optimizers.jme_adam import jme_adam, JmeAdamState

__all__ = ["jme_adam", "JmeAdamState"]
