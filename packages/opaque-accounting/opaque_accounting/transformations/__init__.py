"""Transformations that modify existing DP processes.

Transformations take an existing mechanism (e.g. Gaussian) and return a new
mechanism with adjusted parameters. Unlike amplification (which wraps a
mechanism with subsampling), transformations change the mechanism itself.
"""

from opaque_accounting.transformations.adaclip import AdaClip, adaclip
from opaque_accounting.transformations.jme import Jme, jme_adam

__all__ = ["AdaClip", "adaclip", "Jme", "jme_adam"]
