"""Transformations that modify existing DP processes.

Transformations take an existing mechanism (e.g. Gaussian) and return a new
mechanism with adjusted parameters. Unlike amplification (which wraps a
mechanism with subsampling), transformations change the mechanism itself.
"""

from opaque.accounting.transformations.adaclip import AdaClip, adaclip

__all__ = ["AdaClip", "adaclip"]
