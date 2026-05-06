"""Transformations that modify existing DP processes.

Transformations take an existing mechanism (e.g. Gaussian) and return a new
mechanism with adjusted parameters. Unlike amplification (which wraps a
mechanism with subsampling), transformations change the mechanism itself.

The transformation dataclasses (``AdaClip``, ``SecondMoment``) live in
:mod:`opaque.accounting.transformations.types`.
"""

from opaque.accounting.transformations._adaclip import adaclip
from opaque.accounting.transformations._second_moment import second_moment

__all__ = ["adaclip", "second_moment"]
