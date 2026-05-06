"""Cross-cutting transformations shared across DP-SGD and DP-FTRL.

- :func:`second_moment` — convert a first-moment mechanism to second-moment

``adaclip`` lives in :mod:`opaque.dpsgd.accounting.mechanisms` (it is a
DP-SGD-specific transformation).
"""

from opaque.accounting.transformations._second_moment import second_moment

__all__ = ["second_moment"]
