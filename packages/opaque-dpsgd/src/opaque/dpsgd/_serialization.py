"""Serialization registration hook for :mod:`opaque.dpsgd`.

DP-SGD runtime state uses structural (de)serialisation by default.  Add
:func:`opaque.serialization.register_serialization_type` calls here for
non-default encodings.
"""

from __future__ import annotations

__all__: list[str] = []
