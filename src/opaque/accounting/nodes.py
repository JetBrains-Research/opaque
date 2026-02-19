"""Backward-compatibility shim: re-exports from composition.

.. deprecated::
    Import from ``opaque.accounting.composition`` instead.
"""

from opaque.accounting.composition import (  # noqa: F401
    CachedProcess,
    Composed,
    Identity,
    Repeated,
)

__all__ = ["Identity", "Composed", "Repeated", "CachedProcess"]
