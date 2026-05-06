"""Public type definitions for :mod:`opaque.random`.

Re-exports :class:`RngKey` for type annotations. The functional surface
(``key``, ``split``, ``fold_in``, …) lives in the package init.
"""

from __future__ import annotations

from opaque.random._engine import RngKey

__all__ = ["RngKey"]
