"""Public type definitions for :mod:`opaque.accounting.composition`.

Re-exports the composition-node dataclasses for type annotations and
pattern matching. The factory functions (``compose()``, ``repeat()``,
``cached()``) and the ``*`` / ``|`` operator overloads on
:class:`opaque.accounting.types.DpProcess` are the recommended user
surface.
"""

from __future__ import annotations

from opaque.accounting.composition._cached import CachedProcess
from opaque.accounting.composition._composed import Composed
from opaque.accounting.composition._repeated import Repeated

__all__ = ["Composed", "Repeated", "CachedProcess"]
