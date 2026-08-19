"""The stable runtime identity supplied by a backend provider.

Backend-dispatched operations are registered independently as canonical
primitives. A provider therefore needs only a stable name and may expose any
number of implementations without expanding this protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """Structural provider identity used for primitive resolution."""

    name: str


__all__ = ["Backend"]
