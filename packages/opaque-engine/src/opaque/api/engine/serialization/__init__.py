"""NumPy and optree serialization handlers registered against the base registry.

Importing this module registers the portable ``numpy.ndarray`` and
``optree.PyTreeSpec`` handlers with ``opaque.api.base.serialization``.
Torch tensor and parameter handlers are registered when the ``opaque-torch``
provider loads.
"""

from __future__ import annotations

from opaque.api.engine.serialization import _structural  # noqa: F401

__all__: list[str] = []
