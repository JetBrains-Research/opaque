"""Engine bootstrap.

Importing any module under ``opaque.api.engine`` (or any of the engine
façades — ``opaque.types``, ``opaque.pytree``, …) triggers this
``__init__`` first, which performs the side-effect imports that wire
engine handlers into the foundation registries:

- ``serialization`` registers ``torch.Tensor`` / ``numpy.ndarray`` as
  exact-type handlers with the ``opaque.api.base.serialization``
  registry. Without this import, ``state_dict(tensor)`` would skip
  tensors as opaque leaves.
"""

from __future__ import annotations

import opaque.api.engine.serialization  # noqa: F401  (registers tensor/ndarray handlers)
from opaque.api.engine.backend.torch._serialization import register_torch_serialization

register_torch_serialization()
