"""Functional optimizers façade — re-exports from ``opaque.api.optimizers``.

See :mod:`opaque.api.optimizers` for the canonical implementation.
``opaque.optimizers.types`` mirrors the impl tree's ``types`` module
for callers that import the types submodule by name.
"""

from opaque.api.optimizers import (
    adadelta,
    adafactor,
    adagrad,
    adam,
    adamw,
    ademamix,
    lion,
    radam,
    rmsprop,
    schedule_free,
    sgd,
)
from opaque.optimizers import types  # noqa: F401  (submodule re-export)

__all__ = [
    "adam",
    "adamw",
    "sgd",
    "lion",
    "ademamix",
    "adafactor",
    "rmsprop",
    "adagrad",
    "adadelta",
    "radam",
    "schedule_free",
    "types",
]
