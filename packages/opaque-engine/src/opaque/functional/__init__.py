"""Functional façade — re-exports from ``opaque.api.engine.functional``."""

from opaque.api.engine.functional import (
    empty_collate,
    make_functional,
    with_batch_dim,
)

__all__ = ["make_functional", "with_batch_dim", "empty_collate"]
