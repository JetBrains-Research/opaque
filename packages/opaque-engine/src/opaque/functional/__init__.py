"""Provider-neutral callable helpers."""

from opaque.api.engine.functional import (
    empty_collate,
    with_batch_dim,
)

__all__ = ["empty_collate", "with_batch_dim"]


def __getattr__(name: str):
    # Transitional re-export while downstream packages migrate to the
    # provider wheels; scheduled for removal once the migration completes.
    if name == "make_functional":
        from opaque.torch.functional import make_functional

        return make_functional
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
