"""Opaque core: internal shared primitives.

Not intended for direct user import — everything in here is plumbing used by
the public packages (``opaque.clipping``, ``opaque.random``,
``opaque.functional``, ``opaque.distributed``, ``opaque.dpsgd``,
``opaque.dpftrl``, …). User-facing modules live at the namespace root.

Contents:

- :mod:`opaque.core.noise` — second-moment math + noise distributed-sync
  helpers shared by DP-SGD and DP-FTRL mechanisms.
- :mod:`opaque.core.pytree` — pytree helpers (``tree_map``, ``global_norm``,
  ``partition``, ``merge``, …). Used internally by clipping and
  distributed; users should not need this directly.
"""

# Intentionally empty: no user-facing re-exports. Full dotted paths only.
__all__: list[str] = []
