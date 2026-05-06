"""Power-user clipping building blocks.

The headline entry point for DP training is :func:`opaque.clipping.clipped_grad`.
This submodule exposes the lower-level primitives it builds on, for users
who need to clip arbitrary per-example outputs or compose custom pipelines:

- :func:`clipped_fun` — clip + sum any per-example function
- :func:`clip_pytree` — clip an already-batched pytree to a fixed L2 norm

These are stable, supported public API; they live one level deeper to keep
the headline surface concise.
"""

from __future__ import annotations

from opaque.clipping._clipped_fun import clipped_fun
from opaque.clipping._pytree import clip_pytree

__all__ = ["clipped_fun", "clip_pytree"]
