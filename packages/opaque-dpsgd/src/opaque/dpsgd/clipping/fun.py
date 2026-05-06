"""Power-user DP-SGD clipping building blocks.

Mirrors :mod:`opaque.clipping.fun`: this submodule exposes lower-level
``*_fun`` clipping primitives for users implementing custom per-example
DP-SGD clipping pipelines.  The headline factories
(``adaptive_clipped_grad``, ``auto_clipped_grad``) live in
:mod:`opaque.dpsgd.clipping`.

- :func:`auto_clipped_fun` — AUTO-S smooth scaling on arbitrary
  per-example function outputs (the function-level analogue of
  ``auto_clipped_grad``).

These are stable, supported public API; they live one level deeper to keep
the headline surface concise.
"""

from __future__ import annotations

from opaque.dpsgd.clipping._auto import auto_clipped_fun

__all__ = ["auto_clipped_fun"]
