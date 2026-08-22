"""Pytree types — the structural leaf-path alias.

``ParamPath`` is what :func:`opaque.pytree.param_path` returns and what
``per_group`` groupings are keyed by; it lives here for type annotations,
matching :mod:`opaque.random.types`.
"""

from opaque.api.engine.pytree import ParamPath

__all__ = ["ParamPath"]
