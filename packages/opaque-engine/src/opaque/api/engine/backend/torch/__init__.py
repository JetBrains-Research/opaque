"""Torch implementations for Opaque's portable and optional primitives."""

from opaque.api.engine.backend.torch._core import register_core_primitives
from opaque.api.engine.backend.torch._runtime import register_runtime_primitives

__all__ = ["register_core_primitives", "register_runtime_primitives"]
