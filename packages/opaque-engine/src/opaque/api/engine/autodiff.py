"""Portable, backend-dispatched automatic-differentiation transforms."""

from opaque.api.engine.primitive import Primitive

grad_and_value = Primitive("opaque.autodiff.grad_and_value", tier="core")
vmap = Primitive("opaque.autodiff.vmap", tier="core")

__all__ = ["grad_and_value", "vmap"]
