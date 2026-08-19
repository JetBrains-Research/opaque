"""Portable backend-dispatched automatic-differentiation transforms."""

from opaque.api.engine.autodiff import grad_and_value, vmap

__all__ = ["grad_and_value", "vmap"]
