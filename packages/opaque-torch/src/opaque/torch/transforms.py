"""Introspection of the active ``torch.func`` interpreter stack.

A patch that must behave differently inside a functional transform asks
here rather than probing ``torch._C._functorch`` itself; the private API
moves between releases, so the probe is written once, in the wheel that
owns torch. Power-user surface for patch authors — the provider's own
entry points stay at :mod:`opaque.torch`.
"""

from opaque.api.torch._transforms import under_functorch_transform

__all__ = ["under_functorch_transform"]
