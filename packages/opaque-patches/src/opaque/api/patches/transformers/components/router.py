# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in fp32 top-k router for stacked-expert MoE families.

The stock Hugging Face router computes its logits with ``F.linear`` in the
hidden-state dtype and only the softmax in fp32. Under bf16 the logits carry
exact ties, and the top-k set the aux loss sees (a bf16 softmax) can differ from
the executed one. This module swaps in a router forward that computes the
logits in fp32 (the precision Mellum 2.0 was pretrained with): ties disappear
and the logits handed to the load statistics are the executed ones.

The swap is an instance-level ``types.MethodType`` binding on each router
module, recorded so it can be removed again in-process; the class-level
forward is never touched. It changes the executed routing function on the
small fraction of tokens that sit on a bf16 rounding tie; adapters served
through stock HF run bf16 routes, so the swap is opt-in and off by default.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from opaque.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch.nn as nn

_PREVIOUS_FORWARD_ATTR = "_opaque_fp32_router_previous_forward"
_ROUTER_ATTRS = ("top_k", "num_experts", "norm_topk_prob", "weight")


def _fp32_router_forward(self, hidden_states: torch.Tensor):
    """fp32-logit router forward with the stock ``(logits, scores, indices)`` contract."""
    hidden_dim = self.weight.shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    router_logits = F.linear(hidden_states.float(), self.weight.float())
    router_probs = torch.softmax(router_logits, dim=-1)
    router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
    if self.norm_topk_prob:
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    router_scores = router_top_value.to(hidden_states.dtype)
    return router_logits, router_scores, router_indices


_fp32_router_forward.__opaque_fp32_router__ = True  # type: ignore[attr-defined]


def is_router_module(module: nn.Module) -> bool:
    """Whether ``module`` is a stacked-expert top-k router.

    The single router predicate of the package: a class name containing
    ``"TopKRouter"`` or the stock router attributes (``top_k``,
    ``num_experts``, ``norm_topk_prob``, ``weight``).  The fp32 installer,
    the router-load release and the example loops all count routers with it.
    """
    if "TopKRouter" in type(module).__name__:
        return True
    return all(hasattr(module, attr) for attr in _ROUTER_ATTRS)


def _is_router(module: nn.Module, router_cls: type | None) -> bool:
    if router_cls is not None:
        return type(module) is router_cls
    return is_router_module(module)


def _has_fp32_router(module: nn.Module) -> bool:
    forward = module.__dict__.get("forward")
    return getattr(forward, "__func__", None) is _fp32_router_forward


def install_fp32_router(
    model: nn.Module, *, router_cls: type | None = None
) -> Callable[[], None]:
    """Bind the fp32-logit forward on every router module of ``model``.

    Routers are matched by ``router_cls`` when given, otherwise by
    :func:`is_router_module`. Idempotent per module.

    Returns:
        A callable that removes the swap from ``model`` again.

    Raises:
        ConfigurationError: when no module of ``model`` matches, so a
            requested fp32 router never silently installs nothing.
    """
    installed = 0
    for module in model.modules():
        if not _is_router(module, router_cls):
            continue
        installed += 1
        if _has_fp32_router(module):
            continue
        previous = module.__dict__.get("forward")
        module.__dict__[_PREVIOUS_FORWARD_ATTR] = previous
        module.forward = types.MethodType(_fp32_router_forward, module)
    if installed == 0:
        wanted = router_cls.__name__ if router_cls is not None else "a top-k router"
        raise ConfigurationError(
            *(
                f"install_fp32_router: {type(model).__name__} has no module "
                f"matching {wanted}; the fp32 router (router_fp32=True) has "
                "nothing to install on this model.",
            )
        )
    return lambda: remove_fp32_router(model)


def remove_fp32_router(model: nn.Module) -> None:
    """Undo :func:`install_fp32_router` on ``model`` (no-op when not installed)."""
    for module in model.modules():
        if not _has_fp32_router(module):
            continue
        previous = module.__dict__.pop(_PREVIOUS_FORWARD_ATTR, None)
        if previous is None:
            del module.__dict__["forward"]
        else:
            module.__dict__["forward"] = previous


def has_fp32_router(model: nn.Module) -> bool:
    """``True`` when at least one router module of ``model`` carries the swap."""
    return any(_has_fp32_router(module) for module in model.modules())


__all__ = [
    "has_fp32_router",
    "install_fp32_router",
    "is_router_module",
    "remove_fp32_router",
]
