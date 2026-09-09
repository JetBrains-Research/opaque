# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-example MoE router statistics for DP load balancing.

Every function here is written to run inside ``vmap(grad(...))`` on one
example at a time: reductions are out-of-place, the executed route set is
recovered with ``torch.topk`` on the same fp32 softmax the router used, and the
per-expert indicator is a broadcast compare (``F.one_hot``, ``scatter_add_`` and
``bincount`` are not vmap-safe and are never used).

The statistics follow the Switch Transformer load-balancing loss
(Fedus, Zoph, Shazeer, 2021, https://arxiv.org/abs/2101.03961, eqs. 4-6) as
pooled by Hugging Face's ``load_balancing_loss_func``: the executed load
fraction ``h`` and the mean router probability ``P`` are averaged over all
layers and all valid tokens with one common denominator ``L * T_x``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from opaque.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Sequence


def _binary_mask(
    attention_mask: torch.Tensor | None,
    num_tokens: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Flat fp32 0/1 token mask (``attention_mask != 0``; all ones when absent).

    Accepts the collator's ``(T,)`` or ``(1, T)`` row of one example, or the
    ``(B, T)`` batch that matches ``B * T`` flattened router rows. Any non-zero
    entry counts as valid, so an additive ``0 / -1e9`` mask or a mixed-sign mask
    binarises to the same result as its 0/1 form.
    """
    if attention_mask is None:
        return torch.ones(num_tokens, dtype=torch.float32, device=device)
    mask = (attention_mask != 0).reshape(-1).to(torch.float32)
    if mask.shape[0] != num_tokens:
        ConfigurationError.raise_(
            "attention_mask covers "
            f"{mask.shape[0]} tokens but the router logits cover {num_tokens}"
        )
    return mask


def _check_layers(router_logits: Sequence[torch.Tensor], num_layers: int) -> None:
    if len(router_logits) != num_layers:
        ConfigurationError.raise_(
            f"expected router logits for {num_layers} layers, got "
            f"{len(router_logits)}; a duplicated capture would double the load"
        )


def router_load_and_probs(
    router_logits: Sequence[torch.Tensor],
    attention_mask: torch.Tensor | None,
    *,
    top_k: int,
    num_layers: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-example load fractions, mean router probabilities and token count.

    Args:
        router_logits: One ``(T, E)`` (or ``(..., E)``, flattened) logits tensor
            per routed layer, in the order the model captured them.
        attention_mask: Token validity for the same example (``None`` counts
            every position). Non-binary masks are binarised as
            ``attention_mask != 0``.
        top_k: Experts executed per token.
        num_layers: Expected number of captured layers; a mismatch raises
            :class:`~opaque.exceptions.ConfigurationError` (a ``ValueError``)
            so a duplicated capture cannot silently double the released load.

    Returns:
        ``(h_layers, P, T_x)``: ``h_layers`` is the ``(L, E)`` fp32 executed load
        fraction per layer (``sum_e h[l] = top_k`` on a non-empty row), ``P`` the
        ``(E,)`` fp32 mean router probability pooled over layers and valid tokens
        (``sum_e P = 1``), and ``T_x`` the fp32 valid-token count. A fully masked
        row (``T_x = 0``) returns ``h_layers = 0`` and ``P = 0`` explicitly rather
        than the ``0 / 0`` form, so it contributes nothing downstream.
    """
    _check_layers(router_logits, num_layers)
    first = router_logits[0]
    num_experts = first.shape[-1]
    num_tokens = first.reshape(-1, num_experts).shape[0]
    mask = _binary_mask(attention_mask, num_tokens, device=first.device)
    valid_tokens = mask.sum()
    denominator = valid_tokens.clamp(min=1.0)
    expert_ids = torch.arange(num_experts, device=first.device)

    loads = []
    prob_sum = torch.zeros(num_experts, dtype=torch.float32, device=first.device)
    for logits in router_logits:
        z = logits.reshape(-1, num_experts)
        if z.shape[0] != num_tokens:
            ConfigurationError.raise_(
                "router logits disagree on the token count across layers"
            )
        # Same fp32 softmax and the same topk op the router executes, so the
        # recovered route set is the executed one.
        probs = torch.softmax(z.float(), dim=-1)
        indices = torch.topk(probs, top_k, dim=-1).indices
        one_hot = (indices[..., None] == expert_ids).sum(dim=-2).to(torch.float32)
        loads.append((one_hot * mask[:, None]).sum(dim=0) / denominator)
        prob_sum = prob_sum + (probs * mask[:, None]).sum(dim=0)

    h_layers = torch.stack(loads)
    mean_probs = prob_sum / (len(router_logits) * denominator)
    has_tokens = valid_tokens > 0
    h_layers = torch.where(has_tokens, h_layers, torch.zeros_like(h_layers))
    mean_probs = torch.where(has_tokens, mean_probs, torch.zeros_like(mean_probs))
    return h_layers, mean_probs, valid_tokens


def centred_load(
    h_layers: torch.Tensor,
    *,
    top_k: int,
    valid_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """``h - top_k / E`` per layer; sums to zero over experts on a non-empty row.

    Pass ``valid_tokens`` (the ``T_x`` returned by :func:`router_load_and_probs`)
    to return an explicit zero for a fully masked row instead of the
    ``-top_k / E`` that ``h_layers = 0`` would give. Without it, callers weight
    the row by its token count, which is zero there, so the row still
    contributes nothing.
    """
    num_experts = h_layers.shape[-1]
    centred = h_layers - top_k / num_experts
    if valid_tokens is None:
        return centred
    return torch.where(valid_tokens > 0, centred, torch.zeros_like(centred))


def load_balancing_surrogate(
    P: torch.Tensor,
    f_tilde: torch.Tensor,
    token_weight: torch.Tensor | float,
    *,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    """Per-example surrogate ``E * w_x * <f_tilde - top_k / E, P(x)>``.

    Its gradient with respect to the model equals the gradient of the
    Switch-Transformer load-balancing loss evaluated at the public load vector
    ``f_tilde``; since ``sum_e P = 1`` the centring by ``top_k / E`` leaves the
    gradient unchanged and makes the value vanish at balance.
    """
    centred = f_tilde.to(P.dtype) - top_k / num_experts
    return num_experts * token_weight * (centred * P).sum()


def router_z_loss(
    router_logits: Sequence[torch.Tensor],
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Router z-loss ``(1 / (L T_x)) sum_l sum_t m_t (logsumexp_e z)^2``.

    Follows ST-MoE (Zoph et al., 2022, https://arxiv.org/abs/2202.08906,
    eq. 5) pooled over layers and valid tokens. Returns zero for a fully masked
    row.
    """
    first = router_logits[0]
    num_experts = first.shape[-1]
    num_tokens = first.reshape(-1, num_experts).shape[0]
    mask = _binary_mask(attention_mask, num_tokens, device=first.device)
    valid_tokens = mask.sum()
    total = torch.zeros((), dtype=torch.float32, device=first.device)
    for logits in router_logits:
        z = logits.reshape(-1, num_experts).float()
        total = total + (torch.logsumexp(z, dim=-1).square() * mask).sum()
    z_loss = total / (len(router_logits) * valid_tokens.clamp(min=1.0))
    return torch.where(valid_tokens > 0, z_loss, torch.zeros_like(z_loss))


__all__ = [
    "centred_load",
    "load_balancing_surrogate",
    "router_load_and_probs",
    "router_z_loss",
]
