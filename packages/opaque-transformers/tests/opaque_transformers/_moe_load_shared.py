# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the router-load release tests.

Builds a tiny patched Mellum whose router logits are obtained under ``vmap``
by calling the backbone with ``output_router_logits=True`` and computing the
causal-LM cross-entropy here, so the tests do not depend on the chunked
forward's ``opaque_router_logits`` kwarg.

The statistics module ``opaque.api.patches.transformers.components.moe_stats``
is the contract the helper composes.  When it is not importable (the
``opaque-patches`` side has not landed yet) :func:`ensure_moe_stats` installs
a contract-faithful shim under that module name so the helper tests run
against the same statistics the real module must implement; once the real
module exists the shim is never installed.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch
from torch import nn

from opaque.api.transformers.moe_load import PROBE_NAME, attach_probe
from opaque.functional import make_functional

MOE_STATS_MODULE = "opaque.api.patches.transformers.components.moe_stats"

NUM_EXPERTS = 8
TOP_K = 2
NUM_LAYERS = 2
T_MAX = 16
TINY_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": NUM_LAYERS,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 128,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "rope_theta": 10000.0,
    "num_experts": NUM_EXPERTS,
    "num_experts_per_tok": TOP_K,
    "moe_intermediate_size": 32,
}
_TRAINABLE_SUBSTRINGS = ("q_proj", "k_proj", "v_proj", "o_proj", ".mlp.gate.")


# ---------------------------------------------------------------------------
# Contract shim for the statistics module (installed only when absent)
# ---------------------------------------------------------------------------


def _shim_router_load_and_probs(router_logits, attention_mask, *, top_k, num_layers):
    if len(router_logits) != num_layers:
        raise ValueError(
            f"expected {num_layers} router logit tensors, got {len(router_logits)}"
        )
    num_experts = router_logits[0].shape[-1]
    n_tokens = router_logits[0].reshape(-1, num_experts).shape[0]
    if attention_mask is None:
        mask = torch.ones(n_tokens, dtype=torch.float32, device=router_logits[0].device)
    else:
        mask = (attention_mask.reshape(-1) != 0).to(torch.float32)
    count = mask.sum()
    valid = count > 0
    denom = torch.where(valid, count, torch.ones_like(count))
    experts = torch.arange(num_experts, device=mask.device)
    h_layers = []
    probs = torch.zeros(num_experts, dtype=torch.float32, device=mask.device)
    for z in router_logits:
        z = z.reshape(-1, num_experts)
        p = torch.softmax(z.float(), dim=-1)
        idx = torch.topk(p, top_k, dim=-1).indices
        onehot = (idx[..., None] == experts).sum(-2).float()
        h_layers.append((onehot * mask[:, None]).sum(0) / denom)
        probs = probs + (p * mask[:, None]).sum(0)
    h = torch.stack(h_layers)
    probs = probs / (len(router_logits) * denom)
    h = torch.where(valid, h, torch.zeros_like(h))
    probs = torch.where(valid, probs, torch.zeros_like(probs))
    return h, probs, count


def _shim_centred_load(h_layers, *, top_k):
    return h_layers - top_k / h_layers.shape[-1]


def _shim_load_balancing_surrogate(probs, f_tilde, token_weight, *, num_experts, top_k):
    return num_experts * token_weight * ((f_tilde - top_k / num_experts) * probs).sum()


def _shim_router_z_loss(router_logits, attention_mask):
    num_experts = router_logits[0].shape[-1]
    n_tokens = router_logits[0].reshape(-1, num_experts).shape[0]
    if attention_mask is None:
        mask = torch.ones(n_tokens, dtype=torch.float32, device=router_logits[0].device)
    else:
        mask = (attention_mask.reshape(-1) != 0).to(torch.float32)
    count = mask.sum()
    total = torch.zeros((), dtype=torch.float32, device=mask.device)
    for z in router_logits:
        lse = torch.logsumexp(z.reshape(-1, num_experts).float(), dim=-1)
        total = total + (mask * lse.square()).sum()
    denom = torch.where(count > 0, count, torch.ones_like(count))
    return torch.where(count > 0, total / (len(router_logits) * denom), 0.0 * total)


def ensure_moe_stats() -> types.ModuleType:
    """Import the statistics contract module, installing the shim if absent."""
    try:
        return importlib.import_module(MOE_STATS_MODULE)
    except ImportError:
        shim = types.ModuleType(MOE_STATS_MODULE)
        shim.router_load_and_probs = _shim_router_load_and_probs
        shim.centred_load = _shim_centred_load
        shim.load_balancing_surrogate = _shim_load_balancing_surrogate
        shim.router_z_loss = _shim_router_z_loss
        shim.__opaque_test_shim__ = True
        sys.modules[MOE_STATS_MODULE] = shim
        return shim


# ---------------------------------------------------------------------------
# Tiny Mellum with router logits under vmap
# ---------------------------------------------------------------------------


def token_mean_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-row token-mean causal-LM cross-entropy, ``(B, T, V)`` -> ``(B,)``."""
    shifted = logits[:, :-1]
    targets = labels[:, 1:]
    valid = targets != -100
    logp = torch.log_softmax(shifted, dim=-1)
    nll = -logp.gather(-1, targets.clamp(min=0)[..., None]).squeeze(-1)
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    return nll.sum(-1) / valid.sum(-1).clamp(min=1)


class RouterLogitsLM(nn.Module):
    """Causal LM forward that returns ``(ce_per_row, router_logits)``.

    Calls the backbone with ``output_router_logits=True`` (HF records the
    router logits through its output recorder) and computes the token-mean
    CE on the LM head here, bypassing the HF aux-loss path.
    """

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, input_ids, attention_mask, labels):
        out = self.lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_router_logits=True,
        )
        logits = self.lm.lm_head(out[0])
        return token_mean_ce(logits, labels), out.router_logits


def _patches_models_dir() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "opaque-patches"
        / "tests"
        / "transformers"
        / "models"
    )


def build_tiny_mellum(
    *, seed: int = 0, router_scale: float = 1.0, dtype: torch.dtype = torch.float32
):
    """Build the wrapped tiny Mellum with the probe attached.

    Returns ``(wrapper, modeling_module, fmodel, trainable, frozen)``.  Only
    the attention projections and the router are trainable, plus the probe
    registered on the wrapper (so its key is exactly ``PROBE_NAME``).
    ``router_scale`` multiplies the first two router rows of every layer to
    induce an imbalance.
    """
    models_dir = str(_patches_models_dir())
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    from _test_utils import build_moe_model

    torch.manual_seed(seed)
    model, mod = build_moe_model("mellum", "cpu", **TINY_CONFIG)
    model.train()
    for name, p in model.named_parameters():
        p.requires_grad_(any(s in name for s in _TRAINABLE_SUBSTRINGS))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".mlp.gate." in name:
                p[:2] *= router_scale
    wrapper = RouterLogitsLM(model).to(dtype)
    attach_probe(wrapper, num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS)
    fmodel, trainable, frozen = make_functional(
        wrapper, disable_autograd_tracking=True, partition_trainable=True
    )
    assert PROBE_NAME in trainable
    return wrapper, mod, fmodel, trainable, frozen


def make_data(
    n: int, *, seed: int, t_max: int = T_MAX, min_len: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-padded ragged rows: ``(input_ids, attention_mask, labels)``."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, TINY_CONFIG["vocab_size"], (n, t_max), generator=g)
    low = t_max // 2 if min_len is None else min_len
    lengths = torch.randint(low, t_max + 1, (n,), generator=g)
    mask = (torch.arange(t_max)[None] < lengths[:, None]).long()
    ids = torch.where(mask.bool(), ids, torch.zeros_like(ids))
    labels = torch.where(mask.bool(), ids, torch.full_like(ids, -100))
    return ids, mask, labels


def make_loss_fn(fmodel, frozen, ctx: dict, *, include_ce: bool = True):
    """Per-example loss ``(trainable, ids, mask, labels) -> (loss, stats)``.

    ``ctx`` carries the public per-step constants ``f_tilde``, ``alpha``,
    ``lam`` and ``mean_tokens`` (the closure-tensor delivery the trainer
    uses).  The aux payload is ``(h_layers, P, T_x)`` for test assertions
    only; it never leaves the clipped-gradient transform in a real loop.
    """
    from opaque.api.transformers.moe_load import router_load_terms

    stats = ensure_moe_stats()

    def loss_fn(trainable, ids, mask, labels):
        params = {**frozen, **trainable}
        ce, router_logits = fmodel(params, ids[None], mask[None], labels[None])
        ce = ce[0] if include_ce else ce[0] * 0.0
        loss = router_load_terms(
            ce,
            router_logits,
            mask,
            trainable,
            f_tilde=ctx["f_tilde"],
            alpha=ctx["alpha"],
            lam=ctx["lam"],
            top_k=TOP_K,
            num_layers=NUM_LAYERS,
            num_experts=NUM_EXPERTS,
            mean_tokens=ctx["mean_tokens"],
        )
        h_layers, probs, count = stats.router_load_and_probs(
            router_logits, mask, top_k=TOP_K, num_layers=NUM_LAYERS
        )
        return loss, (h_layers, probs, count)

    return loss_fn


def token_weighted_load(h_layers: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """HF's pooled batch load ``f(B) = sum_x (T_x / T_tot) h(x)``."""
    return (h_layers.mean(1) * counts[:, None]).sum(0) / counts.sum()
