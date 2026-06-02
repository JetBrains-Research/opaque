# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""MPO loss combinator — TRL ``loss_type=list`` weighted blend.

Mixed Preference Optimization (MPO) trains on a weighted sum of several
per-example loss terms simultaneously (e.g. a DPO ``sigmoid`` term plus an
``sft`` regulariser).  TRL exposes this through ``loss_type=["sigmoid",
"sft"]`` with per-term ``loss_weights``; :func:`mpo_combine` is the pure
combinator that performs the blend.

    Reference: TRL ``DPOTrainer`` ``loss_type=list`` blending logic
    (``dpo_trainer.py``); MPO mixed-objective formulation as used by the
    Qwen2-VL / InternVL preference-tuning recipes.

 The combinator is a fixed-weight linear
combination of per-example loss tensors; the output for example *i* depends
only on the per-example loss values for example *i*.  No cross-example
aggregate is introduced.

**vmap-safety:** pure elementwise tensor arithmetic — broadcasting
multiply-add only.  The key-subset check happens on the *static* ``weights``
dict keys (Python strings), never on tensor values, so there is no Python
control flow on tensor data and the body is safe under
``torch.func.vmap(torch.func.grad(...))``.
"""

from __future__ import annotations

import torch

__all__ = ["mpo_combine"]


def mpo_combine(
    losses: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    """Weighted sum of per-example loss tensors (MPO / TRL ``loss_type=list``).

    Computes ``sum_k weights[k] * losses[k]`` over the keys of *weights*.
    Only the terms named in *weights* contribute, so a caller may pass a
    superset of available losses in *losses* and select a subset via
    *weights*.

    Args:
        losses: Mapping from loss name to its per-example loss tensor. All
            selected tensors must broadcast against one another.
        weights: Mapping from loss name to its scalar blend weight. Its keys
            must be a subset of ``losses``'s keys.

    Returns:
        The per-example blended loss tensor, with shape equal to the
        broadcast of the selected loss tensors.

    Raises:
        KeyError: If *weights* names a key that is absent from *losses*.
    """
    missing = weights.keys() - losses.keys()
    if missing:
        raise KeyError(
            f"weights keys {sorted(missing)} are not present in losses "
            f"(available: {sorted(losses)})"
        )

    out: torch.Tensor | None = None
    for name, weight in weights.items():
        term = weight * losses[name]
        out = term if out is None else out + term
    if out is None:
        raise KeyError("weights must select at least one loss term")
    return out
