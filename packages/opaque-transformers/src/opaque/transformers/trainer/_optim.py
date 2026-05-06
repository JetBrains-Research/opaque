"""Optimizer resolution for DPTrainer.

Two-layer surface, both routed through the same builder:

1. **Canonical opaque names** — every factory in
   :mod:`opaque.optimizers` (``adam``, ``adamw``, ``sgd``, ``rmsprop``,
   ``adagrad``, ``adafactor``, ``ademamix``, ``lion``, ``schedule_free``).
   The full opaque parameter surface is reachable through HF-canonical
   ``TrainingArguments`` fields (``learning_rate``, ``weight_decay``,
   ``adam_beta1``, ``adam_beta2``, ``adam_epsilon``) plus DP-specific
   ``dp_*`` fields (``dp_noise_bias_correction``,
   ``dp_decoupled_weight_decay``, ``dp_update_rms_clip``).  Anything
   else flows through ``optim_args``; opaque factories raise
   ``TypeError`` on unknown keys.

2. **HF compat aliases** — HF's ``OptimizerNames`` values that map
   cleanly onto an opaque factory (``adamw_torch`` → ``adamw``,
   ``adafactor`` → ``adafactor``, ``lion_32bit`` → ``lion``, …).  These
   accept the same HF / dp_ fields and route to the same factories.
   Aliases are silently rewritten — DPTrainer never substitutes a
   different update rule for what the alias names; the HF spelling is
   honoured by selecting the opaque factory whose math matches.

Names that have no DP-aware mapping (8-bit, paged, GaLore, fused
torch.optim, plain ``adadelta``) remain rejected with a redirect message.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import opaque.optimizers as opaque_opt

GradientTransformation = Any  # torchopt.base.GradientTransformation, lazy

# ---------------------------------------------------------------------------
# Layer 1 — canonical opaque names
# ---------------------------------------------------------------------------

_OPAQUE_FACTORIES: dict[str, Callable[..., GradientTransformation]] = {
    "adam":          opaque_opt.adam,
    "adamw":         opaque_opt.adamw,
    "sgd":           opaque_opt.sgd,
    "lion":          opaque_opt.lion,
    "ademamix":      opaque_opt.ademamix,
    "adafactor":     opaque_opt.adafactor,
    "rmsprop":       opaque_opt.rmsprop,
    "adagrad":       opaque_opt.adagrad,
    "radam":         opaque_opt.radam,
    "schedule_free": opaque_opt.schedule_free,
}

# Per-factory: which top-level / dp_* fields apply, derived from each
# factory's signature.  Adafactor uses its own ``eps_grad`` / ``eps_root``
# pair rather than HF's ``adam_epsilon`` so it's omitted from EPS;
# Adam's signature has no ``decoupled_weight_decay`` (the decoupled
# variant is the separate ``adamw`` factory).
_APPLIES_BETAS = {"adam", "adamw", "ademamix", "lion", "radam"}
_APPLIES_EPS = {"adam", "adamw", "rmsprop", "adagrad", "radam"}
_APPLIES_WEIGHT_DECAY = {
    "adam",
    "adamw",
    "sgd",
    "rmsprop",
    "adagrad",
    "adafactor",
    "ademamix",
    "lion",
    "radam",
}
_APPLIES_DECOUPLED_WD = {
    "adamw",
    "rmsprop",
    "adagrad",
    "adafactor",
    "ademamix",
    "lion",
    "radam",
}
_APPLIES_UPDATE_RMS_CLIP = {"adam", "adamw", "rmsprop", "adafactor", "ademamix", "radam"}
_APPLIES_NOISE_BC = {
    "adam",
    "adamw",
    "rmsprop",
    "adagrad",
    "ademamix",
    "adafactor",
    "radam",
}

# ---------------------------------------------------------------------------
# Layer 2 — HF compat aliases
# ---------------------------------------------------------------------------

# HF name → (opaque factory key, base kwargs).  ``base kwargs`` apply
# only when the corresponding ``_APPLIES_*`` set contains the canonical
# name.  Use the empty dict for plain rewrites.
_HF_ALIASES: dict[str, tuple[str, dict[str, Any]]] = {
    "adamw_torch":       ("adamw", {}),
    "adamw_torch_fused": ("adamw", {}),  # fused kernel ↦ functional path
    "adamw_hf":          ("adamw", {}),
    "adafactor":         ("adafactor", {}),
    "ademamix":          ("ademamix", {}),
    "lion_32bit":        ("lion", {}),
    # ``lion`` itself is a canonical opaque name; HF's enum still
    # surfaces it, route through the canonical entry above.
    "schedule_free_radam": ("schedule_free", {"base": "radam"}),
}

# ---------------------------------------------------------------------------
# Hard rejections — names with no DP-aware equivalent
# ---------------------------------------------------------------------------

# These remain rejected even after the alias layer.  Keep messages
# specific (point at the redirection or explain the impossibility) so
# users know what to switch to.
_DP_OPTIMIZER_UNSUPPORTED: dict[str, str] = {
    # XLA / NPU paths — Opaque vmap targets CUDA/CPU only.
    "adamw_torch_xla": (
        "torch_xla AdamW is not supported (Opaque vmap targets CUDA/CPU); "
        "pass optim='adamw'."
    ),
    "adamw_torch_npu_fused": (
        "Ascend NPU optimizers are not supported; pass optim='adamw'."
    ),
    # Apex / 4-bit / 8-bit AdamW — quantized state cannot be vmapped
    # cleanly.
    "adamw_apex_fused": (
        "APEX fused AdamW is not supported under DP-SGD; pass optim='adamw'."
    ),
    "adamw_anyprecision": (
        "AnyPrecision AdamW is not supported under DP-SGD; pass optim='adamw'."
    ),
    "adamw_bnb_8bit": "8-bit quantized optimizers are not supported under DP-SGD.",
    "adamw_8bit": "8-bit AdamW is not supported under DP-SGD.",
    "adamw_torch_4bit": "4-bit AdamW is not supported under DP-SGD.",
    "adamw_torch_8bit": "8-bit torch.optim.AdamW is not supported under DP-SGD.",
    "ademamix_8bit": "8-bit AdEMAMix is not supported.",
    "lion_8bit": "8-bit Lion is not supported.",
    # bnb / paged variants — quantization / paging interact with DP
    # state in unsupported ways.
    "paged_adamw_32bit": "Paged optimizers (bitsandbytes) are not supported.",
    "paged_adamw_8bit": "Paged 8-bit AdamW is not supported.",
    "paged_ademamix_32bit": "Paged AdEMAMix is not supported.",
    "paged_ademamix_8bit": "Paged 8-bit AdEMAMix is not supported.",
    "paged_lion_32bit": "Paged Lion is not supported.",
    "paged_lion_8bit": "Paged 8-bit Lion is not supported.",
    "rmsprop_bnb": (
        "bitsandbytes RMSprop is not supported; pass optim='rmsprop' "
        "for the opaque RMSprop factory."
    ),
    "rmsprop_bnb_8bit": (
        "8-bit RMSprop (bitsandbytes) is not supported; pass optim='rmsprop'."
    ),
    "rmsprop_bnb_32bit": (
        "bitsandbytes 32-bit RMSprop is not supported; pass optim='rmsprop'."
    ),
    # Out-of-scope optimizer families.
    "galore_adamw": "GaLore optimizers are not supported under DP-SGD.",
    "galore_adamw_8bit": "GaLore 8-bit AdamW is not supported under DP-SGD.",
    "galore_adafactor": "GaLore Adafactor is not supported under DP-SGD.",
    "galore_adamw_layerwise": (
        "GaLore layer-wise AdamW is not supported under DP-SGD."
    ),
    "galore_adamw_8bit_layerwise": (
        "GaLore layer-wise 8-bit AdamW is not supported under DP-SGD."
    ),
    "galore_adafactor_layerwise": (
        "GaLore layer-wise Adafactor is not supported under DP-SGD."
    ),
    "lomo": "LOMO is not supported under DP-SGD.",
    "adalomo": "AdaLOMO is not supported under DP-SGD.",
    "grokadamw": "GrokAdamW is not supported under DP-SGD.",
    # Re-exported torchopt primitives without a DP-aware mode.
    "adadelta": (
        "DPTrainer does not currently support adadelta — its two-EMA "
        "structure has no published DP bias correction.  Use "
        "opaque.optimizers.adadelta directly through the functional "
        "API if you accept vanilla behaviour under DP noise."
    ),
    "adamax": (
        "Adamax structurally misbehaves under DP: the half-normal noise "
        "mean is permanently absorbed by the max-norm denominator.  "
        "Pass optim='adamw' instead."
    ),
    "apollo_adamw": "APOLLO AdamW is not supported under DP-SGD.",
    "apollo_adamw_layerwise": (
        "APOLLO layer-wise AdamW is not supported under DP-SGD."
    ),
    "stable_adamw": (
        "StableAdamW maps onto opaque.optimizers.adamw with "
        "dp_update_rms_clip=1.0; pass optim='adamw' with that field set."
    ),
    "schedule_free_adamw": (
        "schedule_free_adamw maps onto opaque.optimizers.schedule_free; "
        "pass optim='schedule_free' with optim_args='base=adamw'."
    ),
    "schedule_free_sgd": (
        "schedule_free_sgd maps onto opaque.optimizers.schedule_free; "
        "pass optim='schedule_free' with optim_args='base=sgd'."
    ),
}


def supported_names() -> tuple[str, ...]:
    """Sorted union of canonical opaque names + HF aliases."""
    return tuple(sorted(set(_OPAQUE_FACTORIES) | set(_HF_ALIASES)))


def canonical_optimizer_names() -> tuple[str, ...]:
    """Sorted canonical opaque optimizer names (no HF aliases)."""
    return tuple(sorted(_OPAQUE_FACTORIES))


def normalize_optim(optim: Any) -> str:
    """Coerce HF's ``OptimizerNames`` enum to its string value."""
    return optim.value if hasattr(optim, "value") else str(optim)


def resolve_optimizer_name(optim: Any) -> tuple[str, dict[str, Any]]:
    """Resolve user-supplied ``optim`` to ``(canonical_name, base_kwargs)``.

    Raises ``ValueError`` for names in :data:`_DP_OPTIMIZER_UNSUPPORTED`
    with the redirect message attached, or for any name that's neither
    canonical nor an alias.
    """
    name = normalize_optim(optim)
    if name in _OPAQUE_FACTORIES:
        return name, {}
    if name in _HF_ALIASES:
        canonical, base = _HF_ALIASES[name]
        return canonical, dict(base)
    if name in _DP_OPTIMIZER_UNSUPPORTED:
        raise ValueError(
            f"optim={optim!r} is not supported by DPTrainer: "
            f"{_DP_OPTIMIZER_UNSUPPORTED[name]}  "
            f"Supported optimizers: {supported_names()}."
        )
    raise ValueError(
        f"optim={optim!r} is not supported by DPTrainer; "
        f"expected one of {supported_names()}."
    )


def _apply_top_level_fields(
    canonical: str,
    args: Any,
    kwargs: dict[str, Any],
) -> None:
    """Forward HF / dp_ TrainingArguments fields to factory kwargs.

    Mutates ``kwargs`` in place.  Only fields applicable to the chosen
    factory are forwarded; unrelated fields are silently dropped (a
    factory raises TypeError if a user smuggles something via
    ``optim_args``).
    """
    if canonical in _APPLIES_BETAS:
        # ``ademamix`` needs a 3-tuple ``(β₁, β₂, β₃)``; the third
        # parameter is ademamix-specific (slow EMA half-life).  HF
        # surfaces only ``adam_beta1`` / ``adam_beta2`` so we default
        # β₃ to the opaque factory's default; users can override the
        # full tuple via ``optim_args``.
        if canonical == "ademamix":
            kwargs.setdefault("betas", (args.adam_beta1, args.adam_beta2, 0.9999))
        else:
            kwargs.setdefault("betas", (args.adam_beta1, args.adam_beta2))
    if canonical in _APPLIES_EPS:
        kwargs.setdefault("eps", args.adam_epsilon)
    if canonical in _APPLIES_WEIGHT_DECAY:
        kwargs.setdefault("weight_decay", args.weight_decay)
    if canonical in _APPLIES_DECOUPLED_WD:
        decoupled = getattr(args, "dp_decoupled_weight_decay", None)
        if decoupled is not None:
            kwargs.setdefault("decoupled_weight_decay", bool(decoupled))
    if canonical in _APPLIES_UPDATE_RMS_CLIP:
        rms = getattr(args, "dp_update_rms_clip", None)
        if rms is not None:
            kwargs.setdefault("update_rms_clip", float(rms))
    if canonical in _APPLIES_NOISE_BC:
        bc = getattr(args, "dp_noise_bias_correction", False)
        kwargs.setdefault("noise_bias_correction", bool(bc))


def _build_schedule_free(
    lr_schedule: Any,
    args: Any,
    extra_kwargs: dict[str, Any],
    base_kwargs: dict[str, Any],
) -> GradientTransformation:
    """Construct a ``schedule_free`` wrapper.

    ``optim_args="base=adamw,..."`` selects the inner factory; the
    remaining kwargs are split between the base factory and the
    schedule-free wrapper based on signature.  Missing ``base`` defaults
    to ``adamw`` (HF's ``schedule_free_adamw`` semantic).
    """
    pooled = {**base_kwargs, **extra_kwargs}
    base_name = pooled.pop("base", "adamw")
    if base_name not in _OPAQUE_FACTORIES or base_name == "schedule_free":
        raise ValueError(
            f"schedule_free base={base_name!r} is not a supported "
            "opaque optimizer; pick from "
            f"{tuple(n for n in _OPAQUE_FACTORIES if n != 'schedule_free')}."
        )
    # schedule_free's own kwargs (e.g. ``beta``, ``weight_lr_power``)
    # vs. the base factory's kwargs are disambiguated by signature
    # introspection; opaque.optimizers.schedule_free raises on unknown
    # keys, so we hand it everything left after building the base.
    base_factory = _OPAQUE_FACTORIES[base_name]
    base_only = {}
    for key in list(pooled):
        # Base-factory-only knobs we know about.  Everything else
        # passes to schedule_free; if either factory doesn't recognise
        # a key, the TypeError surfaces the typo.
        if key in {"betas", "eps", "weight_decay", "decoupled_weight_decay",
                   "update_rms_clip", "noise_bias_correction",
                   "alpha", "momentum", "dampening", "nesterov",
                   "beta1", "decay_rate", "eps_grad", "eps_root"}:
            base_only[key] = pooled.pop(key)
    # Apply HF / dp_ fields to the base factory.
    _apply_top_level_fields(base_name, args, base_only)
    base = base_factory(lr=lr_schedule, **base_only)
    return opaque_opt.schedule_free(base, **pooled)


def build_optimizer(
    args: Any,
    lr_schedule: Any,
    extra_kwargs: dict[str, Any] | None = None,
) -> GradientTransformation:
    """Construct the DP-aware opaque optimizer for ``args.optim``.

    ``extra_kwargs`` typically comes from ``parse_optim_args(args.optim_args)``
    and takes precedence over HF / dp_ field defaults.
    """
    canonical, base_kwargs = resolve_optimizer_name(args.optim)
    extra_kwargs = dict(extra_kwargs or {})
    if canonical == "schedule_free":
        return _build_schedule_free(lr_schedule, args, extra_kwargs, base_kwargs)
    factory = _OPAQUE_FACTORIES[canonical]
    kwargs = {**base_kwargs}
    _apply_top_level_fields(canonical, args, kwargs)
    kwargs.update(extra_kwargs)
    return factory(lr=lr_schedule, **kwargs)


__all__ = [
    "build_optimizer",
    "resolve_optimizer_name",
    "canonical_optimizer_names",
    "supported_names",
    "_DP_OPTIMIZER_UNSUPPORTED",
]
