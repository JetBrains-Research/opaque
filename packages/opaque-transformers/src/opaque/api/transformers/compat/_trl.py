"""TRL ``SFTConfig`` / ``DPOConfig`` → opaque ``SFTConfig`` / ``DPOConfig``.

Same bucketing model as ``_hf.py``: DIRECT / RENAME / TRANSFORM /
REJECT_IF_SET / DROP_WITH_WARN. TRL configs subclass HF
``TrainingArguments``, so the converter delegates the HF-base subset to
``_hf.convert_hf_training_arguments`` and only handles the TRL-specific
fields here.

The ``trl`` package is an optional dependency
(``pip install opaque[trl]``). The classmethods on opaque's config classes
gate on the import and raise ``ImportError`` with an install hint when
TRL is missing.
"""

from __future__ import annotations

from typing import Any, Callable

from ._common import (
    apply_manifest,
    get_dataclass_field_values,
    normalize_dp_overrides,
)
from ._hf import (
    HF_DIRECT_FIELDS,
    HF_DROP_FIELDS,
    HF_REJECTED_FIELDS,
    HF_RENAME_MAP,
    HF_TRANSFORM_MAP,
)


def _import_trl() -> Any:
    """Import the optional ``trl`` package or raise with a clear hint."""
    try:
        import trl
    except ImportError as e:  # pragma: no cover — gated by extras
        raise ImportError(
            "Converting from TRL configs requires the optional ``trl`` "
            "dependency. Install with ``pip install opaque[trl]`` (or "
            "``pip install 'trl>=1.0,<2.0'`` if you manage deps yourself)."
        ) from e
    return trl


# ===========================================================================
# TRL SFTConfig — TRL-specific fields only (HF base handled separately)
# ===========================================================================

# DIRECT — TRL field name matches opaque, same semantics.
TRL_SFT_DIRECT_FIELDS: frozenset[str] = frozenset(
    {
        "model_init_kwargs",
        "chat_template_path",
        "dataset_text_field",
        "dataset_num_proc",
        "eos_token",
        "max_length",
        "pad_to_multiple_of",
        "completion_only_loss",
        "assistant_only_loss",
        "loss_type",
        # TRL adds activation_offloading on SFTConfig/DPOConfig (not on HF
        # base ``TrainingArguments``); opaque's base TrainingArguments has
        # the same field with the same semantics.
        "activation_offloading",
    }
)


# RENAME — TRL SFT field name differs from opaque.
TRL_SFT_RENAME_MAP: dict[str, str] = {}


# TRANSFORM — TRL SFT field requires a derivation step.
TRL_SFT_TRANSFORM_MAP: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}


# REJECT_IF_SET — TRL has the field but opaque does not implement it.
def _reject_if_truthy(message: str) -> Callable[[Any], str | None]:
    def inner(value: Any) -> str | None:
        return message if value else None

    return inner


def _reject_truncation_mode(value: Any) -> str | None:
    if value == "keep_end":
        return (
            "TRL deprecated ``truncation_mode='keep_end'``; opaque only "
            "supports ``keep_start`` (the TRL 1.x default). Drop this field "
            "or set it to ``'keep_start'``."
        )
    return None


def _reject_pad_token(value: Any) -> str | None:
    if value is not None:
        return (
            "TRL deprecated ``pad_token`` in 1.x (removed in 2.0). Set "
            "``tokenizer.pad_token`` directly on the tokenizer before "
            "constructing the trainer instead of carrying it on the config."
        )
    return None


TRL_SFT_REJECTED_FIELDS: dict[str, Callable[[Any], str | None]] = {
    "truncation_mode": _reject_truncation_mode,
    "packing": _reject_if_truthy(
        "Sequence packing breaks the fixed per-example batch shape DP-SGD's "
        "per-example vmap requires. Disable ``packing`` or use opaque without "
        "the converter."
    ),
    "padding_free": _reject_if_truthy(
        "``padding_free`` flattens the batch and skips the attention mask "
        "DP-SGD's per-example accounting depends on. Not supported."
    ),
    "eval_packing": _reject_if_truthy(
        "Eval-side sequence packing is not supported for the same reason as "
        "``packing``."
    ),
    "shuffle_dataset": _reject_if_truthy(
        "Opaque's Poisson sampler controls ordering; explicit shuffling "
        "would defeat the DP accounting."
    ),
    "pad_token": _reject_pad_token,
}


# DROP_WITH_WARN — TRL has the field but it's irrelevant on opaque path.
TRL_SFT_DROP_FIELDS: dict[str, str] = {
    "dataset_kwargs": (
        "Opaque does not expose a ``datasets.map`` hook; the collator and "
        "tokenizer handle preprocessing."
    ),
    "packing_strategy": (
        "Only meaningful with ``packing=True``, which opaque does not "
        "support; silently dropped."
    ),
}


# ===========================================================================
# TRL DPOConfig — TRL-specific fields only
# ===========================================================================

TRL_DPO_DIRECT_FIELDS: frozenset[str] = frozenset(
    {
        "model_init_kwargs",
        "disable_dropout",
        "dataset_num_proc",
        "max_length",
        "pad_to_multiple_of",
        "precompute_ref_batch_size",
        "beta",
        "label_smoothing",
        "loss_weights",
        "f_divergence_type",
        "f_alpha_divergence_coef",
        "ld_alpha",
        "use_weighting",
        "discopop_tau",
        "sync_ref_model",
        "ref_model_mixup_alpha",
        "ref_model_sync_steps",
        # TRL's DPOConfig also exposes activation_offloading (not on HF
        # base TrainingArguments).
        "activation_offloading",
        # TRL 1.x SimPO / CPO / ORPO head-specific tunables. Opaque has
        # them on its own DPOConfig with the same names and semantics.
        "simpo_gamma",
        "cpo_alpha",
        "orpo_lambda",
    }
)


TRL_DPO_RENAME_MAP: dict[str, str] = {}


_OPAQUE_DPO_LOSS_TYPES = frozenset(
    {
        "sigmoid",
        "hinge",
        "ipo",
        "robust",
        "exo_pair",
        "nca_pair",
        "bco_pair",
        "sppo_hard",
        "apo_zero",
        "apo_down",
        "discopop",
        "sft",
        "sigmoid_norm",
        # CPO / ORPO / SimPO are assembled specially in opaque but
        # appear in ``DPOConfig.loss_type`` as accepted values.
        "cpo",
        "orpo",
        "simpo",
    }
)


def _loss_type_transform(trl: dict[str, Any]) -> dict[str, Any]:
    """Validate every entry in ``loss_type`` is a head opaque implements.

    TRL 1.x added Adversarial Optimal Transport heads (``aot``,
    ``aot_unpaired``) that opaque does not implement. Reject those
    explicitly so the user knows opaque's surface is narrower than
    upstream TRL.
    """
    loss_type = trl.get("loss_type")
    if loss_type is None:
        return {}
    # TRL stores loss_type as list[str] in 1.x; coerce singletons.
    values = [loss_type] if isinstance(loss_type, str) else list(loss_type)
    unsupported = [v for v in values if v not in _OPAQUE_DPO_LOSS_TYPES]
    if unsupported:
        raise ValueError(
            f"trl_dpo_config.loss_type contains unsupported heads: "
            f"{sorted(set(unsupported))}. Opaque implements: "
            f"{sorted(_OPAQUE_DPO_LOSS_TYPES)}. The Adversarial Optimal "
            f"Transport family (``aot``, ``aot_unpaired``) added in TRL 1.x "
            f"is not in opaque."
        )
    return {"loss_type": values}


TRL_DPO_TRANSFORM_MAP: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "loss_type": _loss_type_transform,
}


def _reject_precompute_ref_log_probs(value: Any) -> str | None:
    if value is False:
        # TRL default for `precompute_ref_log_probs` is False — but opaque
        # only supports the precompute-always mode. Reject when the user
        # explicitly sets False AND the loss_type uses a reference.
        # (The runtime/precompute-always check is enforced at trainer
        # construction; here we just surface the spec mismatch.)
        return (
            "Opaque always precomputes reference logps under DP for "
            "static-reference heads; ``precompute_ref_log_probs=False`` "
            "would force a runtime reference forward inside vmap, which is "
            "incompatible. Remove this field or set it to ``True``."
        )
    return None


TRL_DPO_REJECTED_FIELDS: dict[str, Callable[[Any], str | None]] = {
    "truncation_mode": _reject_truncation_mode,
    "padding_free": _reject_if_truthy(
        "``padding_free`` is not supported for DPO — see the SFT rationale."
    ),
    # NB: ``precompute_ref_log_probs`` is not flagged as REJECT here because
    # opaque's default of always-precompute matches TRL's "True" mode for
    # reference-using heads. A False setting in TRL doesn't translate, but
    # the trainer-side runtime check provides the user-facing error.
    "pad_token": _reject_pad_token,
}


TRL_DPO_DROP_FIELDS: dict[str, str] = {
    "precompute_ref_log_probs": (
        "Opaque always precomputes reference logps under DP for static-"
        "reference heads; this TRL flag is silently honored at its True "
        "mode and ignored otherwise."
    ),
}


# ===========================================================================
# Conversion entry points
# ===========================================================================


def _convert_trl_config(
    trl_cfg: Any,
    *,
    trl_direct: frozenset[str],
    trl_rename: dict[str, str],
    trl_transform: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    trl_reject: dict[str, Callable[[Any], str | None]],
    trl_drop: dict[str, str],
    source_label: str,
    strict: bool,
    dp_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Shared TRL → opaque conversion dispatcher.

    Splits the TRL config's fields into the HF-inherited subset (handled
    by the HF converter machinery) and the TRL-specific subset (handled
    by the per-flavor manifest passed in). Then merges DP overrides on
    top.
    """
    import tempfile

    source_values = get_dataclass_field_values(trl_cfg)

    # Construct a baseline TRL instance to detect "user-set vs default".
    baseline_output_dir = source_values.get("output_dir") or tempfile.mkdtemp(
        prefix="opaque_compat_trl_baseline_"
    )
    baseline = type(trl_cfg)(output_dir=baseline_output_dir)
    source_defaults = get_dataclass_field_values(baseline)

    # The HF-base field names are the union of the HF manifest's buckets
    # — those are handled by the HF dispatcher with HF's own buckets.
    hf_field_names = (
        HF_DIRECT_FIELDS
        | HF_RENAME_MAP.keys()
        | HF_TRANSFORM_MAP.keys()
        | HF_REJECTED_FIELDS.keys()
        | HF_DROP_FIELDS.keys()
    )

    hf_values = {k: v for k, v in source_values.items() if k in hf_field_names}
    hf_defaults = {k: v for k, v in source_defaults.items() if k in hf_field_names}
    trl_values = {k: v for k, v in source_values.items() if k not in hf_field_names}
    trl_defaults = {k: v for k, v in source_defaults.items() if k not in hf_field_names}

    # Layer 1: HF base translation.
    hf_converted = apply_manifest(
        source_values=hf_values,
        source_defaults=hf_defaults,
        direct=HF_DIRECT_FIELDS,
        rename=HF_RENAME_MAP,
        transform=HF_TRANSFORM_MAP,
        reject=HF_REJECTED_FIELDS,
        drop=HF_DROP_FIELDS,
        source_label=source_label,
        strict=strict,
    )

    # Layer 2: TRL-specific field translation.
    trl_converted = apply_manifest(
        source_values=trl_values,
        source_defaults=trl_defaults,
        direct=trl_direct,
        rename=trl_rename,
        transform=trl_transform,
        reject=trl_reject,
        drop=trl_drop,
        source_label=source_label,
        strict=strict,
    )

    # Merge HF + TRL + DP overrides (DP wins at conflicts).
    converted: dict[str, Any] = {}
    converted.update(hf_converted)
    converted.update(trl_converted)
    # Performance kernels are off in HF/TRL, on by default in opaque; default
    # them OFF on conversion to match upstream (Liger, handled in the HF
    # manifest, sets True). A name override below can force either way.
    converted.setdefault("use_performance_kernels", False)
    converted.update(dp_overrides)
    return converted


def convert_trl_sft_config(
    trl_cfg: Any,
    *,
    strict: bool = True,
    **dp_overrides: Any,
) -> dict[str, Any]:
    """Translate a ``trl.SFTConfig`` instance into opaque ``SFTConfig`` kwargs."""
    trl = _import_trl()
    if not isinstance(trl_cfg, trl.SFTConfig):
        raise TypeError(
            f"Expected ``trl.SFTConfig`` instance, got {type(trl_cfg).__name__}."
        )

    dp_layer = normalize_dp_overrides(dp_overrides)
    return _convert_trl_config(
        trl_cfg,
        trl_direct=TRL_SFT_DIRECT_FIELDS,
        trl_rename=TRL_SFT_RENAME_MAP,
        trl_transform=TRL_SFT_TRANSFORM_MAP,
        trl_reject=TRL_SFT_REJECTED_FIELDS,
        trl_drop=TRL_SFT_DROP_FIELDS,
        source_label="trl_sft_config",
        strict=strict,
        dp_overrides=dp_layer,
    )


def convert_trl_dpo_config(
    trl_cfg: Any,
    *,
    strict: bool = True,
    **dp_overrides: Any,
) -> dict[str, Any]:
    """Translate a ``trl.DPOConfig`` instance into opaque ``DPOConfig`` kwargs."""
    trl = _import_trl()
    if not isinstance(trl_cfg, trl.DPOConfig):
        raise TypeError(
            f"Expected ``trl.DPOConfig`` instance, got {type(trl_cfg).__name__}."
        )

    dp_layer = normalize_dp_overrides(dp_overrides)
    return _convert_trl_config(
        trl_cfg,
        trl_direct=TRL_DPO_DIRECT_FIELDS,
        trl_rename=TRL_DPO_RENAME_MAP,
        trl_transform=TRL_DPO_TRANSFORM_MAP,
        trl_reject=TRL_DPO_REJECTED_FIELDS,
        trl_drop=TRL_DPO_DROP_FIELDS,
        source_label="trl_dpo_config",
        strict=strict,
        dp_overrides=dp_layer,
    )
