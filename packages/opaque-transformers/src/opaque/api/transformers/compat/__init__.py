"""HF/TRL → opaque configuration converters.

This package provides explicit, audited translation from upstream Hugging
Face ``transformers.TrainingArguments`` and TRL ``SFTConfig`` / ``DPOConfig``
into the opaque equivalents — covering renamed fields, semantically
non-equivalent fields (e.g. ``gradient_accumulation_steps`` collapsing into
the logical Poisson batch), and fields opaque does not implement at all
(``deepspeed``, ``fsdp``, ``packing``, …).

The conversion is **one-way** (HF/TRL → opaque). Round-tripping is not
supported: opaque allows configurations (e.g. ``microbatch_size`` > batch)
that have no HF expression, and HF carries fields (DeepSpeed, FSDP) that
have no opaque expression.

Public entry points live on the config classes themselves
(``TrainingArguments.from_hf(...)``, ``SFTConfig.from_trl(...)``,
``DPOConfig.from_trl(...)``); this package holds the machinery.
"""

from ._common import DPOverrides, normalize_dp_overrides
from ._hf import (
    HF_DIRECT_FIELDS,
    HF_DROP_FIELDS,
    HF_REJECTED_FIELDS,
    HF_RENAME_MAP,
    HF_TRANSFORM_MAP,
    convert_hf_training_arguments,
)

__all__ = [
    "DPOverrides",
    "normalize_dp_overrides",
    "HF_DIRECT_FIELDS",
    "HF_DROP_FIELDS",
    "HF_REJECTED_FIELDS",
    "HF_RENAME_MAP",
    "HF_TRANSFORM_MAP",
    "convert_hf_training_arguments",
]
