# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Data collator patches for Poisson sampling compatibility.

Poisson sampling (used for privacy amplification in DP-SGD) can yield empty
batches.  HuggingFace data collators crash on empty input lists because they
index ``examples[0]`` unconditionally.  These patches add empty-input guards
that learn the output structure from the first non-empty call.

Controlled by:
    OPAQUE_SKIP_TRANSFORMERS_DATA_PATCHES: "all" or "collator"
"""

import copy

from opaque._env import parse_skip_env
from opaque.sampling.collate import _empty_like

_TEMPLATE_ATTR = "_opaque_collate_template"


def apply_data_patches() -> None:
    """Patch HuggingFace data collators to handle empty example lists.

    Currently patches:

    - ``DataCollatorForLanguageModeling.torch_call``
    """
    skip = parse_skip_env("OPAQUE_SKIP_TRANSFORMERS_DATA_PATCHES")
    if "all" in skip:
        return

    try:
        from transformers import DataCollatorForLanguageModeling
    except ImportError:
        return

    if "collator" in skip:
        return

    _original_torch_call = DataCollatorForLanguageModeling.torch_call

    def _patched_torch_call(self, examples):
        if not examples:
            template = getattr(self, _TEMPLATE_ATTR, None)
            if template is not None:
                return _empty_like(template)
            # No template yet — fall back to hardcoded empty dict so
            # callers that depend on dict keys still get something usable.
            import torch

            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "labels": torch.empty(0, 0, dtype=torch.long),
            }

        result = _original_torch_call(self, examples)

        if not hasattr(self, _TEMPLATE_ATTR):
            setattr(self, _TEMPLATE_ATTR, copy.deepcopy(result))

        return result

    DataCollatorForLanguageModeling.torch_call = _patched_torch_call
