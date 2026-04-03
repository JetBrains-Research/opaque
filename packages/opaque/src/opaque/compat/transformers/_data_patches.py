# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Data collator patches for Poisson sampling compatibility.

Poisson sampling (used for privacy amplification in DP-SGD) can yield empty
batches.  HuggingFace data collators crash on empty input lists because they
index ``examples[0]`` unconditionally.  These patches add empty-input guards.

Controlled by:
    OPAQUE_SKIP_TRANSFORMERS_DATA_PATCHES: "all" or "collator"
"""

import torch

from opaque._env import parse_skip_env


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
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "labels": torch.empty(0, 0, dtype=torch.long),
            }
        return _original_torch_call(self, examples)

    DataCollatorForLanguageModeling.torch_call = _patched_torch_call
