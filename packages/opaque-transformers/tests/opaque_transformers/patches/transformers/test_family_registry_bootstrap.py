# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Dispatch must find the shipped families however the caller got here.

The built-in families register themselves as a side effect of importing
``opaque.api.transformers.patches.models``. Nothing on the path from
``opaque.transformers.patches.apply_model_patches`` to the router imports that
package, so if registration were left to an importer the router would look up
an empty registry, report the family as unsupported, and silently apply no
model patches — a failure that looks like "opaque did nothing" rather than an
error.

These run in subprocesses: the registry and the patches themselves are
process-global, so a probe only means something in an interpreter where no
other test has already imported the families seam.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("transformers")


def _run(body: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_model_patches_reach_a_family_without_the_families_seam():
    """``apply_model_patches`` alone rebinds the model module's mask builder."""
    out = _run("""
        from opaque.transformers.patches import apply_model_patches
        from transformers import LlamaConfig, LlamaForCausalLM
        from transformers.models.llama import modeling_llama

        before = modeling_llama.create_causal_mask.__module__
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=8,
        )
        apply_model_patches(LlamaForCausalLM(config))
        after = modeling_llama.create_causal_mask.__module__

        print(before.startswith("transformers"), after.startswith("opaque"))
    """)
    assert out == "True True"


def test_supported_families_is_populated_from_a_cold_import():
    """The introspection API reports the built-ins, not an empty registry."""
    out = _run("""
        from opaque.api.transformers.patches._registry import supported_families

        families = supported_families()
        print("llama" in families, len(families) > 1)
    """)
    assert out == "True True"
