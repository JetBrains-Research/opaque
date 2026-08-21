# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""One call applies both layers, and one flag reports the result.

Making Hugging Face work under a functional transform requires fixing torch
first, so ``apply_runtime_patches`` forwards its flags to the Torch provider
before applying the Hugging Face layer. A caller needs the one call.

The Torch-core checkpoint patches used to have two orchestrators — one here, one
in the provider — each with its own ``_is_checkpoint_patched`` module global.
Whichever ran, the other's probe reported ``False``, so a caller could not ask
whether the process was patched and get a true answer.

These run in subprocesses: the patches are process-global and idempotent, so a
probe is only meaningful in an interpreter that has not been patched yet.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run(body: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_provider_probe_sees_patches_applied_through_this_layer():
    out = _run("""
        import opaque.transformers.patches
        from opaque.torch.checkpoint import is_checkpoint_patched

        before = is_checkpoint_patched()
        opaque.transformers.patches.apply_runtime_patches()
        print(before, is_checkpoint_patched())
    """)
    assert out == "False True"


def test_one_call_applies_both_halves():
    pytest.importorskip("transformers")
    out = _run("""
        import transformers
        import opaque.transformers.patches
        from opaque.torch.checkpoint import is_checkpoint_patched

        model_cls = transformers.modeling_utils.PreTrainedModel
        before = model_cls.enable_input_require_grads
        opaque.transformers.patches.apply_runtime_patches()
        print(
            is_checkpoint_patched(),
            model_cls.enable_input_require_grads is not before,
        )
    """)
    assert out == "True True"


def test_disabling_the_concern_suppresses_both_halves():
    pytest.importorskip("transformers")
    out = _run("""
        import transformers
        import opaque.transformers.patches
        from opaque.torch.checkpoint import is_checkpoint_patched

        model_cls = transformers.modeling_utils.PreTrainedModel
        before = model_cls.enable_input_require_grads
        opaque.transformers.patches.apply_runtime_patches(vmap_checkpointing=False)
        print(
            is_checkpoint_patched(),
            model_cls.enable_input_require_grads is not before,
        )
    """)
    assert out == "False False"


def test_hugging_face_glue_is_idempotent():
    """Repeated application must not stack wrappers on ``PreTrainedModel``."""
    pytest.importorskip("transformers")
    out = _run("""
        import transformers
        import opaque.transformers.patches
        from opaque.api.transformers.patches.runtime.checkpoint import (
            apply_checkpoint_patches,
        )

        model_cls = transformers.modeling_utils.PreTrainedModel
        opaque.transformers.patches.apply_runtime_patches()
        once = (
            model_cls.enable_input_require_grads,
            model_cls.gradient_checkpointing_enable,
        )
        apply_checkpoint_patches()
        twice = (
            model_cls.enable_input_require_grads,
            model_cls.gradient_checkpointing_enable,
        )
        print(once == twice)
    """)
    assert out == "True"
