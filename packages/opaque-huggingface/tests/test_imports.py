# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the re-rooted ``opaque.huggingface`` namespace."""

import subprocess
import sys
import textwrap


def test_import_huggingface_facade():
    import opaque.huggingface as hf

    assert hf.__version__
    assert callable(hf.patch_all)
    assert callable(hf.is_patched)
    assert callable(hf.is_kernel_patched)
    assert callable(hf.is_vmap_patched)
    assert callable(hf.patch_lora_model)


def test_import_patches_module():
    from opaque.huggingface import patches

    assert callable(patches.apply_transformers_patches)
    assert callable(patches.is_transformers_patched)


def test_placeholder_subpackages_exist():
    from opaque.huggingface import callbacks, data, integrations, models, trainer

    for mod in (trainer, callbacks, integrations, data, models):
        assert mod.__doc__ is not None


def test_import_does_not_load_transformers():
    """Plain ``import opaque.huggingface`` must not import transformers."""
    code = textwrap.dedent(
        """
        import sys
        import opaque.huggingface  # noqa: F401
        assert "transformers" not in sys.modules, (
            "import opaque.huggingface must not pull transformers "
            "(got: %r)" % sorted(m for m in sys.modules if "transformers" in m)
        )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_compat_namespace_gone():
    import pytest

    with pytest.raises(ModuleNotFoundError):
        __import__("opaque.compat.transformers")
