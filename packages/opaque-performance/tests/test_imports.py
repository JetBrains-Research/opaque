# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the re-rooted ``opaque.performance`` namespace."""

import sys


def test_import_performance():
    import opaque.performance as perf

    assert perf.__version__
    assert callable(perf.patch_all)
    assert callable(perf.patch_checkpoint)
    assert callable(perf.unpatch_checkpoint)
    assert callable(perf.is_checkpoint_patched)


def test_import_kernels_does_not_need_transformers():
    sys.modules.pop("transformers", None)
    import opaque.performance.kernels as kernels  # noqa: F401

    assert "transformers" not in sys.modules
    # Public fallback-aware API is available regardless of Triton.
    assert callable(kernels.opaque_swiglu)
    assert callable(kernels.opaque_cross_entropy_loss)


def test_checkpoint_module():
    from opaque.performance.torch import checkpoint

    assert callable(checkpoint.patch_checkpoint)
    assert callable(checkpoint.is_checkpoint_patched)


def test_unpatch_checkpoint_is_not_supported():
    import pytest

    from opaque.performance import unpatch_checkpoint

    with pytest.raises(NotImplementedError):
        unpatch_checkpoint()


def test_compat_namespace_gone():
    import pytest

    with pytest.raises(ModuleNotFoundError):
        __import__("opaque.compat.pytorch")
    with pytest.raises(ModuleNotFoundError):
        __import__("opaque.compat.kernels")
