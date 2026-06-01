"""Regression: ``apply_model_patches`` must propagate ``performance`` /
``compat`` / per-flag kwargs to its auto-triggered ``apply_runtime_patches``
call.

Before the fix, the auto-call was ``apply_runtime_patches()`` with no kwargs,
so ``apply_model_patches(model, compat=False)`` silently enabled all runtime
compat patches (``vmap_masking``, ``empty_batches``, ``vmap_checkpointing``)
anyway — defeating the public-API contract that the group flags imply.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_runtime_flag():
    import opaque.api.patches as patches_init

    saved = patches_init._runtime_patches_applied
    patches_init._runtime_patches_applied = False
    yield
    patches_init._runtime_patches_applied = saved


class _StubCfg:
    model_type = "definitely-not-registered-runtime-propagation-test"


class _StubModel:
    config = _StubCfg()


def test_apply_model_patches_propagates_compat_false_to_runtime(monkeypatch):
    """``compat=False`` on the model entrypoint must reach the runtime entrypoint."""
    import opaque.api.patches as patches_init

    captured: dict = {}

    def fake_apply_runtime_patches(**kwargs):
        captured.update(kwargs)
        patches_init._runtime_patches_applied = True

    monkeypatch.setattr(
        patches_init, "apply_runtime_patches", fake_apply_runtime_patches
    )

    patches_init.apply_model_patches(_StubModel(), compat=False, peft=False)

    assert captured.get("compat") is False
    assert captured.get("performance") is True


def test_apply_model_patches_propagates_performance_false_to_runtime(monkeypatch):
    import opaque.api.patches as patches_init

    captured: dict = {}

    def fake_apply_runtime_patches(**kwargs):
        captured.update(kwargs)
        patches_init._runtime_patches_applied = True

    monkeypatch.setattr(
        patches_init, "apply_runtime_patches", fake_apply_runtime_patches
    )

    patches_init.apply_model_patches(_StubModel(), performance=False, peft=False)

    assert captured.get("performance") is False
    assert captured.get("compat") is True


def test_apply_model_patches_propagates_explicit_runtime_kwargs(monkeypatch):
    """Per-flag overrides (e.g. ``vmap_checkpointing=False``) pass through too."""
    import opaque.api.patches as patches_init

    captured: dict = {}

    def fake_apply_runtime_patches(**kwargs):
        captured.update(kwargs)
        patches_init._runtime_patches_applied = True

    monkeypatch.setattr(
        patches_init, "apply_runtime_patches", fake_apply_runtime_patches
    )

    patches_init.apply_model_patches(
        _StubModel(),
        peft=False,
        vmap_checkpointing=False,
        empty_batches=False,
    )

    assert captured.get("vmap_checkpointing") is False
    assert captured.get("empty_batches") is False


def test_apply_model_patches_compat_false_does_not_enable_runtime_patches(monkeypatch):
    """End-to-end: when compat=False, none of the three runtime sub-patches fire."""
    import opaque.api.patches as patches_init
    from opaque.api.patches.transformers.runtime import (
        collator as collator_runtime,
        masking as masking_runtime,
    )
    from opaque.api.patches.torch import runtime as torch_runtime

    calls: list[str] = []

    monkeypatch.setattr(
        masking_runtime,
        "apply_masking_patches",
        lambda **kw: calls.append("masking"),
    )
    monkeypatch.setattr(
        collator_runtime,
        "apply_collator_patches",
        lambda **kw: calls.append("collator"),
    )
    monkeypatch.setattr(
        torch_runtime,
        "apply_checkpoint_patch",
        lambda **kw: calls.append("checkpoint"),
    )

    patches_init.apply_model_patches(_StubModel(), compat=False, peft=False)

    assert calls == [], (
        f"compat=False should suppress all runtime compat patches; got {calls}"
    )


def test_apply_model_patches_default_compat_enables_runtime_patches(monkeypatch):
    """Sanity: default behavior (compat=True) still applies the runtime patches."""
    import opaque.api.patches as patches_init
    from opaque.api.patches.transformers.runtime import (
        collator as collator_runtime,
        masking as masking_runtime,
    )
    from opaque.api.patches.torch import runtime as torch_runtime

    calls: list[str] = []

    monkeypatch.setattr(
        masking_runtime,
        "apply_masking_patches",
        lambda **kw: calls.append("masking"),
    )
    monkeypatch.setattr(
        collator_runtime,
        "apply_collator_patches",
        lambda **kw: calls.append("collator"),
    )
    monkeypatch.setattr(
        torch_runtime,
        "apply_checkpoint_patch",
        lambda **kw: calls.append("checkpoint"),
    )

    patches_init.apply_model_patches(_StubModel(), peft=False)

    assert set(calls) == {"masking", "collator", "checkpoint"}


# Regression: apply_checkpoint_patch must run all of Patches 1-8 (a stale
# import once aborted it mid-way, silently disabling cpu_offload + gc).


@pytest.fixture
def _reset_checkpoint_flag():
    import opaque.api.patches.torch.runtime as rt

    saved = rt._is_checkpoint_patched
    rt._is_checkpoint_patched = False
    yield rt
    rt._is_checkpoint_patched = saved


def test_set_module_params_import_path_is_valid():
    from opaque.api.engine.functional import _set_module_params  # noqa: F401


def test_checkpoint_patch_applies_all_patches(_reset_checkpoint_flag):
    import torch.autograd.graph as autograd_graph
    import torch.utils.checkpoint as checkpoint

    rt = _reset_checkpoint_flag

    rt.apply_checkpoint_patch(vmap_checkpointing=True)

    assert rt._is_checkpoint_patched is True
    assert autograd_graph.save_on_cpu.__name__ == "_VmapSaveOnCpu"
    assert hasattr(checkpoint._NoopSaveInputs, "vmap")


def test_checkpoint_patch_skipped_when_disabled(_reset_checkpoint_flag):
    rt = _reset_checkpoint_flag
    rt.apply_checkpoint_patch(vmap_checkpointing=False)
    assert rt._is_checkpoint_patched is False
