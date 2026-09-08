# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the legacy checkpoint reparameterization backport."""

from __future__ import annotations

import pytest
import torch
import torch.nn.utils.stateless as stateless
from torch.utils.checkpoint import _CheckpointFrame

from opaque.api.patches.torch.checkpoint import reparametrize_recompute as patch


def _frame(recompute_fn):
    return _CheckpointFrame(
        recompute_fn,
        early_stop=True,
        unpack_error_cb=None,
        metadata_fn=lambda value: value,
    )


@pytest.fixture
def legacy_reparametrize_patch():
    """Force-install the backport and restore every global it changes."""
    before_reparametrize = stateless._reparametrize_module
    before_frame_init = _CheckpointFrame.__init__
    before_orig_reparametrize = patch._orig_reparametrize
    had_stack = hasattr(patch._active, "stack")
    before_stack = getattr(patch._active, "stack", None)

    already_installed = before_reparametrize.__module__ == patch.__name__
    try:
        if not already_installed:
            patch.apply()
        yield
    finally:
        stateless._reparametrize_module = before_reparametrize
        _CheckpointFrame.__init__ = before_frame_init
        patch._orig_reparametrize = before_orig_reparametrize
        if had_stack:
            patch._active.stack = before_stack
        elif hasattr(patch._active, "stack"):
            del patch._active.stack


def test_frames_capture_same_immutable_snapshot(legacy_reparametrize_patch):
    first = torch.nn.Linear(2, 2, bias=False)
    second = torch.nn.Linear(2, 2, bias=False)
    empty = patch._stack()

    assert isinstance(empty, tuple)
    assert patch._stack() is empty

    with stateless._reparametrize_module(first, {"weight": torch.ones(2, 2)}):
        one = patch._stack()
        assert isinstance(one, tuple)
        assert len(one) == 1

        with stateless._reparametrize_module(
            second, {"weight": torch.full((2, 2), 2.0)}
        ):
            nested = patch._stack()
            first_frame = _frame(lambda: None)
            second_frame = _frame(lambda: None)

            assert first_frame.recompute_fn.__kwdefaults__["_snapshot"] is nested
            assert second_frame.recompute_fn.__kwdefaults__["_snapshot"] is nested
            assert patch._stack() is nested

        assert patch._stack() is one

    assert patch._stack() is empty


def test_single_reparameterization_recompute_avoids_exit_stack(
    legacy_reparametrize_patch, monkeypatch
):
    module = torch.nn.Linear(2, 2, bias=False)
    original = module.weight
    replacement = torch.full((2, 2), 3.0)

    def fail_exit_stack():
        raise AssertionError("single reparameterization must not allocate ExitStack")

    monkeypatch.setattr(patch.contextlib, "ExitStack", fail_exit_stack)
    with stateless._reparametrize_module(module, {"weight": replacement}):
        frame = _frame(lambda: module.weight.detach().clone())

    assert module.weight is original
    torch.testing.assert_close(frame.recompute_fn(), replacement)
    assert module.weight is original


def test_nested_recompute_restores_modules_after_exception(legacy_reparametrize_patch):
    first = torch.nn.Linear(2, 2, bias=False)
    second = torch.nn.Linear(2, 2, bias=False)
    first_original = first.weight
    second_original = second.weight
    first_replacement = torch.full((2, 2), 4.0)
    second_replacement = torch.full((2, 2), 5.0)

    def fail_during_recompute():
        torch.testing.assert_close(first.weight, first_replacement)
        torch.testing.assert_close(second.weight, second_replacement)
        raise RuntimeError("recompute failed")

    with (
        stateless._reparametrize_module(first, {"weight": first_replacement}),
        stateless._reparametrize_module(second, {"weight": second_replacement}),
    ):
        frame = _frame(fail_during_recompute)

    with pytest.raises(RuntimeError, match="recompute failed"):
        frame.recompute_fn()

    assert first.weight is first_original
    assert second.weight is second_original
    assert patch._stack() == ()


def test_tied_parameters_are_rebound_during_recompute(legacy_reparametrize_patch):
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(2, 2, bias=False)
            self.second = torch.nn.Linear(2, 2, bias=False)
            self.second.weight = self.first.weight

    module = Tied()
    original = module.first.weight
    replacement = torch.full((2, 2), 6.0)

    def observe_weights():
        return (
            module.first.weight.detach().clone(),
            module.second.weight.detach().clone(),
            module.first.weight is module.second.weight,
        )

    with stateless._reparametrize_module(
        module, {"first.weight": replacement}, tie_weights=True
    ):
        frame = _frame(observe_weights)

    first_weight, second_weight, are_tied = frame.recompute_fn()
    torch.testing.assert_close(first_weight, replacement)
    torch.testing.assert_close(second_weight, replacement)
    assert are_tied
    assert module.first.weight is original
    assert module.second.weight is original
