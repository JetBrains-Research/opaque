# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Pre-split behavior pins for the backend-neutral engine on Torch.

Each test locks in a behavior the monolithic Torch engine provided and
the provider split must preserve: RNG-bearing models under per-example
gradients, serialization before backend activation, backend-free
distributed helpers, and CPU-safe profiling.
"""

import pytest
import torch
import torch.nn as nn

from opaque.api.engine.backend import clear_backend, ensure_backend
from opaque.dpsgd.clipping import clipped_grad
from opaque.torch.functional import make_functional


def test_dropout_works_under_per_example_gradients() -> None:
    """vmap randomness defaults to "same": stochastic models keep working."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5), nn.Linear(8, 2))
    model.train()
    fmodel, params = make_functional(model)
    data = torch.randn(16, 8)
    labels = torch.randint(0, 2, (16,))

    def loss_fn(p, example):
        x, t = example
        return torch.nn.functional.cross_entropy(
            fmodel(p, x.unsqueeze(0)), t.unsqueeze(0)
        )

    grad_fn, state = clipped_grad(
        loss_fn, argnums=0, batch_argnums=1, clipping_norm=1.0, normalize_by=16
    )
    grads, _ = grad_fn(params, (data, labels), state=state)
    for leaf in grads.pytree:
        assert torch.isfinite(leaf).all()


def test_state_dict_resolves_tensors_before_backend_activation() -> None:
    """The serialization fallback activates the provider on first contact."""
    from opaque.serialization import from_state_dict, state_dict

    clear_backend()
    try:
        value = torch.randn(3)
        sd = state_dict({"w": value})
        assert torch.equal(sd["w"], value)
        restored = from_state_dict({"w": torch.zeros(3)}, sd)
        assert torch.equal(restored["w"], value)
    finally:
        clear_backend()
        ensure_backend(torch.empty(0))


def test_distributed_helpers_safe_before_backend_activation() -> None:
    """Rank-0 guards during program setup must not require a backend."""
    from opaque.api.engine.distributed._state import reduce_scalar
    from opaque.distributed import (
        get_rank,
        get_world_size,
        is_distributed,
        is_main_process,
    )

    clear_backend()
    try:
        assert is_distributed() is False
        assert is_main_process() is True
        assert get_rank() == 0
        assert get_world_size() == 1
        assert reduce_scalar(2.5, op="sum") == 2.5
        assert reduce_scalar(2**24 + 1, op="mean") == 2**24 + 1
    finally:
        clear_backend()
        ensure_backend(torch.empty(0))


def test_reset_peak_memory_is_noop_on_cpu() -> None:
    from opaque.profiling import reset_peak_memory

    reset_peak_memory("cpu")


def test_step_perf_to_dict_omits_unknown_metrics_on_cpu() -> None:
    from opaque.profiling import step_perf

    with step_perf("cpu", batch_size=4) as sp:
        pass
    entries = sp.perf.to_dict()
    assert entries
    assert all(value is not None for value in entries.values())


def test_make_functional_is_plain_for_non_hf_kwarg_names() -> None:
    """No implicit HF batch adaptation: kwargs named like HF inputs keep
    their caller-provided ranks unless ``hf_batch_adaptation=True``."""

    class Rank3Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 2)

        def forward(self, input_ids=None):
            assert input_ids.ndim == 3
            return self.proj(input_ids)

    fmodel, params = make_functional(Rank3Model())
    out = fmodel(params, input_ids=torch.randn(2, 5, 4))
    assert out.shape == (2, 5, 2)


def test_in_place_collectives_available_from_torch_wheel() -> None:
    from opaque.torch.distributed import (
        all_reduce_,
        reduce_pytree_,
        sum_gradients_,
    )

    grads = (torch.ones(3), torch.ones(2))
    # Not distributed: in-place reduction is a no-op but must not raise.
    reduce_pytree_(grads)
    sum_gradients_(grads)
    with pytest.raises(RuntimeError, match="not initialized"):
        all_reduce_(torch.ones(1))


def test_pinned_transform_revalidates_after_clear_backend() -> None:
    """A transform wrapper built under Torch re-resolves from its arguments
    once the constructing selection is cleared, instead of silently running
    the transform pinned at construction."""
    from opaque.api.engine import autodiff
    from opaque.api.engine.backend import active_backend

    ensure_backend(torch.empty(0))
    doubled = autodiff.vmap(lambda x: x * 2.0)
    clear_backend()

    out = doubled(torch.arange(4.0))
    torch.testing.assert_close(out, torch.arange(4.0) * 2.0)
    # The eager call went through argument inference, not the stale pin.
    active = active_backend()
    assert active is not None
    assert active.name == "torch"


def test_eager_dispatch_is_context_local() -> None:
    """A context that never selected a backend fails closed on no-argument
    primitives even while another context holds the process's only active
    backend — the compile-time global mirror must not leak into eager."""
    import contextvars

    from opaque.api.engine.backend import BackendNotSelectedError
    from opaque.ops import float32

    clear_backend()
    pristine = contextvars.copy_context()
    ensure_backend(torch.empty(0))
    assert float32() == torch.float32

    with pytest.raises(BackendNotSelectedError):
        pristine.run(float32)


def test_backend_activates_by_name_before_any_provider_import() -> None:
    """``use_backend("torch")`` loads the provider factory on demand.

    Regression: with only engine facades imported, the ``normal`` CORE
    primitive was declared but had no registration, so activating by name
    failed core-profile validation instead of loading the Torch provider.
    """
    import subprocess
    import sys

    script = (
        "import torch\n"
        "from opaque.backend import set_backend, use_backend, clear_backend\n"
        "from opaque.random import key, normal\n"
        "with use_backend('torch'):\n"
        "    sample = normal(key(42), (3,))\n"
        "assert sample.dtype == torch.float32\n"
        "set_backend('torch')\n"
        "sample = normal(key(42), (3,))\n"
        "assert sample.shape == (3,)\n"
        "clear_backend()\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
