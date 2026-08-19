"""Streaming banded MF execution: state stays O(bands), per-step cost flat.

These are the scale gates for the plan executor: a regression that
re-materializes the full noise history (O(n_steps) trees) or convolves
over the whole past (O(step) work) fails here long before it would OOM
real training runs.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from opaque.api.dpftrl.noise import _engine as engine
from opaque.api.dpftrl.noise._band_mf import band_mf_strategy
from opaque.api.dpftrl.noise._bisr import bisr_strategy
from opaque.api.dpftrl.noise._plan import toeplitz_execution_plan
from opaque.backend import clear_backend, set_backend
from opaque.random import key


@pytest.fixture(autouse=True)
def _torch_backend():
    set_backend("torch")
    yield
    clear_backend()


def _step_through(plan, template, steps):
    state = engine._initial_inner_state(plan, template, torch.float32)
    outputs = []
    for t in range(steps):
        noise = {name: torch.randn_like(leaf) for name, leaf in template.items()}
        out, state = engine._apply_plan(
            plan,
            noise,
            state,
            step=t,
            target_tree=template,
            stddev=1.0,
            key=key(0),
            compute_dtype=torch.float32,
        )
        outputs.append((noise, out))
    return outputs, state


class TestStateStaysBounded:
    def test_band_mf_state_is_bands_minus_one_trees(self):
        n_steps, bands = 200, 6
        plan = band_mf_strategy(bands=bands).execution_plan(
            n_steps=n_steps, min_sep=1, max_participations=None
        )
        assert plan.strategy_bands == bands
        template = {"w": torch.zeros(32)}
        _, state = _step_through(plan, template, 50)
        assert len(state) == bands - 1

    def test_bisr_state_is_bandwidth_minus_one_trees(self):
        n_steps, bandwidth = 200, 4
        plan = bisr_strategy(bandwidth=bandwidth).execution_plan(
            n_steps=n_steps, min_sep=1, max_participations=None
        )
        assert plan.inverse_bands <= bandwidth
        template = {"w": torch.zeros(32)}
        _, state = _step_through(plan, template, 50)
        assert len(state) == plan.inverse_bands - 1

    def test_state_bytes_bounded_by_bands_times_leaf(self):
        n_steps, bands = 400, 9
        coefs = np.zeros(n_steps)
        coefs[:bands] = np.linspace(1.0, 0.1, bands)
        plan = toeplitz_execution_plan(coefs)
        leaf = torch.zeros(1024)
        _, state = _step_through(plan, {"w": leaf}, 100)
        state_bytes = sum(t["w"].numel() * t["w"].element_size() for t in state)
        assert state_bytes <= bands * leaf.numel() * leaf.element_size()


class TestStreamingMatchesDenseReference:
    @pytest.mark.parametrize("bands", [1, 2, 5])
    def test_banded_recurrence_equals_inverse_convolution(self, bands):
        # Decaying positive bands, like real optimized MF strategies —
        # their Toeplitz inverses stay bounded, so the recurrence and the
        # dense convolution agree to accumulation roundoff.
        n_steps = 60
        coefs = np.zeros(n_steps)
        coefs[:bands] = 1.0 / (1.0 + np.arange(bands)) ** 1.5
        plan = toeplitz_execution_plan(coefs)
        template = {"w": torch.zeros(16)}
        outputs, _ = _step_through(plan, template, n_steps)
        inverse = np.array(plan.inverse_coefficients)
        noises = [noise["w"] for noise, _ in outputs]
        for t, (_, got) in enumerate(outputs):
            want = (
                sum(float(inverse[j]) * noises[t - j] for j in range(t + 1))
                * plan.column_scales[t]
            )
            torch.testing.assert_close(got["w"], want, atol=1e-4, rtol=1e-4)

    def test_bisr_convolution_equals_dense_reference(self):
        n_steps = 40
        plan = bisr_strategy(bandwidth=3).execution_plan(
            n_steps=n_steps, min_sep=1, max_participations=None
        )
        template = {"w": torch.zeros(16)}
        outputs, _ = _step_through(plan, template, n_steps)
        inverse = np.array(plan.inverse_coefficients)
        noises = [noise["w"] for noise, _ in outputs]
        for t, (_, got) in enumerate(outputs):
            want = (
                sum(float(inverse[j]) * noises[t - j] for j in range(t + 1))
                * plan.column_scales[t]
            )
            torch.testing.assert_close(got["w"], want, atol=1e-4, rtol=1e-4)


class TestPerStepCostFlat:
    def test_late_steps_no_slower_than_early_steps(self):
        """O(bands) execution: step 500 costs the same as step 5.

        The pre-streaming executor did O(step) tree work per call, making
        the tail of a long run ~10x slower than its head; the streaming
        recurrence is flat. The 4x bound tolerates CI timer noise while
        still failing any O(step) regression at this horizon.
        """
        n_steps, bands = 600, 8
        coefs = np.zeros(n_steps)
        coefs[:bands] = np.linspace(1.0, 0.2, bands)
        plan = toeplitz_execution_plan(coefs)
        template = {"w": torch.zeros(2048)}
        state = engine._initial_inner_state(plan, template, torch.float32)
        durations = []
        for t in range(n_steps):
            noise = {"w": torch.randn(2048)}
            start = time.perf_counter()
            _, state = engine._apply_plan(
                plan,
                noise,
                state,
                step=t,
                target_tree=template,
                stddev=1.0,
                key=key(0),
                compute_dtype=torch.float32,
            )
            durations.append(time.perf_counter() - start)
        early = float(np.median(durations[10:110]))
        late = float(np.median(durations[-100:]))
        assert late <= max(4.0 * early, early + 1e-3), (early, late)
