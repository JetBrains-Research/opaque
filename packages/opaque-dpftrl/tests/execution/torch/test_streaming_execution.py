"""Torch-native performance tests for DP-FTRL streaming execution."""

from __future__ import annotations

import time

import numpy as np
import torch

from opaque.api.dpftrl.noise import _engine as engine
from opaque.api.dpftrl.noise._plan import toeplitz_execution_plan
from opaque.random import key


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
