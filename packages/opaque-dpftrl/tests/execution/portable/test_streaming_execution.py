from __future__ import annotations

import numpy as np
import pytest

from opaque.api.dpftrl.noise import _engine as engine
from opaque.api.dpftrl.noise._band_mf import band_mf_strategy
from opaque.api.dpftrl.noise._bisr import bisr_strategy
from opaque.api.dpftrl.noise._plan import toeplitz_execution_plan
from opaque.random import key


def _step_through(backend_case, plan, template, steps):
    state = engine._initial_inner_state(plan, template, backend_case.dtype("float32"))
    outputs = []
    for t in range(steps):
        noise = {
            name: backend_case.array(
                np.random.RandomState(42)
                .randn(*backend_case.to_host(leaf).shape)
                .astype(np.float32)
            )
            for name, leaf in template.items()
        }
        out, state = engine._apply_plan(
            plan,
            noise,
            state,
            step=t,
            target_tree=template,
            stddev=1.0,
            key=key(0),
            compute_dtype=backend_case.dtype("float32"),
        )
        outputs.append((noise, out))
    return outputs, state


class TestStateStaysBounded:
    def test_band_mf_state_is_bands_minus_one_trees(self, backend_case):
        n_steps, bands = 200, 6
        plan = band_mf_strategy(bands=bands).execution_plan(
            n_steps=n_steps, min_sep=1, max_participations=None
        )
        assert plan.strategy_bands == bands
        template = {"w": backend_case.array(np.zeros(32, dtype=np.float32))}
        _, state = _step_through(backend_case, plan, template, 50)
        assert len(state) == bands - 1

    def test_bisr_state_is_bandwidth_minus_one_trees(self, backend_case):
        n_steps, bandwidth = 200, 4
        plan = bisr_strategy(bandwidth=bandwidth).execution_plan(
            n_steps=n_steps, min_sep=1, max_participations=None
        )
        assert plan.inverse_bands <= bandwidth
        template = {"w": backend_case.array(np.zeros(32, dtype=np.float32))}
        _, state = _step_through(backend_case, plan, template, 50)
        assert len(state) == plan.inverse_bands - 1

    def test_state_bytes_bounded_by_bands_times_leaf(self, backend_case):
        n_steps, bands = 400, 9
        coefs = np.zeros(n_steps)
        coefs[:bands] = np.linspace(1.0, 0.1, bands)
        plan = toeplitz_execution_plan(coefs)
        leaf = backend_case.array(np.zeros(1024, dtype=np.float32))
        _, state = _step_through(backend_case, plan, {"w": leaf}, 100)
        # We can just count numpy element size since it represents the same structural cost
        state_bytes = sum(
            backend_case.to_host(t["w"]).size * backend_case.to_host(t["w"]).itemsize
            for t in state
        )
        assert (
            state_bytes
            <= bands
            * backend_case.to_host(leaf).size
            * backend_case.to_host(leaf).itemsize
        )


class TestStreamingMatchesDenseReference:
    @pytest.mark.parametrize("bands", [1, 2, 5])
    def test_banded_recurrence_equals_inverse_convolution(self, backend_case, bands):
        n_steps = 60
        coefs = np.zeros(n_steps)
        coefs[:bands] = 1.0 / (1.0 + np.arange(bands)) ** 1.5
        plan = toeplitz_execution_plan(coefs)
        template = {"w": backend_case.array(np.zeros(16, dtype=np.float32))}
        outputs, _ = _step_through(backend_case, plan, template, n_steps)
        inverse = np.array(plan.inverse_coefficients)
        noises = [backend_case.to_host(noise["w"]) for noise, _ in outputs]
        for t, (_, got) in enumerate(outputs):
            want = (
                sum(float(inverse[j]) * noises[t - j] for j in range(t + 1))
                * plan.column_scales[t]
            )
            backend_case.assert_allclose(got["w"], want, atol=1e-4, rtol=1e-4)

    def test_bisr_convolution_equals_dense_reference(self, backend_case):
        n_steps = 40
        plan = bisr_strategy(bandwidth=3).execution_plan(
            n_steps=n_steps, min_sep=1, max_participations=None
        )
        template = {"w": backend_case.array(np.zeros(16, dtype=np.float32))}
        outputs, _ = _step_through(backend_case, plan, template, n_steps)
        inverse = np.array(plan.inverse_coefficients)
        noises = [backend_case.to_host(noise["w"]) for noise, _ in outputs]
        for t, (_, got) in enumerate(outputs):
            want = (
                sum(float(inverse[j]) * noises[t - j] for j in range(t + 1))
                * plan.column_scales[t]
            )
            backend_case.assert_allclose(got["w"], want, atol=1e-4, rtol=1e-4)
