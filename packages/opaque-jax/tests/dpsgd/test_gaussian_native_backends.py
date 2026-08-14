"""Native JAX Gaussian noise behavior for DP-SGD."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from opaque import ops
from opaque.api.engine.backend import clear_backend
from opaque.distributed import sync
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key
from opaque.types import PerGroup, SecondMomentClippingOutput, clipped


@pytest.fixture(autouse=True)
def _unselected_backend() -> None:
    clear_backend()
    yield
    clear_backend()


def _all_close(left, right) -> bool:
    return float(ops.scalar_item(ops.sum(ops.square(ops.subtract(left, right))))) == 0.0


def test_gaussian_noise_with_native_jax_arrays() -> None:
    grads = {
        "weight": jnp.zeros(128, dtype=jnp.float32),
        "bias": jnp.zeros(16, dtype=jnp.float32),
    }
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(17))
    output, updated_state = noise_fn(clipped(grads, max_norm=1.0), state)
    assert all(ops.is_array(value) for value in output.pytree.values())
    assert ops.dtype(output.pytree["weight"]) == ops.dtype(grads["weight"])
    assert ops.shape(output.pytree["bias"]) == (16,)
    assert bool(ops.scalar_item(ops.all(ops.isfinite(output.pytree["weight"]))))
    assert updated_state._step_counter == 1
    assert sync(updated_state) is updated_state

    clear_backend()
    replay_fn, replay_state = gaussian_noise(noise_multiplier=1.0, key=key(17))
    replay, replay_updated_state = replay_fn(clipped(grads, max_norm=1.0), replay_state)
    assert _all_close(output.pytree["weight"], replay.pytree["weight"])
    assert replay_updated_state == updated_state

    low, high = -1.5, 2.0
    bounded_fn, bounded_state = gaussian_noise(
        noise_multiplier=1.0, bound=(low, high), key=key(29)
    )
    bounded, _ = bounded_fn(
        clipped(jnp.zeros(4096, dtype=jnp.float32), max_norm=1.0), bounded_state
    )
    assert bool(ops.scalar_item(ops.all(ops.isfinite(bounded.pytree))))
    assert bool(ops.scalar_item(ops.all(ops.greater(bounded.pytree, low - 1e-6))))
    assert bool(
        ops.scalar_item(
            ops.all(ops.greater(ops.subtract(high + 1e-6, bounded.pytree), 0.0))
        )
    )
    assert 0.1 < float(ops.scalar_item(ops.mean(ops.square(bounded.pytree)))) < 1.0

    dtype_fn, dtype_state = gaussian_noise(
        noise_multiplier=1.0, compute_dtype=jnp.float32, key=key(37)
    )
    half = jnp.zeros(128, dtype=jnp.float16)
    dtype_output, _ = dtype_fn(clipped(half, max_norm=1.0), dtype_state)
    assert ops.dtype(dtype_output.pytree) == ops.dtype(half)

    norms = PerGroup(
        groups={"small": "small", "large": "large"}, values={"small": 1.0, "large": 4.0}
    )
    per_group_fn, per_group_state = gaussian_noise(noise_multiplier=1.0, key=key(43))
    per_group, per_group_state = per_group_fn(
        clipped(
            {
                "small": jnp.zeros(4096, dtype=jnp.float32),
                "large": jnp.zeros(4096, dtype=jnp.float32),
            },
            max_norm=norms,
        ),
        per_group_state,
    )
    assert per_group.noise_stddev.values == {
        "small": pytest.approx(math.sqrt(5.0)),
        "large": pytest.approx(math.sqrt(20.0)),
    }
    assert float(
        ops.scalar_item(ops.mean(ops.square(per_group.pytree["large"])))
    ) > 2.5 * float(ops.scalar_item(ops.mean(ops.square(per_group.pytree["small"]))))
    paired, _ = per_group_fn(
        SecondMomentClippingOutput(
            grads=clipped(jnp.zeros(128, dtype=jnp.float32), max_norm=1.0),
            squared_grads=clipped(jnp.zeros(128, dtype=jnp.float32), max_norm=1.0),
        ),
        per_group_state,
    )
    assert not _all_close(paired.noisy_grads.pytree, paired.noisy_squared_grads.pytree)
