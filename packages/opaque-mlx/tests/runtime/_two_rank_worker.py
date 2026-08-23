"""Worker process used by the MLX two-rank runtime smoke test."""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

from opaque import ops
from opaque.api.engine import runtime
from opaque.backend import set_backend
from opaque.distributed import sum_gradients, sync
from opaque.dpftrl.noise import band_mf_strategy, mf_gaussian_noise
from opaque.dpsgd.noise import gaussian_noise
from opaque.mlx.distributed import clear_group, initialize
from opaque.random import key
from opaque.types import clipped


def main() -> None:
    """Exercise every public runtime collective in a launched MLX rank."""
    group = initialize(backend="ring", strict=True)
    set_backend("mlx")
    rank = group.rank()

    reduced = runtime.distributed_all_reduce(mx.array([rank + 1], dtype=mx.float32))
    scalar = runtime.distributed_all_reduce(
        (2**60) + rank,
        runtime.ReduceOp.SUM,
    )
    gathered = runtime.distributed_all_gather(
        mx.array([[rank, rank + 10]], dtype=mx.int32), axis=1
    )
    objects = runtime.distributed_all_gather_object({"rank": rank})
    local_grads = clipped(
        {"weight": mx.array([rank + 1], dtype=mx.float32)}, max_norm=1.0
    )
    dpsgd_noise, dpsgd_state = gaussian_noise(noise_multiplier=0.0, key=key(71))
    dpsgd_noised, dpsgd_state = dpsgd_noise(local_grads, dpsgd_state)
    dpsgd_total = sum_gradients(dpsgd_noised)
    dpsgd_state = sync(dpsgd_state)

    dpftrl_noise, dpftrl_state = mf_gaussian_noise(
        {"weight": mx.zeros((1,), dtype=mx.float32)},
        band_mf_strategy(bands=2, momentum=0.5),
        n_steps=2,
        noise_multiplier=0.0,
        key=key(73),
    )
    dpftrl_noised, dpftrl_state = dpftrl_noise(local_grads, dpftrl_state)
    dpftrl_total = sum_gradients(dpftrl_noised)
    dpftrl_state = sync(dpftrl_state)
    runtime.distributed_barrier("runtime-smoke")
    mx.eval(reduced, gathered, dpsgd_total.pytree, dpftrl_total.pytree)

    print(
        json.dumps(
            {
                "rank": rank,
                "world_size": group.size(),
                "reduced": np.asarray(ops.to_host(reduced)).tolist(),
                "scalar": scalar,
                "gathered": np.asarray(ops.to_host(gathered)).tolist(),
                "objects": objects,
                "dpsgd_total": np.asarray(
                    ops.to_host(dpsgd_total.pytree["weight"])
                ).tolist(),
                "dpsgd_step": dpsgd_state._step_counter,
                "dpftrl_total": np.asarray(
                    ops.to_host(dpftrl_total.pytree["weight"])
                ).tolist(),
                "dpftrl_step": dpftrl_state._step_counter,
            }
        )
    )
    clear_group()


if __name__ == "__main__":
    main()
