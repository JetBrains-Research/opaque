"""Worker process for the JAX distributed runtime smoke test."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--rank", required=True, type=int)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.distributed.initialize(
        coordinator_address=args.coordinator,
        num_processes=2,
        process_id=args.rank,
        local_device_ids=[0],
    )
    try:
        from opaque.api.engine import runtime
        from opaque.api.engine.backend import use_backend
        from opaque.distributed import all_reduce, gather_for_metrics, wait_for_everyone
        from opaque.jax import jax_backend

        with use_backend(jax_backend()):
            assert runtime.distributed_rank() == args.rank
            assert runtime.distributed_world_size() == 2
            local = jnp.array([args.rank + 1.0, 4.0 - args.rank])
            reductions = {
                operation: all_reduce(local, op=operation)
                for operation in ("sum", "mean", "min", "max", "product")
            }
            gathered = gather_for_metrics(jnp.arange(args.rank + 1))
            objects = runtime.distributed_all_gather_object({"rank": args.rank})
            wait_for_everyone()

        expected = {
            "sum": [3.0, 7.0],
            "mean": [1.5, 3.5],
            "min": [1.0, 3.0],
            "max": [2.0, 4.0],
            "product": [2.0, 12.0],
        }
        for operation, result in reductions.items():
            np.testing.assert_array_equal(result, expected[operation])
        np.testing.assert_array_equal(gathered, [0, 0, 1])
        assert objects == [{"rank": 0}, {"rank": 1}]
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
