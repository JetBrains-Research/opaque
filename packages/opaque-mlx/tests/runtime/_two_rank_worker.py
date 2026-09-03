"""Worker process used by the MLX two-rank runtime smoke test."""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

from opaque import ops
from opaque.api.engine import runtime
from opaque.backend import set_backend
from opaque.mlx.distributed import clear_group, initialize


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
    runtime.distributed_barrier("runtime-smoke")
    mx.eval(reduced, gathered)

    print(
        json.dumps(
            {
                "rank": rank,
                "world_size": group.size(),
                "reduced": np.asarray(ops.to_host(reduced)).tolist(),
                "scalar": scalar,
                "gathered": np.asarray(ops.to_host(gathered)).tolist(),
                "objects": objects,
            }
        )
    )
    clear_group()


if __name__ == "__main__":
    main()
