"""Worker process for the MLX distributed runtime smoke test."""

from __future__ import annotations


def main() -> None:
    import mlx.core as mx

    from opaque.api.engine import runtime
    from opaque.api.engine.backend import use_backend
    from opaque.distributed import all_reduce, gather_for_metrics, wait_for_everyone
    from opaque.mlx import mlx_backend

    with use_backend(mlx_backend()):
        rank = runtime.distributed_rank()
        assert runtime.distributed_world_size() == 2
        local = mx.array([rank + 1.0, 4.0 - rank])
        reductions = {
            operation: all_reduce(local, op=operation)
            for operation in ("sum", "mean", "min", "max", "product")
        }
        gathered = gather_for_metrics(mx.arange(rank + 1))
        objects = runtime.distributed_all_gather_object({"rank": rank})
        wait_for_everyone()

    mx.eval(*reductions.values(), gathered)
    expected = {
        "sum": [3.0, 7.0],
        "mean": [1.5, 3.5],
        "min": [1.0, 3.0],
        "max": [2.0, 4.0],
        "product": [2.0, 12.0],
    }
    for operation, result in reductions.items():
        assert result.tolist() == expected[operation]
    assert gathered.tolist() == [0, 0, 1]
    assert objects == [{"rank": 0}, {"rank": 1}]


if __name__ == "__main__":
    main()
