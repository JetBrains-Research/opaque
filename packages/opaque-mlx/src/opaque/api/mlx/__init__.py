"""MLX provider implementation package."""

from opaque.api.engine.distributed.collectives import register_distributed_probe
from opaque.api.mlx.backend import mlx_backend
from opaque.api.mlx.distributed import _process_group_view

try:
    from opaque.api.mlx.backend._serialization import register_mlx_serialization
except ModuleNotFoundError as exc:
    if exc.name != "mlx" and not (exc.name or "").startswith("mlx."):
        raise
else:
    register_mlx_serialization()


register_distributed_probe(_process_group_view, backend_factory=mlx_backend)

__all__ = ["mlx_backend"]
