# opaque-mlx

MLX provider for Opaque's backend-neutral engine primitives.

## Install

On Apple Silicon, install the standalone wheel:

```bash
pip install opaque-mlx
```

Or install it with the Opaque bundle:

```bash
pip install "opaque[mlx]"
```

## Usage

MLX arrays select this provider automatically. Explicit selection is available
through `opaque.mlx.mlx_backend()`.