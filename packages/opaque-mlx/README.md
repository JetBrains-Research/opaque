# opaque-mlx

MLX backend support for Opaque's backend-neutral DP clipping primitives.

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

```python
from opaque.backend import set_backend
from opaque.mlx import mlx_backend

set_backend(mlx_backend())
```

Importing `opaque.mlx` does not load MLX. The backend is loaded only when
`mlx_backend()` is called.