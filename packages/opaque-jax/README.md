# opaque-jax

JAX backend support for Opaque's backend-neutral DP clipping primitives.

## Install

Install the standalone wheel:

```bash
pip install opaque-jax
```

Or install it with the Opaque bundle:

```bash
pip install "opaque[jax]"
```

## Usage

```python
from opaque.backend import set_backend
from opaque.jax import jax_backend

set_backend(jax_backend())
```

Importing `opaque.jax` does not load JAX. The backend is loaded only when
`jax_backend()` is called.