# opaque-jax

JAX provider for Opaque's backend-neutral engine primitives.

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

JAX arrays select this provider automatically. Explicit selection is available
through `opaque.jax.jax_backend()`.