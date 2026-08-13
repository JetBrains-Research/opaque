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

Provider activation registers the portable core, eager process-level
distributed and baseline observability profiles, the optional
`ExecutionProfile` transforms (`compile`, `checkpoint`,
`optimize_saved_activations`), JAX profiler annotations, and native
`jax.Array` serialization. JAX allocator cache clearing and peak reset are
intentionally unavailable. Device memory fields not exposed by
`Device.memory_stats()` remain `None`.

The JAX execution provider implements:

- `compile(fn)` → `jax.jit(fn)`
- `checkpoint(fn)` → `jax.checkpoint(fn)`
- `optimize_saved_activations(fn)` → `jax.checkpoint(fn, policy=offload_dot_with_no_batch_dims("device", "pinned_host"))`