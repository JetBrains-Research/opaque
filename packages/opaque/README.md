# opaque

Meta-package for the Opaque functional DP-SGD library for PyTorch.

Installing `opaque` pulls in the default bundle:

- `opaque-core` — RNG, pytree, clipping, sampling, distributed, profiling
- `opaque-dpsgd` — Gaussian / truncated Gaussian noise, per-group, AdamW-BC
- `opaque-auditing` — empirical privacy auditing (one-run, coin-flip, loss attacks)
- `opaque-accounting` — PLD-based privacy accounting (Rust-backed)

Plus the `opaque.accounting` convenience shim (re-exports `opaque_accounting`).

## Extras

```bash
pip install "opaque[mf]"            # + opaque-mf (correlated-noise mechanisms, AdamW-JME)
pip install "opaque[performance]"   # + opaque-performance (fused Triton kernels, checkpoint patches)
pip install "opaque[huggingface]"   # + opaque-huggingface (transformers patches) + performance
pip install "opaque[optimizers]"    # + torchopt wiring for AdamW-BC / AdamW-JME
pip install "opaque[all]"           # everything above
```

## Development

From monorepo root:

```bash
uv sync
uv run pytest
```

Each subpackage can be tested in isolation:

```bash
uv run pytest packages/opaque-core/tests/
uv run pytest packages/opaque-dpsgd/tests/
uv run pytest packages/opaque-mf/tests/
# etc.
```

## Import layout

```
opaque.clipping, opaque.sampling, opaque.random, opaque.utils,
opaque.distributed, opaque.profiling        <- opaque-core
opaque.noise.types                           <- opaque-core
opaque.noise.gaussian, opaque.noise.truncated_gaussian,
opaque.noise.per_group_noise                 <- opaque-dpsgd
opaque.mf.noise.*                            <- opaque-mf
opaque.optimizers.adamw_bc                   <- opaque-dpsgd
opaque.mf.optimizers.adamw_jme                  <- opaque-mf
opaque.auditing.*                            <- opaque-auditing
opaque.compat.kernels, opaque.compat.pytorch <- opaque-performance
opaque.compat.transformers                   <- opaque-huggingface
opaque.accounting                            <- opaque (meta, shim to opaque_accounting)
```

`opaque`, `opaque.noise`, `opaque.optimizers`, `opaque.compat` are
[PEP 420](https://peps.python.org/pep-0420/) namespace packages contributed
to by multiple distributions — there is no top-level `opaque/__init__.py`,
so `from opaque import X` no longer works. Always import from the relevant
submodule (`from opaque.clipping import clipped_grad`).

## HuggingFace auto-patching

Patches used to run on `import opaque`. They are now opt-in:

```python
from opaque.compat.pytorch import apply_pytorch_patches
from opaque.compat.transformers import apply_transformers_patches

apply_pytorch_patches()        # checkpoint patches for vmap
apply_transformers_patches()   # vmap + kernel + KV-cache patches for HF models
```
