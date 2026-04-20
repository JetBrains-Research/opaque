# opaque-dpsgd

Standard DP-SGD primitives:

- `opaque.noise.gaussian` — Gaussian noise injection
- `opaque.noise.truncated_gaussian` — truncated Gaussian for bounded-norm releases
- `opaque.noise.per_group_noise` — per-group stddev calibration
- `opaque.optimizers.adamw_bc` — AdamW with DP bias correction (requires `torchopt`)

Depends on `opaque-core`. Install with `pip install opaque-dpsgd[optimizers]` for AdamW-BC.
