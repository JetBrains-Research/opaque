# opaque-core

Core primitives for the Opaque differential-privacy ecosystem:

- `opaque.random` — functional RNG keys
- `opaque.utils` — pytree ops, `make_functional`, `PerGroup`
- `opaque.clipping` — per-example / per-group / adaptive / auto clipping
- `opaque.sampling` — Poisson, truncated Poisson, balls-in-bins, b-min-sep, cyclic, sequential
- `opaque.distributed` — DDP plumbing
- `opaque.profiling` — memory profiling
- `opaque.noise.types` — shared `NoiseState` base

No DP mechanism is implemented here. Install `opaque-dpsgd` or `opaque-mf` for noise mechanisms.
