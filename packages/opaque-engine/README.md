# opaque-engine

Backend-neutral execution substrate for the Opaque library. Framework
implementations are supplied by provider wheels such as `opaque-torch`,
`opaque-jax`, and `opaque-mlx`.

- `opaque.api.engine.types` — `ClippedPytree`, `NoisedPytree`,
  `PerGroup`, `ClipState`, `NoiseState`, paired second-moment outputs.
- `opaque.api.engine.pytree` — provider-dispatched pytree operations plus
  portable `partition`, `merge`, and `global_norm` compositions.
- `opaque.api.engine.random` — immutable `RngKey`, `key`, `split`, and
  `fold_in` helpers plus provider-dispatched sampling.
- `opaque.api.engine.serialization` — NumPy and optree handler registration
  with the base-side serialization registry.
- `opaque.api.engine.distributed` — provider-neutral collective and sync
  helpers.
- `opaque.api.engine.noise_allocation` — per-group / paired-stream
  noise stddev math, shared between DP-SGD and DP-FTRL.
- `opaque.api.engine.clipping` — fixed + AUTO-S clipping primitives
  (constant-sensitivity; usable by both DP-SGD and DP-FTRL).
- `opaque.api.engine.functional` — functional model and batch helpers.
- `opaque.api.engine.scheduling` — step-indexed schedules + warmup /
  restarts composition.
- `opaque.api.engine.profiling` — portable profiling interfaces.

User-facing façades live at `opaque.types`, `opaque.pytree`,
`opaque.random`, `opaque.distributed`, `opaque.functional`,
`opaque.scheduling`, `opaque.profiling`. **No `opaque.clipping`
façade** — clipping is reached via stack façades
(`opaque.dpsgd.clipping`, `opaque.dpftrl.clipping`).
