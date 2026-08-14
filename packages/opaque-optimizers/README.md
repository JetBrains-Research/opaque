# opaque-optimizers

Backend-neutral functional optimizers with a DP-aware update surface.

The implementation lives at `opaque.api.engine.optimizers.*`; the user-facing
façade lives at `opaque.optimizers`. Factories accept parameter pytrees and
return `(step, state)`; use `opaque.optimizers.apply_updates` for signed
updates. The wheel ships:

- `opaque.api.engine.optimizers.{adamw,adam,sgd,radam,adafactor,adagrad,
  adadelta,rmsprop,lion,ademamix,schedule_free}` — optimizer factories.
- `opaque.api.engine.optimizers.types` — state dataclasses for non-trivial
  optimizers.
- `opaque.api.engine.optimizers._chain` — `make_optimizer_chain` (DP-aware
  chain wrapper).
- `opaque.api.engine.optimizers._bias_correction` — `is_per_group`,
  `resolve_noise_variance` helpers.

Depends on `opaque-engine` (for `ClippedPytree` / `NoisedPytree` /
`PerGroup`). It is installed by the `optimizers` extras of
`opaque-dpsgd` and `opaque-dpftrl`.
