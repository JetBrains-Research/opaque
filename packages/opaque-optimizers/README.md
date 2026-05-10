# opaque-optimizers

Functional torchopt-based optimizers with a DP-aware update surface.

Implementation lives at `opaque.api.optimizers.*`; user-facing façade
at `opaque.optimizers`. The wheel ships:

- `opaque.api.optimizers.{adamw,adam,sgd,radam,adafactor,adagrad,
  adadelta,rmsprop,lion,ademamix,schedule_free}` — optimizer factories.
- `opaque.api.optimizers.types` — state dataclasses for non-trivial
  optimizers.
- `opaque.api.optimizers._chain` — `make_optimizer_chain` (DP-aware
  chain wrapper).
- `opaque.api.optimizers._bias_correction` — `is_per_group`,
  `resolve_noise_variance` helpers.

Depends on `opaque-engine` (for `ClippedPytree` / `NoisedPytree` /
`PerGroup`). Used as an extras-installed dependency from
`opaque-dpsgd[optimizers]` and `opaque-dpftrl[optimizers]`.
