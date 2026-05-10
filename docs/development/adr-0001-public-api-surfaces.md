# ADR 0001: Public import surfaces

Where application code should import training primitives from.

## Training packages

| Use case | Import root |
|----------|-------------|
| DP-SGD | `opaque.dpsgd` and submodules `noise`, `sampling`, `clipping`, `accounting` |
| DP-FTRL | `opaque.dpftrl` and submodules `noise`, `sampling`, `clipping`, `accounting` |
| Composition / accountants | `opaque.accounting` |
| DP-SGD accountants | `opaque.dpsgd.accounting` |
| DP-FTRL accountants | `opaque.dpftrl.accounting` |

## Shared utilities

| Concern | Module |
|---------|--------|
| RNG keys | `opaque.random` |
| `torch.func` / model bridges | `opaque.functional` |
| DDP / `sync` | `opaque.distributed` |
| Checkpoints | `opaque.serialization` |
| Step schedules | `opaque.scheduling` |
| Optimizers | `opaque.optimizers` |
| Pipeline types | `opaque.types` |
| PyTree helpers | `opaque.pytree` |

## Clipping

| Setting | Module |
|---------|--------|
| DP-SGD (fixed, AUTO-S, per-group, adaptive) | `opaque.dpsgd.clipping` |
| DP-FTRL (fixed, AUTO-S, per-group) | `opaque.dpftrl.clipping` |

Lower-level helpers and state dataclasses live under `.types` and `.fun` on those same subpackages where applicable.

## Sampling

| Class | Module |
|-------|--------|
| `PoissonSubsampler` | `opaque.dpsgd.sampling` |
| `CyclicPoissonSampler` | `opaque.dpftrl.sampling` |

Constructor keyword for inclusion probability: `sample_rate`.
