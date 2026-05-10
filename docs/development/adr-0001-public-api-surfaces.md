# ADR 0001: Public API surfaces by wheel

This document freezes the **user-facing** import contract for Opaque (pre-1.0). Implementation may live under private modules (`opaque._clipping`, etc.); tutorials and application code should follow these roots only.

## Training narratives

| Use case | Primary import roots |
|----------|----------------------|
| DP-SGD training | `opaque.dpsgd` (and submodules `noise`, `sampling`, `clipping`, `accounting`) |
| DP-FTRL training | `opaque.dpftrl` (and submodules `noise`, `sampling`, `clipping`, `accounting`) |
| Privacy accounting algebra | `opaque.accounting` |
| DP-SGD-specific accountants | `opaque.dpsgd.accounting` |
| DP-FTRL-specific accountants | `opaque.dpftrl.accounting` |

## Cross-cutting infrastructure (not algorithm choice)

| Concern | Module |
|---------|--------|
| RNG keys | `opaque.random` |
| Functional models / `torch.func` helpers | `opaque.functional` |
| DDP / `sync` | `opaque.distributed` |
| Checkpoints / flat state | `opaque.serialization` |
| Step schedules | `opaque.scheduling` |
| Optimizers (AdamW-BC, etc.) | `opaque.optimizers` |
| Pipeline wrapper types | `opaque.types` |
| PyTree helpers | `opaque.pytree` |

## Clipping

- **Fixed clipping and AUTO-S** are implemented in `opaque-core` under **`opaque._clipping`** (private). Do not import this path from application code; CI may forbid it outside approved packages.
- **DP-SGD:** `opaque.dpsgd.clipping` — `clipped_grad`, `auto_clipped_grad`, `per_group`, `adaptive_clipped_grad`, plus `opaque.dpsgd.clipping.types` / `.fun` for annotations and AUTO-S fun-level APIs.
- **DP-FTRL:** `opaque.dpftrl.clipping` — MF-safe `clipped_grad`, `auto_clipped_grad`, `per_group` (same math as SGD; different **documented** entry point so users stay inside the FTRL tree).

## Sampling (post–API unification)

| Class | Package | Role |
|-------|---------|------|
| `PoissonSubsampler` | `opaque.dpsgd.sampling` | Standard / truncated Poisson subsampling for DP-SGD |
| `CyclicPoissonSampler` | `opaque.dpftrl.sampling` | Cyclic-band Poisson for BandMF-style amplification (`bands` ≥ 1) |

Both use the keyword **`sample_rate`** for the per-example inclusion probability.

## Future packages (e.g. Lipschitz DP)

Follow the same pattern: `opaque.<algorithm>.*` for all training-facing symbols; register `opaque.distributed.sync` handlers inside that package; do not depend on `opaque-dpsgd` or `opaque-dpftrl` wheels.
