# Base-vs-branch parity harness

Reproducible evidence scripts for `docs/development/multiplatform-migration-analysis.md`.

Every `*_run.py` script auto-detects whether it is running on the multi-platform
branch (presence of `opaque.api.torch`) or on the pre-migration baseline, so the
same file runs unmodified in both environments.

## Setup

```bash
# branch environment (this checkout)
uv sync --group dev --all-packages --extra all

# baseline environment (merge base 2aabb0d in a worktree)
git worktree add ../opaque-base 2aabb0d
cd ../opaque-base && uv sync --group dev --all-packages --extra all
```

## Scripts

| Script | Purpose |
|---|---|
| `parity_run.py` / `parity_compare.py` | Seeded bit-parity: key derivation, generator streams, fixed + AUTO-S clipping, Gaussian noise streams, 10 optimizer-rule trajectories, e2e DP-SGD, accounting calibration, MF noise streams |
| `dist_run.py` / `dist_compare.py` | Statistical equivalence: 50k-draw KS tests for the Gaussian mechanism, MF marginal stds + cross-step correlation matrices, adafactor drift probe |
| `bench_run.py` | Wall-clock DP-step microbenchmark (dispatch overhead) |
| `ckpt_save.py` / `ckpt_load.py` | Cross-version checkpoint compatibility (base-written → branch-loaded) |

## Usage

```bash
# run each *_run.py once per environment, then compare
cd ../opaque-base && uv run python <this-dir>/parity_run.py --out /tmp/base.pt
cd ../opaque      && uv run python <this-dir>/parity_run.py --out /tmp/branch.pt
uv run python <this-dir>/parity_compare.py /tmp/base.pt /tmp/branch.pt
```
