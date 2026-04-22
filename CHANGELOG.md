# Changelog

<!-- markdownlint-disable MD024 -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-04-22

### Highlights

### What's changed

### [0.2.2] - 2026-04-22

#### CI/CD

- Draft-release-driven pipeline + PR preview wheels (#144) ([f6ac256](https://github.com/JetBrains-Research/opaque/commit/f6ac256bf7e2cf2c4a5696f997e29cf52807d416))

#### Fixed

- **deps:** Pin torch<2.11 for CUDA 12 compat and restore peft in examples (#143) ([fa7a7fd](https://github.com/JetBrains-Research/opaque/commit/fa7a7fd3c3e76c262202a58891f10288822b41bb))
- **examples:** Default eval_batch_size to microbatch_size (#145) ([5463a02](https://github.com/JetBrains-Research/opaque/commit/5463a0267a1a4839514b52d12d88bd50633f6539))

### Contributors


## [0.2.1] - 2026-04-22

**Patch release — release-pipeline hardening only. No user-facing
code changes.** Cutting this release ships `main` through the fixed
pipeline end-to-end; functional code is identical to 0.2.0.

### CI/CD

- **Dev-wheel publish from `main` now works.** The preflight
  ([`set_build_versions.sh`](.github/scripts/set_build_versions.sh))
  normalizes `git describe` output to PEP 440 instead of passing the
  raw `0.2.0-5-g<sha>` form straight into `pyproject.toml` (which
  `uv build` refused to parse). Both the main-branch dev publish and
  the tag-push release publish are now scoped to the `release` GitHub
  Environment so they auth as the service account that's actually
  bound as `roles/artifactregistry.writer` on the target GCP
  Artifact Registry repo. ([#137](https://github.com/JetBrains-Research/opaque/pull/137))
- **Conventional Commits enforced on PR titles.** The repo
  squash-merges, so the PR title becomes the commit message git-cliff
  reads. The PR gate now rejects titles cliff wouldn't categorize.
  Accepted types mirror `cliff.toml`'s `commit_parsers` exactly.
  ([#138](https://github.com/JetBrains-Research/opaque/pull/138))
- **Release tag guard.** The release workflow fails fast if the
  pushed tag isn't reachable from `origin/main`. Release-time tests
  aren't re-run — the main-branch gate is the primary enforcement —
  so off-main tags would otherwise ship untested code.
  ([#139](https://github.com/JetBrains-Research/opaque/pull/139))
- **Preflight tag-shape lockdown.** Only `X.Y.Z` (release) and
  `X.Y.Z.devN` (dev-cycle anchor) are accepted. Hyphen-form anchors,
  rc/alpha/beta tags, and legacy `-test` markers fail-fast with a
  clear error instead of silently producing invalid PEP 440.
  ([#140](https://github.com/JetBrains-Research/opaque/pull/140))
- **`cliff.toml` anchor-tag ignore pattern** now accepts both
  `v0.X.Y.dev0` (dot, PEP 440) and `v0.X.Y-dev0` (hyphen, semver)
  forms so the anchor doesn't leak into future release notes
  regardless of which form the cycle uses.
- **Branch protection.** The `main` ruleset now requires the 8 PR-gate
  status checks (CPU/MPS/Rust tests, cross-package import smoke, docs
  build, Python/Rust format checks, Conventional Commits PR title)
  before merge. Stale repo-level GCP auth vars removed in favor of
  the `release` environment.

## [0.2.0] - 2026-04-21

**Major restructure.** The monorepo is split into seven first-class
PEP 420 namespace distributions, the user-facing API is promoted to the
`opaque.*` root, and a new tag-triggered release pipeline drives the
release process. Alongside the reshape, 0.2.0 ships new optimizers,
clipping strategies, noise types, samplers, and accounting mechanisms.
**No backwards-compatibility shims** for the pre-0.2 import layout —
downstream code must migrate in one pass.

### Added

#### Features

New functionality introduced since v0.1.0, independent of the reshape:

- **Optimizers** — `AdamW-BC` (bias-corrected AdamW for DP-SGD) at
  `opaque.dpsgd.optimizers.adamw_bc`; `AdamW-JME` (AdamW with JME
  dual-stream noise for DP-FTRL, following Kalinin et al.,
  arXiv:2502.06597) at `opaque.dpftrl.optimizers.adamw_jme`; internal
  optimizer-chaining helper.
- **Clipping** — `auto` clipping strategy as a heuristic alternative to
  the quantile-based `adaptive` clipping
  (`opaque.dpsgd.clipping.auto`); per-group clipping helper
  (`opaque.core.clipping.per_group`, re-exported from `opaque.core`).
- **Noise** — per-group Gaussian noise
  (`opaque.dpsgd.noise.per_group_noise`); JME noise and a unified
  MF-noise dispatcher for DP-FTRL
  (`opaque.dpftrl.noise.{jme,dispatcher}`).
- **Samplers** — `BMinSepSampler` (b-min-separation, with transcript
  cache), `BallsInBinsSampler`, and the sequential sampler, all at
  `opaque.dpftrl.sampling.*`.
- **Accounting mechanisms** — `BiSR`, `BSR`, `λ-CGD`, `nonprivate`
  baseline, and `JME` transformation.
- **Amplification** — b-min-sep and balls-in-bins analyses in
  `opaque.accounting.amplification`.
- **Distributed** — `opaque.distributed.collectives` (collective-ops
  helpers) and `opaque.distributed.shard` (shard utilities broken out
  from `sampling.distributed`).

#### Package layout

- Seven first-class distributions under the `opaque` namespace
  (PEP 420, no shared `__init__.py`):
  - `opaque-core` — RNG, functional helpers, pytree, clipping
    primitives, Poisson sampling core, distributed, profiling hooks.
  - `opaque-dpsgd` — Gaussian / truncated-Gaussian / per-group noise,
    `AdamW-BC`, `TruncatedPoissonSampler`, `adaptive` + `auto` clipping.
  - `opaque-dpftrl` — BLT / BSR / BiSR / band-MF / JME / λ-CGD
    mechanisms, `AdamW-JME`, b-min-sep / cyclic-Poisson /
    balls-in-bins / sequential samplers.
  - `opaque-auditing` — curated facade for empirical privacy auditing.
  - `opaque-performance` — fused Triton kernels
    (`opaque.performance.kernels`) and PyTorch-version patches
    (`opaque.performance.torch.checkpoint`).
  - `opaque-huggingface` — HF Transformers compatibility patches plus
    scaffolded `trainer/`, `callbacks/`, `integrations/`, `data/`,
    `models/` subpackages.
  - `opaque-accounting` — Python facade over the PyO3 native extension
    (mounted at `opaque.accounting._native`).
- Umbrella `opaque` distribution reduced to metadata that `==`-pins
  sub-packages.
- Per-package `README.md` files and per-package
  `[project.optional-dependencies]` extras
  (`opaque-performance[kernels]`, `opaque-huggingface[peft,kernels]`,
  `opaque-accounting[cross-validation]`, …).

### Changed

- **Patching model** is per-sub-package and on-import: importing
  `opaque.performance` or `opaque.huggingface` applies their patches
  automatically, gated by `OPAQUE_SKIP_PYTORCH_PATCHES` and
  `OPAQUE_SKIP_TRANSFORMERS_PATCHES` respectively. Kernel patches that
  wire Triton kernels into HF model classes now live in
  `opaque.performance.huggingface`, gated by
  `OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES`. `opaque.huggingface`
  keeps only the compatibility patches (vmap-safe attention, KV cache,
  Poisson collator). Umbrella `opaque.patch_all()` is gone.
- **Test markers** collapsed from 6 to 3: `cuda`, `mps`, `slow`. The
  legacy `gpu` marker (and its `mps_compatible` modifier) is replaced
  by the orthogonal `cuda` / `mps` pair; `hf_auth_required` is
  replaced by a runtime `@requires_hf_auth` skipif helper keyed on
  `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` / `HUGGINGFACE_TOKEN`.
  CI lane expressions updated accordingly; `slow` runs on push to
  `main` only.
- **PyO3 native module** is installed at `opaque.accounting._native`
  via maturin's `module-name`; Rust crate/identifier name is unchanged.
- **PEP 420 invariant** enforced inline in
  `.github/workflows/ci.yml` (no stray `src/opaque/__init__.py` in
  sub-packages). Replaces `scripts/check_namespaces.py`
  (legacy-token and refactor-diary checks dropped post-migration).

### Removed

- **Accounting mechanisms**: `band_mf_amplified` (folded into
  `band_mf`), `dense_mf` (use `band_mf` / `blt`), `rectified_gaussian`,
  and `truncated_gaussian` as an accounting mechanism. The
  truncated-Gaussian *noise* mechanism still ships as
  `opaque.dpsgd.noise.truncated_gaussian`. `blt_mf` renamed to `blt`.
- `opaque_accounting` top-level module — use `opaque.accounting`.
- `opaque.compat.*` — split into `opaque.performance.kernels`,
  `opaque.performance.torch.checkpoint`, and
  `opaque.huggingface.patches`.
- `opaque.patch_all()` and `OPAQUE_SKIP_COMPAT_PATCHES` — use the
  per-sub-package auto-patching and the specific `OPAQUE_SKIP_*`
  variants.
- Auto-import-time patching of HuggingFace models at the umbrella
  `opaque` level — patch application is now tied to importing
  `opaque.huggingface` / `opaque.performance` explicitly.
- Legacy workflows: `publish.yml`, `release.lock.yml`, `release.md`,
  `docs-check.yml`.

### Breaking — migration map

No backward-compatibility shims. Every old import path must be
rewritten.

**Relocations inside the namespace reshape** (what used to sit under
`opaque.core.*` or `opaque.mf.*`):

| Old path | New path |
|----------|----------|
| `opaque.core.utils.functional.*` | `opaque.functional.*` |
| `opaque.core.utils.pytree.*` | `opaque.core.pytree.*` |
| `opaque.core.utils.per_group.*` | `opaque.core.clipping.per_group.*` (re-exported from `opaque.core`) |
| `opaque.core.distributed.*` | `opaque.distributed.*` (split into `collectives`, `gradients`, `state`, `shard`) |
| `opaque.core.sampling.PoissonSampler` | `opaque.dpsgd.sampling.PoissonSampler` |
| `opaque.core.sampling.PartitionType` | `opaque.dpftrl.sampling._partitions.PartitionType` (private) |
| `opaque.core.sampling.poisson_collate` | `opaque.core.sampling.empty_collate` (renamed) |
| `opaque.core.sampling.distributed.local_shard` | `opaque.distributed.local_shard` |
| `opaque.core.profiling.*` | `opaque.performance.profiling.*` |
| `opaque.huggingface.patches._kernel_patches` | `opaque.performance.huggingface.kernel_patches` |
| `opaque.mf.*` (package `opaque-mf`) | `opaque.dpftrl.*` (package `opaque-dpftrl`) |

**Sub-package splits from the old monolithic `opaque.*`**:

| Old path | New path |
|----------|----------|
| `opaque_accounting.*` | `opaque.accounting.*` |
| `opaque.sampling.truncated_poisson` | `opaque.dpsgd.sampling.truncated_poisson` |
| `opaque.sampling.{b_min_sep,cyclic_poisson,balls_in_bins,sequential}` | `opaque.dpftrl.sampling.*` |
| `opaque.clipping.{adaptive,auto}` | `opaque.dpsgd.clipping.*` |
| `opaque.noise.<dp-sgd-mechanism>` | `opaque.dpsgd.noise.*` |
| `opaque.noise.mf.*` | `opaque.dpftrl.noise.*` |
| `opaque.optimizers.adamw_bc` | `opaque.dpsgd.optimizers.adamw_bc` |
| `opaque.optimizers.adamw_jme` | `opaque.dpftrl.optimizers.adamw_jme` |
| `opaque.compat.*` | `opaque.performance.*` / `opaque.huggingface.patches` |
| `opaque.patch_all()` | **removed** — per-sub-package auto-patching |
| `OPAQUE_SKIP_COMPAT_PATCHES` | **removed** — use `OPAQUE_SKIP_PYTORCH_PATCHES`, `OPAQUE_SKIP_TRANSFORMERS_PATCHES`, `OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES` |

### Release infrastructure

- New tag-triggered release workflow (`.github/workflows/release.yml`):
  pushing a `v*.*.*` tag builds wheels for all seven Python
  distributions in parallel plus a matrix of native `opaque-accounting`
  wheels (`linux/amd64`, `linux/arm64`, `macos/arm64`), publishes to
  GCP Artifact Registry via workload identity federation, and cuts a
  GitHub Release with notes generated from the Conventional Commit log
  by git-cliff.
- Split CI: `pr.yml` runs a lighter PR gate; `ci.yml` runs the full
  suite (including `slow`) on pushes to `main`.
- Versioning: Python sub-packages use setuptools-scm for dynamic
  versioning from the git tag. `.github/scripts/set_build_versions.sh`
  preflights the build by rewriting `opaque-accounting/pyproject.toml`,
  the workspace `Cargo.toml`, and the umbrella's `opaque-*==X` pins so
  maturin and Cargo agree on the tag version. PEP 440 → SemVer mapping
  applied for the Rust crate.
- `cliff.toml` at the repo root drives both CHANGELOG and release-note
  generation.

### Documentation

- Full rewrite of every page under `docs/` for the new `opaque.*`
  namespace — landing pages, user guides, API reference, mechanisms
  reference, tutorials, release instructions.
- Root `README.md` rewritten for the post-refactor structure.
- Per-package `README.md` files added for `opaque-core`, `opaque-dpsgd`,
  `opaque-dpftrl`, `opaque-auditing`, `opaque-performance`,
  `opaque-huggingface`.

## [0.1.0] - 2026-03-11

**First public release of Opaque's functional DP-SGD stack for PyTorch.**

### Added

- Functional DP-SGD primitives built on `torch.func`, including per-example gradient clipping, Gaussian and bounded Gaussian noise, and Poisson-family samplers
- Rust-backed privacy accounting via `opaque-accounting`, including calibration, composition, matrix-factorization mechanisms, and multiple privacy metrics
- Distributed DP training helpers, empirical privacy auditing, and memory profiling utilities
- HuggingFace compatibility patches, fused kernel optimizations, and gradient-checkpointed LLM fine-tuning support

### Changed

- Refactored privacy accounting around query-time discretization, universal PLD caching, and function-first `opaque.accounting` exports
- Simplified the sampling and RNG APIs with immutable keys, automatic distributed sharding, and `local_shard()` / variadic `fold_in()` helpers
- Redesigned the auditing API around `coin_flip()`, `loss_scores()`, and `one_run()` for HuggingFace-oriented workflows
- Reworked memory profiling, Transformers patching, and kernel integration for more stable CPU, MPS, and CUDA behavior

### Fixed

- Stabilized fused kernels, gradient checkpointing, training scripts, and compatibility tests across the supported device matrix
- Fixed Artifact Registry publishing and release workflow validation for the automated release pipeline

### Documentation

- Refreshed the documentation set across landing pages, user guides, mechanisms reference, tutorials, and release instructions

## [0.0.0] - 2025-11-08

**Initial Setup**: Project structure and infrastructure.

### Added

- Initial project structure and repository setup
- Package structure (`src/opaque/`)
- Comprehensive documentation using Material for MkDocs
  - Getting Started guides (installation, quickstart stub)
  - User guides (DP basics, clipping stub, LoRA stub)
  - API reference structure
  - Development guides (architecture, TDD workflow, design decisions, roadmap, stage plans)
- Testing infrastructure
  - pytest configuration with custom markers
  - `jax-validation` marker for optional JAX cross-validation tests
  - Test directory structure (`tests/clipping/`, `tests/jax_validation/`)
- Development tooling
  - uv package manager configuration
  - ruff for code formatting and linting
  - Pre-configured dependency groups (dev, docs, jax-validation)
- Documentation files
  - README.md - Project overview and quick start
  - CONTRIBUTING.md - Contribution guidelines
  - CLAUDE.md - Agent-only briefing document
  - CHANGELOG.md - This file
- Configuration files
  - pyproject.toml - Project metadata and dependencies
  - mkdocs.yml - Documentation site configuration
  - .editorconfig - Code style settings
  - .gitignore - Git ignore patterns

### Project Decisions

- Use Material for MkDocs for documentation (over Sphinx)
- Use `torch.utils._pytree` for PyTree operations (with wrapper layer)
- Follow strict TDD workflow: tests before implementation
- Assume JAX-Privacy reference at `../jax_privacy/` for validation
- Focus on LoRA fine-tuning as primary use case
- Port JAX-Privacy's functional API (`experimental/clipping.py`)

---
