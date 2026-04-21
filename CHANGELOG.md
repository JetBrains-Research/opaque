# Changelog

<!-- markdownlint-disable MD024 -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking — modularization (Option B namespace layout)

The monorepo has been split into first-class, standalone distributions,
each with its own real `__init__.py` under a dedicated namespace root.
There are **no backward-compatibility shims**. Every old import path is
removed; downstream code must migrate in one pass.

#### Added

- `opaque.core` (was top-level `opaque.*` primitives) — RNG, pytree
  helpers, clipping primitives, Poisson sampling, distributed, profiling,
  utils
- `opaque.dpsgd` — Gaussian / truncated-Gaussian / per-group noise,
  AdamW-BC, `TruncatedPoissonSampler`, adaptive + auto clipping
- `opaque.mf` — BLT / BSR / BiSR / band-MF / JME / λ-CGD mechanisms,
  AdamW-JME, b-min-sep / cyclic-Poisson / balls-in-bins / sequential
  samplers
- `opaque.auditing` — curated facade for empirical privacy auditing
- `opaque.performance` — fused Triton kernels (`opaque.performance.kernels`)
  and PyTorch-version patches (`opaque.performance.torch.checkpoint`)
- `opaque.huggingface` — HF Transformers patches (`opaque.huggingface.patches`)
  plus scaffolded `trainer/`, `callbacks/`, `integrations/`, `data/`,
  `models/` subpackages
- `opaque.accounting` — Python facade over the native PyO3 extension
  (mounted at `opaque.accounting._native`)
- `opaque.patch_all()` curated umbrella facade honoring
  `OPAQUE_SKIP_COMPAT_PATCHES` (`all`, `huggingface`, `performance`,
  comma-combo)
- `scripts/check_namespaces.py` and `scripts/check_negative_imports.py`
  as CI guardrails enforcing the new layout

#### Removed (no replacement with compatibility layer)

- `opaque_accounting` top-level module — use `opaque.accounting`
- `opaque.compat` namespace — split into `opaque.performance.kernels`,
  `opaque.performance.torch.checkpoint`, and `opaque.huggingface.patches`
- `opaque.sampling.truncated_poisson` — use `opaque.dpsgd.sampling.truncated_poisson`
- `opaque.sampling.b_min_sep` / `.cyclic_poisson` / `.balls_in_bins` /
  `.sequential` — use `opaque.mf.sampling.*`
- `opaque.clipping.adaptive` / `opaque.clipping.auto` — use
  `opaque.dpsgd.clipping.*`
- `opaque.noise.<dp-sgd-mechanism>` — use `opaque.dpsgd.noise.*`
- `opaque.noise.mf.*` — use `opaque.mf.noise.*`
- `opaque.optimizers.adamw_bc` / `.adamw_jme` — use
  `opaque.dpsgd.optimizers.adamw_bc` / `opaque.mf.optimizers.adamw_jme`
- Auto-import-time patching of HuggingFace models — patching is now
  opt-in via `opaque.patch_all()` (or `opaque.huggingface.patch_all()`)

#### Changed

- Umbrella `opaque` distribution pins sub-packages with `==` instead of
  `>=` to prevent skew
- Per-package `[project.optional-dependencies]` now cover
  `opaque-performance[kernels]`, `opaque-huggingface[peft,kernels]`,
  `opaque-accounting[cross-validation]`, and similar
- The PyO3 native module is installed at `opaque.accounting._native` via
  maturin's `module-name`; Rust crate/identifier name is unchanged

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
