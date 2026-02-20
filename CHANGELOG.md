# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive documentation refresh for all user-facing docs
- Cross-linking between documentation sections

### Changed

- Accounting: removed `accumulate()` in favor of `parallel_poisson()`
- Accounting: `adaclip()` now returns an `AdaClip` process (class in transformations submodule)
- Accounting: all DP processes implement `state_dict()` for serialization
- Accounting: top-level `opaque.accounting` exports functions only

## [0.2.0] - 2025-11-14

**Complete DP-SGD library with functional API, Rust-based accounting, and HuggingFace integration.**

### Added

- **Privacy Accounting** (`opaque.accounting`)
  - Rust-based PLD engine (`opaque_accounting` via PyO3)
  - Mechanism constructors: `gaussian()`, `poisson()`, `truncated_poisson()`, `accumulate()`, `adaclip()`, `eps_delta()`, `identity()`
  - Composition via `*` (repeat) and `|` (compose) operators on `DpProcess` objects
  - Privacy metrics: `.epsilon_at()`, `.delta_at()`, `.advantage()`, `.beta_at()`, `.risk_at()`
  - `Accountant` class for training-loop integration with optional budget checking
  - Calibration: `calibrate()` with target factories `epsilon()`, `delta()`, `advantage()`, `beta()`, `risk()`
  - Configurable PLD precision via `DiscretizationConfig`
- **Noise Injection** (`opaque.noise`)
  - `gaussian_noise()` — Stateful Gaussian noise with reproducible RNG
  - `bounded_gaussian_noise()` — Truncated Gaussian within bounds
  - Matrix factorization noise: `band_mf_noise()`, `blt_mf_noise()`, `dense_mf_noise()`, `custom_mf_noise()`, `identity_mf_noise()`
- **Sampling Mechanisms** (`opaque.sampling`)
  - `PoissonSampler` — Standard Poisson sampling for privacy amplification
  - `TruncatedPoissonSampler` — Bounded batch sizes
  - `CyclicPoissonSampling` — Cyclic Poisson for BandMF amplification
- **Adaptive Clipping** (`opaque.clipping.adaptive`)
  - `adaptive_clipped_grad()` — Adaptive gradient clipping with explicit state-passing
  - `AdaptiveClipState` — Immutable state with sensitivity computation
- **Privacy Auditing** (`opaque.auditing`)
  - `setup()` / `evaluate()` convenience API
  - `AuditResult` with epsilon estimation, AUROC, bootstrap confidence intervals
  - `CoinFlipExperiment` for membership inference
- **Memory Profiling** (`opaque.profiling`)
  - `MemoryProfiler`, `profile_memory()`, `find_max_microbatch_size()`
- **HuggingFace Compatibility** (`opaque.compat`)
  - Auto-patching for vmap-compatible forward passes
  - Supported: LLaMA, Mistral, Qwen2, Phi, OLMo, Gemma2

### Changed

- **Modular Architecture**: Split clipping module into proper package structure
  - `clipping/types.py` — Type definitions (`ClipState`, `FixedClipState`, `AdaptiveClipState`, auxiliary outputs)
  - `clipping/pytree.py` — Low-level PyTree clipping
  - `clipping/clipped_fun.py` — Primary clipping API
  - `clipping/clipped_grad.py` — Gradient clipping wrapper
  - `clipping/adaptive.py` — Adaptive clipping with explicit state
- **API Organization**: Reorganized package exports for clarity
  - Top-level exports: `clip_pytree`, `clipped_fun`, `clipped_grad`, `adaptive_clipped_grad`
  - Submodule exports: `opaque.accounting`, `opaque.noise`, `opaque.sampling`, `opaque.auditing`

### Tests

- **458 tests passing** (108 auditing + 55 accounting + 56 optimizer + 239 other)
- **Parallel execution** with pytest-xdist (~3.17x speedup)
- Numerical equivalence with JAX-Privacy confirmed (atol=1e-5)

### Documentation

- **Tutorial 02**: Differential Privacy Noise and Accounting (functional API)
- **Tutorial 03**: Complete DP-SGD Training
- **Tutorial 04**: Functional DP Training with TorchOpt
- **Tutorial 05**: Sampling and Microbatching
- **Tutorial 06**: LoRA HuggingFace DP Training
- Updated all tutorials to use functional accounting API

## [0.1.0] - 2025-11-11

**Stage 1 Complete**: Full single-device gradient clipping API with PyTree support.

### Added

- **Gradient Clipping** (`opaque.clipping`)
  - `clip_pytree()` - Low-level PyTree clipping with all edge cases (zero, inf, NaN-safe)
  - `clipped_fun()` - Primary API for clipping function outputs
  - `clipped_grad()` - High-level gradient clipping wrapper
  - Support for nested PyTrees (dictionaries of tensors)
  - Microbatching support for memory efficiency
  - `BoundedSensitivityCallable` wrapper with `sensitivity()` method
  - `AuxiliaryOutput` for auxiliary outputs
- **PyTree Utilities** (`opaque.utils`)
  - `tree_map()` - Apply function to PyTree leaves
  - `global_norm()` - L2 norm across PyTree
  - `tree_leaves()` - Extract leaves from PyTree
  - `make_functional()` - Convert module to functional form
- **Full JAX-Privacy API parity** for single-device features
  - All parameters from JAX-Privacy main branch implemented
  - `return_zero` support for privacy amplification via padding
  - `has_aux` field in `BoundedSensitivityCallable`

### Changed

- **Migrated to JAX-Privacy main branch API** (from v1.0.0)
- **Simplified `clipped_grad()`** to ~30 lines using `torch.func.grad(has_aux=True)`
- **Output signature**: `return_norms=True` now returns `(value, (aux, norms))`

### Tests

- **70 tests passing** with ~90% coverage
- **25 unit tests** covering all edge cases
- **45 JAX validation tests** with numerical equivalence (atol=1e-5)
- All tests validate against JAX-Privacy reference implementation

### Documentation

- **Tutorial 01**: Gradient Clipping from Basics
- Stage 1 implementation plan
- Complete TDD workflow documentation
- 10 documented design decisions with rationale

### Tech Debt

- `prng_argnum` parameter not implemented (requires sophisticated PRNG splitting)
- `microbatch_size` parameter deferred (requires inmemory_microbatched_fn_general wrapper)
- `spmd_axis_name` parameter deferred (JAX SPMD only, not needed for single-device)

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

**Note**: This project is in early development. Version 0.1.0 will be the first release with working implementations.
