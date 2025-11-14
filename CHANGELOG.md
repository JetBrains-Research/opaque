# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- User guide documentation for all major concepts
- API reference structure for all modules
- Cross-linking between documentation sections

## [0.2.0] - 2025-11-14

**Stage 2 Complete**: Privacy accounting, noise injection, optimizer integration, and sampling mechanisms.

### Added

- **Privacy Accounting** (`opaque.accounting`)
  - Functional API with immutable state (`create()`, `compose_*()`, `get_*()`)
  - Support for (ε, δ)-DP, f-DP advantage, and (α, β) error rates
  - Composition methods: `compose_gaussian()`, `compose_poisson_gaussian()`, `compose_sampled_gaussian()`,
    `compose_truncated_poisson_gaussian()`
  - Privacy queries: `get_epsilon()`, `get_beta()`, `get_advantage()`
  - Calibration using riskcal.calibration.core primitives
  - Custom calibration: `find_noise_multiplier_for_epsilon_delta()`
  - Re-exported calibration: `find_noise_multiplier_for_advantage()`, `find_noise_multiplier_for_err_rates()`
- **Noise Injection** (`opaque.noise`)
  - `add_gaussian_noise()` - Stateless Gaussian noise generation
  - Support for noise multiplier calibration
- **Sampling Mechanisms** (`opaque.sampling`)
  - `PoissonSampler` - Standard Poisson sampling for privacy amplification
  - `TruncatedPoissonSampler` - Bounded batch sizes for variable-length datasets
  - Solves variable batch size problem in DP-SGD
- **DP Optimizers** (`opaque.optimizers`)
  - `adaptive_clipping()` - Adaptive gradient clipping wrapper for any TorchOpt optimizer
  - `ClipBuffer` - Maintains clipping norm statistics with exponential moving average
  - `LRScheduler` - Learning rate scaling based on clipping statistics
  - Works with any TorchOpt optimizer (SGD, Adam, AdamW, etc.)
  - Integration with functional clipping and accounting APIs

### Changed

- **Modular Architecture**: Split clipping module into proper package structure
  - `clipping/types.py` - Type definitions (`BoundedSensitivityCallable`, `AuxiliaryOutput`)
  - `clipping/pytree.py` - Low-level PyTree clipping
  - `clipping/clipped_fun.py` - Primary clipping API
  - `clipping/clipped_grad.py` - Gradient clipping wrapper
  - `clipping/_helpers.py` - Internal utilities
- **API Organization**: Reorganized package exports for clarity
  - Top-level exports: `clip_pytree`, `clipped_fun`, `clipped_grad`
  - Submodule exports: `opaque.accounting`, `opaque.noise`, `opaque.sampling`, `opaque.optimizers`

### Tests

- **111 tests passing** (55 accounting + 56 optimizer tests)
- **Parallel execution** with pytest-xdist (~3.17x speedup, 12 workers)
- **Accounting tests** (55 total):
  - 13 composition tests
  - 16 privacy query tests
  - 21 calibration tests
  - 5 JAX validation tests (cross-framework numerical equivalence)
- **Optimizer tests** (56 total):
  - 37 adaptive clipping tests
  - 19 DP optimizer integration tests
- **76% coverage** for accounting module

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
