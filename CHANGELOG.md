# Changelog

<!-- markdownlint-disable MD024 -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
