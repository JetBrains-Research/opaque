# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and repository setup
- Core package structure (`src/opaque/core/`)
  - `pytree_utils.py` - PyTree operations (stubs)
  - `clipping.py` - Gradient clipping (stubs)
- Comprehensive documentation using Material for MkDocs
  - Getting Started guides (installation, quickstart)
  - User guides (DP basics, clipping, LoRA)
  - API reference structure
  - Development guides (architecture, TDD workflow, design decisions, roadmap, stage plans)
- Testing infrastructure
  - pytest configuration with custom markers
  - `jax-validation` marker for optional JAX cross-validation tests
  - Test utilities and helpers
  - Test directory structure (`tests/core/`, `tests/jax_validation/`)
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

### Documentation
- Comprehensive Stage 1 implementation plan
- Complete TDD workflow documentation with examples
- Architecture overview with PyTorch/JAX equivalents
- 10 documented design decisions with rationale
- Full 5-stage project roadmap
- JAX-Privacy API comparison and porting strategy

### Project Decisions
- Use Material for MkDocs for documentation (over Sphinx)
- Use `torch.utils._pytree` for PyTree operations (with wrapper layer)
- Follow strict TDD workflow: tests before implementation
- Assume JAX-Privacy reference at `../jax_privacy/` for validation
- Focus on LoRA fine-tuning as primary use case
- Port JAX-Privacy's functional API (`experimental/clipping.py`)

## [0.1.0] - TBD

*First implementation release will be tagged after Stage 1 completion*

### Planned for 0.1.0
- `opaque.core.pytree_utils` implementation
- `opaque.core.clipping` implementation
- Comprehensive test suite (unit tests + JAX validation)
- Example: Linear regression with DP-SGD
- API documentation from docstrings

---

**Note**: This project is in early development. Version 0.1.0 will be the first release with working implementations.
