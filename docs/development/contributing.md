# Contributing

See [CONTRIBUTING.md](https://github.com/evgri243/opaque/blob/main/CONTRIBUTING.md) in the repository root for quick start and contribution guidelines.

---

## Development Guides

This section provides detailed guides for contributors:

- **[TDD Workflow](tdd-workflow.md)** - Step-by-step test-driven development process
- **[JAX-Privacy Comparison](jax-privacy-comparison.md)** - Which API we're porting and why
- **[Design Decisions](design-decisions.md)** - Technical choices and rationale
- **[Roadmap](roadmap.md)** - Full project timeline and stages
- **[Stage 1 Plan](stage1-plan.md)** - Current implementation details

---

## Quick Reference

### Setup

```bash
git clone https://github.com/evgri243/opaque.git
cd opaque
cd .. && git clone https://github.com/google-deepmind/jax_privacy.git && cd opaque
uv sync
```

### Testing

```bash
uv run pytest                                        # All tests
uv run pytest --cov=opaque --cov-report=html        # With coverage
uv run --group jax-validation pytest -m jax_validation  # JAX validation
```

### Code Quality

```bash
uv run ruff format src/ tests/                       # Format
uv run ruff check src/ tests/                        # Lint
```

### Documentation

```bash
uv run --group docs mkdocs serve                     # Serve locally
uv run --group docs mkdocs build                     # Build static site
```

---

## Getting Help

- **Questions**: [Discussions](https://github.com/evgri243/opaque/discussions)
- **Bugs**: [Issues](https://github.com/evgri243/opaque/issues)
