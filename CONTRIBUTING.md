# Contributing to Opaque

Thank you for your interest in contributing to Opaque!

**For comprehensive development documentation, see [docs/development/](docs/development/).**

---

## Quick Start

### Setup

```bash
# Clone repository
git clone https://github.com/evgri243/opaque.git
cd opaque

# Clone JAX-Privacy reference (required)
cd .. && git clone https://github.com/google-deepmind/jax_privacy.git && cd opaque

# Install dependencies
uv sync

# Verify installation
uv run pytest
```

### Optional: JAX Validation

```bash
uv sync --group jax-validation
uv run --group jax-validation pytest -m jax_validation
```

---

## Development Philosophy

1. **Correctness First**: Validate against JAX-Privacy before optimizing
2. **Explicit > Implicit**: Fail fast rather than hide issues
3. **Documentation Required**: Code without docs is incomplete
4. **Security-Critical**: DP guarantees depend on correct implementation

---

## TDD Workflow

Opaque follows a Test-Driven Development workflow:

1. **Discover**: Study JAX-Privacy implementation in `../jax_privacy`
2. **JAX Test** (optional): Create reference test
3. **Failing Test**: Write Opaque test defining API
4. **Implement**: Make test pass
5. **Document**: Add docstrings and examples
6. **Validate**: Compare numerically with JAX-Privacy

**See**: [Detailed TDD Workflow](docs/development/tdd-workflow.md)

---

## Testing

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=opaque --cov-report=html

# JAX validation tests
uv run --group jax-validation pytest -m jax_validation

# Specific test
uv run pytest tests/core/test_clipping.py::test_clip_pytree -v
```

**Requirements**:
- ✅ Tests for all new functionality
- ✅ Maintain >80% coverage
- ✅ Pass all existing tests
- ✅ Include docstring examples

---

## Code Quality

```bash
# Format code
uv run ruff format src/ tests/

# Check linting
uv run ruff check src/ tests/

# Fix auto-fixable issues
uv run ruff check --fix src/ tests/
```

**Standards**:
- Line length: 100 characters
- Type hints for public APIs
- Google-style docstrings

---

## Pull Request Process

### Before Submitting

1. **Run tests**: `uv run pytest`
2. **Format code**: `uv run ruff format --check src/ tests/`
3. **Check linting**: `uv run ruff check src/ tests/`
4. **Update docs**: Ensure docstrings are complete

### PR Checklist

- [ ] Tests pass locally
- [ ] Code is formatted with Ruff
- [ ] Docstrings follow Google style
- [ ] Type hints added for public APIs
- [ ] Linked to relevant issue(s)

---

## Commit Messages

**Format**:
```
<type>: <subject>

<body>
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

**Example**:
```
feat: implement basic clipped_grad

- Add clipped_grad function to opaque.core.clipping
- Support single parameter and batch data
- Include tests comparing against JAX-Privacy

Closes #5
```

---

## Documentation

### Build Docs Locally

```bash
# Install docs dependencies
uv sync --group docs

# Serve with live reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

### Writing Docs

- **User guides**: `docs/user-guide/`
- **API reference**: Auto-generated from docstrings
- **Development**: `docs/development/`

---

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/evgri243/opaque/discussions)
- **Bugs**: Open an [Issue](https://github.com/evgri243/opaque/issues)

---

## Detailed Documentation

**For comprehensive guides, see:**

- [TDD Workflow](docs/development/tdd-workflow.md) - Step-by-step development process
- [Architecture](docs/development/architecture.md) - System design and abstractions
- [Design Decisions](docs/development/design-decisions.md) - Technical choices and rationale
- [Stage 1 Plan](docs/development/stage1-plan.md) - Current implementation plan
- [Roadmap](docs/development/roadmap.md) - Full project timeline

---

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

## Acknowledgments

Thank you for contributing to differential privacy research and making LLM fine-tuning more privacy-preserving!

**Key Resources**:
- [JAX-Privacy](https://github.com/google-deepmind/jax_privacy) - Reference implementation
- [Full Opaque Documentation](docs/)
