# Development Guide

## Project Structure

```
external/
├── opaque/              # This project
│   ├── opaque/          # Main package
│   │   ├── layers/      # Per-sample gradient computations
│   │   ├── grad_sample/ # Hook/functorch infrastructure
│   │   ├── optimizers/  # DP optimizer (clipping + noise)
│   │   ├── accountants/ # Privacy accounting (RDP, PRV)
│   │   └── privacy_engine.py  # Main API
│   ├── tests/           # Unit tests
│   ├── examples/        # Usage examples
│   └── docs/            # Documentation
│
└── opacus/              # Reference implementation
    └── (full opacus codebase)
```

## Keeping Opacus Reference

The Opacus codebase is kept as a sibling directory for reference:

```bash
# You're currently in: /Users/evgri243/Workspaces/external/opacus
cd /Users/evgri243/Workspaces/external

# Work on opaque
cd opaque

# Reference opacus code
cd ../opacus
```

### Benefits of this approach:

1. **Easy code reference**: Just open `../opacus/` in your editor
2. **Easy comparison**: `diff ../opacus/opacus/grad_sample/linear.py opaque/layers/linear.py`
3. **No git complexity**: No submodules or subtrees
4. **Clean separation**: Both are independent git repos
5. **IDE workspace**: Add both to your IDE workspace

## Setting up Development Environment

```bash
cd /Users/evgri243/Workspaces/external/opaque

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev,peft,examples]"
```

## Running Tests

```bash
pytest tests/
pytest tests/ -v  # Verbose
pytest tests/ --cov=opaque  # With coverage
```

## Code Style

We use:
- **Black** for formatting
- **Ruff** for linting
- **MyPy** for type checking

```bash
# Format code
black opaque/ tests/

# Lint
ruff check opaque/ tests/

# Type check
mypy opaque/
```

## Referencing Opacus Code

When adapting code from Opacus:

1. Open the reference file: `../opacus/opacus/grad_sample/controller.py`
2. Copy relevant sections
3. Simplify for Linear-only use case
4. Add comment with source:

```python
# Adapted from opacus/grad_sample/controller.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
```

## Key Simplifications from Opacus

| Opacus | Opaque | Reason |
|--------|--------|--------|
| 14 layer types | 1 (Linear) | LoRA only |
| Ghost clipping | Removed | Loss wrapping issues |
| Module wrapping | Hooks only | Simpler, more stable |
| DPDataLoader | None | Users handle sampling |
| Distributed | None (v1) | Added complexity |
| Validators | Simple check | Linear-only validation |

## Architecture Decisions

### Why hooks-only (initially)?

- ✅ Stable, battle-tested
- ✅ No loss wrapping
- ✅ Works with any optimizer
- ✅ ~400 lines for full implementation

### Why Linear-only?

- ✅ LoRA adapters are Linear layers
- ✅ Reduces codebase by 70%
- ✅ Simpler to audit for security
- ✅ Easier to optimize later

### Why no DataLoader wrapping?

- ✅ HuggingFace trainers handle sampling
- ✅ Users can implement Poisson sampling if needed
- ✅ Reduces API surface area
- ✅ Documented in examples

## Adding Features

### To add a new layer type:

1. Add grad sampler to `opaque/layers/`
2. Register in `opaque/layers/__init__.py`
3. Update validation in `privacy_engine.py`
4. Add tests in `tests/layers/`

### To add a new accounting mechanism:

1. Add accountant to `opaque/accountants/`
2. Follow interface from `accountants/base.py`
3. Add tests in `tests/accountants/`

## Performance Notes

For ~100k LoRA parameters:
- Hooks overhead: ~5-10%
- Worth it for stability and simplicity
- Can optimize with Unsloth-style kernels later

## Release Process

1. Update version in `pyproject.toml` and `__init__.py`
2. Run full test suite: `pytest tests/`
3. Build: `python -m build`
4. Upload to PyPI: `twine upload dist/*`

## Questions?

Open an issue or discussion on GitHub!
