# Opaque: Agent Briefing

**Project**: PyTorch port of JAX-Privacy for DP-SGD LoRA fine-tuning
**Status**: Stage 0 Complete (setup) → Stage 1 Ready (implementation)
**JAX-Privacy Reference**: `../jax_privacy/` (always available)

---

## Current State

**Repository Phase**: Infrastructure complete, no implementation yet

**Key Constraint**: Follow TDD workflow strictly - write tests before implementation

**Stage 1 Goal**: Implement `opaque.core.pytree_utils` and `opaque.core.clipping`

---

## Quick Orientation

### What This Is
PyTorch implementation of differential privacy for LLM fine-tuning:
- **DP-SGD**: Per-example gradient clipping + Gaussian noise
- **LoRA Focus**: Parameter-efficient fine-tuning only
- **Functional API**: Porting JAX-Privacy's `experimental/clipping.py` (not OOP API)

### Essential Context Files
1. **Implementation Plan**: [docs/development/stage1-plan.md](docs/development/stage1-plan.md)
2. **TDD Workflow**: [docs/development/tdd-workflow.md](docs/development/tdd-workflow.md)
3. **Architecture**: [docs/development/architecture.md](docs/development/architecture.md)
4. **Design Decisions**: [docs/development/design-decisions.md](docs/development/design-decisions.md)
5. **Full Roadmap**: [docs/development/roadmap.md](docs/development/roadmap.md)

### Current Codebase
```
src/opaque/core/
├── pytree_utils.py    # STUB - PyTree operations (tree_map, global_norm)
└── clipping.py        # STUB - Gradient clipping (clip_pytree, clipped_grad)

tests/
├── core/              # Empty - ready for TDD
└── jax_validation/    # Empty - optional JAX cross-validation tests
```

---

## TDD Workflow (Required Process)

1. **Discover**: Read JAX-Privacy implementation in `../jax_privacy/`
2. **JAX Test** (optional): Create reference test with `@pytest.mark.jax_validation`
3. **Failing Test**: Write Opaque test defining API
4. **Implement**: Make test pass
5. **Document**: Add Google-style docstrings with examples
6. **Validate**: Numerical comparison with JAX-Privacy

**Commands**:
```bash
uv run pytest                                      # Run unit tests
uv run --group jax-validation pytest -m jax_validation  # JAX validation
uv run ruff format src/ tests/                    # Format
uv run ruff check src/ tests/                     # Lint
```

---

## Key Technical Mappings

**JAX → PyTorch**:
- `jax.vmap` → `torch.func.vmap`
- `jax.grad` → `torch.func.grad`
- `jax.tree_util.tree_map` → `torch.utils._pytree.tree_map`
- PyTree = nested dict of tensors

**DP-SGD Algorithm**:
1. Per-example gradients (via `vmap`)
2. Clip each gradient to max L2 norm
3. Sum clipped gradients
4. Add Gaussian noise (Stage 2)

---

## Design Decisions Summary

Full details: [docs/development/design-decisions.md](docs/development/design-decisions.md)

1. **PyTree**: Use `torch.utils._pytree` (private API, wrap it)
2. **Microbatching**: Explicit `microbatch_size` parameter
3. **Numerical Tolerance**: `atol=1e-5` for JAX validation
4. **Error Handling**: Fail-fast by default
5. **Documentation**: Material for MkDocs

---

## Stage 1 Next Steps

See [docs/development/stage1-plan.md](docs/development/stage1-plan.md) for full plan.

**Week 1**: PyTree utilities
- Implement `global_norm()`, `tree_leaves()`, `tree_map()`
- Reference: `../jax_privacy/src/training/grad_clipping_utils.py`

**Week 2**: Basic clipping
- Implement `clip_pytree()`, `clipped_grad()` (single param)
- Reference: `../jax_privacy/src/experimental/clipping.py`

**Week 3**: Full clipping + validation
- Support PyTree parameters, microbatching
- Create example: linear regression

---

## Fast Context Recovery

**Recent Changes** (2025-11-10):
- ✅ Deleted IMPLEMENTATION_PLAN.md, OPEN_QUESTIONS.md (content → docs/)
- ✅ Condensed CONTRIBUTING.md (367→203 lines)
- ✅ Switched Sphinx → Material for MkDocs
- ✅ Renamed `jax-tdd` → `jax-validation` (all files)
- ✅ Created docs structure (7 development guides)
- ✅ Removed premature implementation (examples, tests)
- ✅ Verified stubs only in `src/opaque/core/`

**No Implementation Yet**: All `src/opaque/core/*.py` are stubs with `NotImplementedError`

---

## Critical Reminders

1. **No premature implementation**: Tests first, always
2. **JAX-Privacy at `../jax_privacy`**: Use for reference and validation
3. **Security-critical**: DP guarantees depend on correct implementation
4. **Mark JAX tests**: Use `@pytest.mark.jax_validation` (optional group)
5. **Fail-fast**: Prefer exceptions over silent failures

---

## Documentation Structure

All human-facing docs in [docs/](docs/):
- **Getting Started**: Installation, quickstart
- **User Guide**: DP basics, clipping, LoRA
- **API Reference**: Auto-generated from docstrings
- **Development**: Architecture, TDD, decisions, plans
- **Examples**: Linear regression (to be created)

This file is **agent-only** - users should read [README.md](README.md) or [docs/](docs/).
